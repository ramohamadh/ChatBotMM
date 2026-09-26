# راهنمای اتصال gorest به سرویس چت‌بات

ChatBotMM یک **سرویس مستقل** است که با `chatbot serve` بالا می‌آید و gorest از طریق HTTP صدایش می‌زند. تاریخچه چت هر کاربر در PostgreSQL (دیتابیس `ai`، جدول `chat_history` — با اولین اتصال خودکار ساخته می‌شود) ذخیره می‌شود.

## بالا آوردن سرویس

```bash
chatbot serve --host 0.0.0.0 --port 8000
# مستندات تعاملی: http://<host>:8000/docs
```

اتصال به دیتابیس از متغیرهای محیطی یا فایل `.env` (کپی از `.env.example` — در
gitignore است تا پسورد وارد گیت نشود) خوانده می‌شود:
`CHATBOT_PG_HOST`، `CHATBOT_PG_PORT`، `CHATBOT_PG_USER`، `CHATBOT_PG_PASSWORD`، `CHATBOT_PG_DBNAME`، `CHATBOT_PG_SSLMODE`.
اگر پسورد تنظیم نشده یا دیتابیس در دسترس نباشد، سرویس بدون خطا به SQLite محلی برمی‌گردد و کار ادامه پیدا می‌کند.

## قرارداد API

### `POST /ask` — پرسیدن سوال

```json
{
  "question": "چجوری فاکتور اصلاحی بزنم؟",
  "uuid": "<user.UUID از میدل‌ور>",
  "taxpayer_id": "<Moadi UUID>",
  "session_id": "<شناسه گفتگو، اختیاری>"
}
```

- `uuid` همان شناسه کاربر لاگین‌شده است که **میدل‌ور Casdoor خودش در context گذاشته** (`userInfo["uuid"]`) — نیازی به کوئری دیتابیس نیست. (`user_id` هم به عنوان alias پذیرفته می‌شود.)
- همه فیلدهای هویتی اختیاری‌اند؛ ولی برای گزارش «کدام کاربر چه پرسیده» باید `uuid` ارسال شود.

پاسخ:

```json
{
  "answer": "…",
  "score": 1.0,
  "sources": ["03-rahnamaye-sodur-faktur.md"],
  "timings": {"search_s": 1.2, "answer_s": 25.0}
}
```

> ⚠️ **تایم‌اوت:** تولید جواب روی CPU ده‌ها ثانیه طول می‌کشد. کلاینت HTTP سمت gorest
> باید برای این endpoint تایم‌اوت **حداقل ۱۲۰ ثانیه** داشته باشد (پیش‌فرض ۱۰ ثانیه‌ی
> `httpClient` کافی نیست) و retry هم نزند.

### `GET /history?user_id=<uuid>&limit=50` — سوالات یک کاربر (یا همه)

```json
{
  "data": [
    {"asked_at": "2026-09-26T22:05:11", "question": "…", "answer": "…",
     "confidence": 98, "user_id": "…", "taxpayer_id": "…", "session_id": "…"}
  ],
  "meta": {"size": 1, "total_count": 42}
}
```

### `GET /users` — کدام کاربر چند سوال پرسیده

```json
{
  "data": [
    {"user_id": "…", "questions": 12,
     "first_asked_at": "2026-09-20T10:00:00", "last_asked_at": "2026-09-26T22:05:11"}
  ],
  "meta": {"total_count": 3}
}
```

### سایر endpoint ها

`GET /health` (برای readiness probe)، `GET /stats`، `POST /index` (بازسازی ایندکس پس از تغییر پایگاه دانش).

## سمت gorest — پیاده‌سازی شده ✅

ماژول چت‌بات در خود gorest پیاده‌سازی و تست شده است (`api/modules/chatbot/`):

| مسیر gorest | کار |
|---|---|
| `POST /v1/chatbot/ask` | سوال را با `userInfo["uuid"]` از میدل‌ور Casdoor به چت‌بات می‌فرستد (بدنه: `question` اجباری، `taxpayer_id` و `session_id` اختیاری) |
| `GET /v1/chatbot/history` | تاریخچه‌ی خود کاربر لاگین‌شده؛ با `?user_id=` تاریخچه‌ی کاربر دیگر (گزارش ادمین) |
| `GET /v1/chatbot/users` | کدام کاربر چند سوال پرسیده |

فایل‌ها:

- `config/config.go` → struct جدید `ChatbotConfig {Endpoint, UseSSL}` + فیلد `Chatbot` در `Config`؛ بخش `chatbot:` به هر سه فایل `config-*.yml` اضافه شده (پیش‌فرض `127.0.0.1:8000`).
- `api/modules/chatbot/{dto,service,controllers,routes}` — سرویس با کلاینت اختصاصی **۱۸۰ ثانیه‌ای** برای `/ask` (بدون retry) و کلاینت ۱۰ ثانیه‌ای برای history/users. متغیر محیطی `CHATBOT_ENDPOINT` روی کانفیگ اولویت دارد.
- روت‌ها در `api/routes/routes.go` زیر `protectedRoutes` (پشت `IntrospectionMiddleware`) ثبت شده‌اند — شناسه کاربر جعل‌شدنی نیست چون از body خوانده نمی‌شود.

### تست یکپارچه بدون بالا آوردن کل gorest

`cmd/chatbotdemo/main.go` همان روت‌ها و هندلرهای واقعی را با یک میدل‌ور شبیه‌ساز لاگین بالا می‌آورد — بدون Postgres/Redis/RabbitMQ/Casdoor؛ فقط با سرویس چت‌بات و دیتابیس `ai` کار دارد:

```bash
# سرویس ۱ — چت‌بات
chatbot serve --port 8000
# سرویس ۲ — دموی gorest (فقط ماژول چت‌بات)
cd gorest && CHATBOT_ENDPOINT=http://127.0.0.1:8000 go run ./cmd/chatbotdemo

curl -X POST http://127.0.0.1:8070/v1/chatbot/ask \
     -H 'Content-Type: application/json' \
     -d '{"question": "چجوری فاکتور بزنم؟"}'
```

## جدول دیتابیس

```sql
CREATE TABLE IF NOT EXISTS chat_history (
    id          BIGSERIAL PRIMARY KEY,
    asked_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    confidence  INTEGER,
    user_id     TEXT,        -- user.UUID از gorest
    taxpayer_id TEXT,        -- Moadi UUID
    session_id  TEXT
);
```

ایندکس روی `user_id` و `taxpayer_id` و `asked_at` وجود دارد؛ برای گزارش‌گیری مستقیم SQL هم آماده است:

```sql
-- سوالات هر کاربر
SELECT user_id, COUNT(*) FROM chat_history GROUP BY user_id ORDER BY 2 DESC;
```
