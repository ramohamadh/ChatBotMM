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

## سمت gorest

### ۱. کانفیگ (`config/config.go` و فایل‌های `config-*.yml`)

```go
type ChatbotConfig struct {
    Endpoint string
    UseSSL   bool
}
// در struct Config:  Chatbot ChatbotConfig
```

```yaml
chatbot:
  endpoint: <chatbot-host>:8000
  useSSL: false
```

### ۲. هندلر — شناسه کاربر مستقیم از میدل‌ور

```go
func AskChatbot(c *gin.Context) {
    userInfo := c.MustGet("userInfo").(map[string]interface{})
    userUUID := fmt.Sprintf("%v", userInfo["uuid"]) // از میدل‌ور Casdoor؛ بدون کوئری DB

    var req struct {
        Question   string `json:"question" binding:"required"`
        TaxpayerID string `json:"taxpayer_id"`
        SessionID  string `json:"session_id"`
    }
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }

    cfg := config.GetConfig()
    scheme := "http"
    if cfg.Chatbot.UseSSL {
        scheme = "https"
    }

    response, err := Invogate.GenericHTTPClientInvogate(c.Request.Context(), &Invogate.RequestConfig{
        Method:   http.MethodPost,
        BaseURL:  fmt.Sprintf("%s://%s", scheme, cfg.Chatbot.Endpoint),
        Endpoint: "/ask",
        Headers:  map[string]string{"Content-Type": "application/json"},
        Body: map[string]string{
            "question":    req.Question,
            "uuid":        userUUID,
            "taxpayer_id": req.TaxpayerID,
            "session_id":  req.SessionID,
        },
        Client: &http.Client{Timeout: 120 * time.Second}, // تولید جواب کند است
    })
    if err != nil {
        c.JSON(http.StatusBadGateway, gin.H{"error": "chatbot unavailable"})
        return
    }
    c.JSON(http.StatusOK, models.Response[any]{Data: response})
}
```

گزارش ادمین («کدوم کاربر چیا پرسیده») هم با همان الگو `GET /users` و `GET /history?user_id=…` را proxy کنید.

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
