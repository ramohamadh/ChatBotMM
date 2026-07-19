<div dir="rtl">

# ChatBotMM — چت‌بات محلی فارسی/انگلیسی مبتنی بر RAG

یک سیستم کاملاً محلی «تولید پاسخ به‌کمک بازیابی» (RAG) با پایتون. اسناد را
می‌خواند، آن‌ها را به بردار تبدیل می‌کند، در یک ایندکس برداری FAISS ذخیره می‌کند و
با استفاده از یک **مدل زبانی مولّد محلی** به سؤالات پاسخ می‌دهد — پاسخ‌هایی روان و
طبیعی.

**از فارسی و انگلیسی پشتیبانی می‌کند!** 🇮🇷 🇬🇧

> 🇬🇧 The English version is available in [README.md](README.md).

## ویژگی‌ها

- **خواندن اسناد:** فایل‌های PDF، DOCX، TXT و MD
- **قطعه‌بندی هوشمند:** اندازه و هم‌پوشانی قابل تنظیم، همراه با نرمال‌سازی فارسی
- **بردارسازی چندزبانه:** مدل `intfloat/multilingual-e5-base` (بازیابی فارسی بسیار بهتر)
- **ذخیره‌سازی برداری:** FAISS با قابلیت ذخیره/بارگذاری و به‌روزرسانی تدریجی
- **بازیابی ترکیبی:** جست‌وجوی معنایی + کلیدواژه‌ای
- **پاسخ مولّد:** مدل زبانی محلی (`Qwen2.5-1.5B-Instruct`) پاسخ روان فارسی می‌نویسد — نه فقط یک تکه‌ی کپی‌شده. یک موتور استخراجی هم به‌عنوان جایگزین موجود است.
- **بسته‌بندی استاندارد:** ساختار `src/`، فایل `pyproject.toml`، دستور خط فرمان و تست‌ها

## نصب

<div dir="ltr">

```bash
# از ریشه‌ی پروژه
python -m venv venv && source venv/bin/activate
pip install -e .            # نصب پکیج و دستور chatbot
# برای توسعه (تست‌ها و linterها):
pip install -e ".[dev]"
```

</div>

اسناد خود را در پوشه‌ی `data/docs/` قرار دهید (PDF، DOCX، TXT یا MD).

## شروع سریع

سریع‌ترین راه — یک دستور که وابستگی‌ها را نصب، ایندکس را می‌سازد و چت را شروع می‌کند:

<div dir="ltr">

```bash
python main.py cli      # یا اگر نصب شده باشد: chatbot cli
```

</div>

دستور `cli` تکرارپذیر است: وابستگی‌ها را فقط در صورت نبودن نصب می‌کند، ایندکس را
فقط اگر وجود نداشته باشد می‌سازد، و سپس وارد حالت چت تعاملی می‌شود.

یا دستورها را جداگانه اجرا کنید:

<div dir="ltr">

```bash
# ۱. ایندکس کردن اسناد موجود در data/docs/ به data/vectorstore/
chatbot index

# ۲. پرسیدن سؤال (فارسی یا انگلیسی)
chatbot ask                                  # حالت تعاملی
chatbot ask "این سند درباره چیست؟"           # یک سؤال
chatbot ask "What is this about?" --context  # همراه با قطعات بازیابی‌شده

# ساخت دوباره‌ی ایندکس از ابتدا
chatbot rebuild        # یا: chatbot index --force
```

</div>

> همه‌ی دستورها بدون نصب هم کار می‌کنند: با شیمِ سورس `python main.py <command>`
> یا به‌صورت ماژول `python -m chatbot <command>`.
> رابط خط فرمان با [Typer](https://typer.tiangolo.com/) ساخته شده است.

> ⚠️ **اولین اجرا، مدل مولّد (حدود ۳ گیگابایت) را از HuggingFace دانلود می‌کند** و
> یک‌بار به اینترنت نیاز دارد. پس از آن کاملاً آفلاین کار می‌کند. راهنمای کامل در
> [USAGE_FA.md](USAGE_FA.md).

## دستورها

| دستور | کاری که انجام می‌دهد |
| --- | --- |
| `chatbot cli` | راه‌اندازی یک‌مرحله‌ای: نصب وابستگی‌ها ← ایندکس ← چت (هر مرحله اگر انجام شده باشد رد می‌شود). با `--skip-install` مرحله‌ی نصب نادیده گرفته می‌شود. |
| `chatbot index` | ایندکس کردن `data/docs/` در ذخیره‌ساز برداری. `--force` / `-f` از نو می‌سازد. |
| `chatbot rebuild` | ساخت دوباره‌ی ایندکس از ابتدا (معادل `index --force`). |
| `chatbot ask [QUESTION]` | پرسیدن سؤال؛ بدون `QUESTION` وارد حالت تعاملی می‌شود. `--context` / `-c` قطعات بازیابی‌شده را نشان می‌دهد. |
| `chatbot serve` | راه‌اندازی سرور REST API (با FastAPI). آدرس و پورت با `--host` / `--port` (پیش‌فرض `127.0.0.1:8000`). |

برای جزئیات کامل `chatbot --help` یا `chatbot <command> --help` را اجرا کنید.

## وب‌سرویس REST

دستور `chatbot serve` همان خط لوله‌ی RAG را روی HTTP در دسترس قرار می‌دهد
(مستندات تعاملی در `http://127.0.0.1:8000/docs`):

<div dir="ltr">

```bash
chatbot serve                 # یا: python main.py serve

curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "این سند در مورد چیست؟"}'
# {"answer": "...", "score": 1.0, "sources": ["file.pdf"], "timings": {...}}
```

</div>

| Endpoint | کاری که انجام می‌دهد |
| --- | --- |
| `POST /ask` | پاسخ به یک سؤال. بدنه: `{"question": "...", "return_context": false}` |
| `POST /index` | ایندکس (دوباره‌ی) `data/docs/`. بدنه: `{"force": false}` |
| `GET /stats` | آمار ایندکس |
| `GET /health` | وضعیت سرویس (`indexed` و تعداد قطعات) |

مدل پاسخ‌دهی یک‌بار هنگام راه‌اندازی سرور بارگذاری می‌شود، بنابراین اولین
درخواست هم سریع است. سؤال‌ها یکی‌یکی پاسخ داده می‌شوند (مدل CPU-محور است) و
درخواست‌های هم‌زمان در صف می‌مانند.

## استفاده در کد پایتون

<div dir="ltr">

```python
from chatbot import RAGPipeline

rag = RAGPipeline(docs_directory="data/docs", vectorstore_directory="data/vectorstore")
rag.index_documents()

response = rag.ask("موضوع اصلی سند چیست؟")
print(response["answer"])
print(f"Sources: {len(response['source_chunks'])} chunks")
```

</div>

## ساختار پروژه

<div dir="ltr">

```
.
├── main.py                 # شیم سورس: python main.py <command>
├── pyproject.toml          # بسته‌بندی، دستور خط فرمان، تنظیمات ruff/black/pytest
├── requirements.txt        # وابستگی‌ها
├── README.md  FAREADME.md  USAGE_FA.md
├── data/
│   ├── docs/               # اسناد خود را اینجا بگذارید
│   └── vectorstore/        # ایندکس FAISS تولیدشده (در gitignore)
├── src/chatbot/
│   ├── __init__.py         # صادرکردن RAGPipeline و __version__
│   ├── __main__.py         # امکان اجرای python -m chatbot
│   ├── config.py           # تنظیمات (مسیرها از طریق متغیر محیطی قابل تغییرند)
│   ├── cli.py              # رابط خط فرمان نازک با Typer (دستور chatbot)
│   ├── api.py              # وب‌سرویس REST با FastAPI (دستور chatbot serve)
│   ├── commands.py         # منطق دستورها مستقل از فریم‌ورک
│   ├── bootstrap.py        # نصب خودکار وابستگی‌ها + اجرای مجدد
│   └── rag/
│       ├── ingestion.py    # خواندن اسناد
│       ├── chunker.py      # قطعه‌بندی + normalize_persian()
│       ├── embeddings.py   # تولید بردار
│       ├── vectorstore.py  # ذخیره‌سازی FAISS
│       ├── retriever.py    # بازیابی ترکیبی
│       ├── generative_qa.py# پاسخ‌دهی با مدل مولّد محلی (پیش‌فرض)
│       ├── extractive_qa.py# موتور استخراجی جایگزین
│       └── pipeline.py     # کلاس سطح‌بالای RAGPipeline
└── tests/                  # مجموعه تست pytest (chunker, config, generative, pipeline, cli)
```

</div>

## پیکربندی

فایل [src/chatbot/config.py](src/chatbot/config.py) را ویرایش کنید:

- `USE_GENERATIVE` — مقدار `True` (پیش‌فرض) برای پاسخ‌های مولّد روان؛ `False` برای موتور استخراجی
- `GENERATIVE_MODEL` — پیش‌فرض `Qwen/Qwen2.5-1.5B-Instruct`؛ روی سیستم‌های کم‌رم از `Qwen/Qwen2.5-0.5B-Instruct` استفاده کنید
- `EMBEDDING_MODEL`، `CHUNK_SIZE`، `CHUNK_OVERLAP`، `TOP_K` و وزن‌های جست‌وجوی ترکیبی

مسیر داده‌ها را می‌توان با متغیرهای محیطی تغییر داد:
`CHATBOT_DATA_DIR`، `CHATBOT_DOCS_DIR`، `CHATBOT_VECTORSTORE_DIR`.

## نحوه‌ی کارکرد

۱. **خواندن** اسناد از `data/docs/` و استخراج متن و فراداده
۲. **قطعه‌بندی** به بخش‌های هم‌پوشان (با نرمال‌سازی فارسی)
۳. **بردارسازی** با مدل چندزبانه (محلی)
۴. **ذخیره** بردارها در ایندکس FAISS
۵. **بازیابی** قطعات مرتبط با جست‌وجوی ترکیبی (معنایی + کلیدواژه‌ای)
۶. **تولید** پاسخِ مبتنی بر متن با یک مدل زبانی محلی

## مدل‌ها

- **بردارسازی:** `intfloat/multilingual-e5-base` (۷۶۸ بُعدی، چندزبانه با فارسیِ قوی)
- **تولید پاسخ:** `Qwen/Qwen2.5-1.5B-Instruct` (چندزبانه، فارسیِ خوب)
- **جایگزین استخراجی:** `mrm8488/bert-multi-cased-finetuned-xquadv1`

مدل‌ها در اولین استفاده به‌صورت خودکار دانلود و در `~/.cache/huggingface/` ذخیره می‌شوند.

## توسعه

<div dir="ltr">

```bash
pip install -e ".[dev]"
pytest          # اجرای تست‌ها
ruff check .    # بررسی کد (lint)
black .         # قالب‌بندی کد
```

</div>

## رفع اشکال

- **پاسخ‌های فارسیِ نامفهوم یا بی‌کیفیت:** مطمئن شوید `USE_GENERATIVE = True` است. اگر ایندکس قدیمی از نسخه‌ی قبلی دارید، `chatbot rebuild` را اجرا کنید.
- **ایندکس پیدا نشد:** ابتدا `chatbot index` را اجرا کنید.
- **کمبود حافظه هنگام بارگذاری مدل:** مقدار `GENERATIVE_MODEL` را به `Qwen/Qwen2.5-0.5B-Instruct` تغییر دهید.
- **کندی اولین اجرا:** مدل مولّد (~۳ گیگابایت) در حال دانلود است؛ اجراهای بعدی سریع و آفلاین‌اند.

## مجوز

MIT — برای استفاده‌ی آموزشی و تولیدی، به‌همان‌صورت که هست ارائه شده است.

</div>
