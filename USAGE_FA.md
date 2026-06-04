<div dir="rtl">

# راهنمای استفاده — ChatBotMM

> 🇬🇧 برای معرفی کامل پروژه به [README.md](README.md) و برای نسخه‌ی فارسی آن به
> [FAREADME.md](FAREADME.md) مراجعه کنید.

## ۰️⃣ نصب (یک‌بار)

<div dir="ltr">

```bash
python -m venv venv && source venv/bin/activate
pip install -e .          # نصب پکیج و دستور chatbot
```

</div>

سپس فایل‌های خود را در پوشه‌ی `data/docs/` بگذارید (PDF، DOCX، TXT یا MD).

## 🚀 راه‌اندازی سریع (یک دستور)

<div dir="ltr">

```bash
python main.py cli        # یا اگر نصب شده: chatbot cli
```

</div>

این دستور سه کار را پشت سر هم انجام می‌دهد و هر مرحله را اگر قبلاً انجام شده باشد
رد می‌کند:

۱. نصب وابستگی‌ها (در صورت نبودن)
۲. ایندکس کردن اسناد `data/docs/` (اگر ایندکس وجود نداشته باشد)
۳. ورود به حالت چت تعاملی

با `python main.py cli --skip-install` می‌توانید مرحله‌ی نصب را رد کنید.

## مراحل به‌صورت جداگانه

### 1️⃣ ایندکس کردن اسناد

<div dir="ltr">

```bash
chatbot index
```

</div>

**چه اتفاقی می‌افتد:** اسناد `data/docs/` خوانده، به قطعات کوچک تقسیم، برای هر
قطعه بردار (embedding) تولید و همه‌چیز در `data/vectorstore/` ذخیره می‌شود.

> **نکته:** بار اول ممکن است چند دقیقه طول بکشد (مدل‌ها دانلود می‌شوند).

### 2️⃣ پرسیدن سؤال

<div dir="ltr">

```bash
chatbot ask                          # حالت تعاملی (چند سؤال)
chatbot ask "این سند درباره چیست؟"   # یک سؤال
chatbot ask "سؤال شما" --context      # همراه با قطعات بازیابی‌شده
```

</div>

## همه‌ی دستورها

<div dir="ltr">

```bash
chatbot cli                # نصب + ایندکس + چت (یک‌مرحله‌ای)
chatbot index              # ایندکس کردن
chatbot index --force      # ایندکس مجدد (پاک‌کردن و ساخت دوباره)
chatbot rebuild            # معادل index --force
chatbot ask                # پرسیدن سؤال (تعاملی)
chatbot ask "سؤال شما"      # یک سؤال
chatbot ask "سؤال" --context  # با جزئیات بیشتر
```

</div>

> همه‌ی این دستورها با `python main.py <command>` و `python -m chatbot <command>`
> هم بدون نصب کار می‌کنند.

## استفاده در کد

<div dir="ltr">

```python
from chatbot import RAGPipeline

rag = RAGPipeline("data/docs")
rag.index_documents()

response = rag.ask("این PDF درباره چیست؟")
print(response["answer"])
print(f"تعداد منابع: {len(response['source_chunks'])}")
```

</div>

## 🆕 موتور پاسخ‌دهی هوشمند (Generative)

ربات به جای کپی‌کردن یک تکه از متن، با یک مدل زبانی محلی
(**Qwen2.5-1.5B-Instruct**) پاسخ را به **فارسی روان** تولید می‌کند. این مدل
چندزبانه است، فارسی را خوب می‌فهمد و کاملاً محلی (آفلاین) اجرا می‌شود.

### ⚠️ فقط بار اول به اینترنت نیاز است

بار اولی که سؤال می‌پرسید، مدل (حدود ۳ گیگابایت) از HuggingFace دانلود می‌شود.
برای این یک‌بار باید به اینترنت وصل باشید. پس از دانلود، مدل در
`~/.cache/huggingface/` ذخیره می‌شود و دفعات بعد آفلاین کار می‌کند.

### تنظیمات

در فایل [src/chatbot/config.py](src/chatbot/config.py):

- `USE_GENERATIVE = True` → موتور هوشمند (پیش‌فرض، پاسخ روان فارسی)
- `USE_GENERATIVE = False` → موتور استخراجی قدیمی (سریع‌تر ولی فقط تکه‌ای از متن)
- `GENERATIVE_MODEL` → اگر رم کم دارید مدل سبک‌تر `Qwen/Qwen2.5-0.5B-Instruct` را بگذارید

## نکات مهم

- ✅ بعد از اولین بار، ایندکس در `data/vectorstore/` ذخیره می‌شود
- ✅ دفعات بعد سریع‌تر است (نیازی به ایندکس مجدد نیست)
- ✅ اگر سند جدید اضافه کردید، دوباره `chatbot index` را اجرا کنید
- ✅ همه‌چیز محلی اجرا می‌شود (بعد از دانلود اولیه‌ی مدل‌ها، نیازی به اینترنت نیست)
- ✅ می‌توانید سؤالات را به فارسی یا انگلیسی بپرسید

</div>
