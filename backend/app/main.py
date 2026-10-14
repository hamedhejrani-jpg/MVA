from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import re
import uuid
import os
import urllib.request
import urllib.error

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
OUTPUTS = ROOT / "outputs"
app = FastAPI(title="MVA — دستیار نصب تلویزیون", version="0.3.0")
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


class InstallationRequest(BaseModel):
    full_name: str = ""
    national_id: str = ""
    phone: str = ""
    mobile: str = ""
    mobile2: str = ""
    product: str = "تلویزیون"
    serial_number: str = ""
    install_type: str = ""
    consent_confirmed: bool = False


class ExportPayload(BaseModel):
    request: InstallationRequest
    source: str = Field(default="local-mvp", max_length=40)


class ConversationMessage(BaseModel):
    session_id: str = ""
    message: str = Field(default="", max_length=2000)


# Conversation sessions exist only in process memory and disappear on restart.
SESSIONS: dict[str, dict[str, Any]] = {}


def normalize_digits(value: str) -> str:
    table = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
    return re.sub(r"\D", "", (value or "").translate(table))


def valid_mobile(value: str) -> bool:
    number = normalize_digits(value)
    if len(number) == 10 and number.startswith("9"):
        number = "0" + number
    return len(number) == 11 and number.startswith("09")


def validation_issues(data: InstallationRequest) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if not data.full_name.strip():
        issues.append({"field": "full_name", "message": "نام و نام خانوادگی وارد نشده است."})
    if len(normalize_digits(data.national_id)) != 10:
        issues.append({"field": "national_id", "message": "کد ملی باید ۱۰ رقم باشد."})
    mobiles = [x for x in (data.mobile, data.mobile2) if x.strip()]
    if data.phone.strip():
        if len(normalize_digits(data.phone)) < 8:
            issues.append({"field": "phone", "message": "شماره تلفن ثابت را بررسی کنید."})
        if not mobiles or not any(valid_mobile(x) for x in mobiles):
            issues.append({"field": "mobile", "message": "حداقل یک شماره همراه معتبر لازم است."})
    else:
        if len(mobiles) < 2:
            issues.append({"field": "phone", "message": "تلفن ثابت و همراه، یا دو شماره همراه لازم است."})
        if not all(valid_mobile(x) for x in mobiles):
            issues.append({"field": "mobile", "message": "شماره‌های همراه را بررسی کنید."})
    if data.product.strip() != "تلویزیون":
        issues.append({"field": "product", "message": "نسخه اول فقط برای نصب تلویزیون است."})
    if not data.serial_number.strip():
        issues.append({"field": "serial_number", "message": "شماره سریال محصول وارد نشده است."})
    if data.install_type not in ("دیواری", "زمینی"):
        issues.append({"field": "install_type", "message": "نوع نصب را دیواری یا زمینی انتخاب کنید."})
    return issues


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "project": "MVA", "version": app.version}


def ai_configuration() -> dict[str, Any]:
    # OpenAI-compatible endpoint; deliberately disabled unless explicitly configured.
    return {
        "configured": bool(os.getenv("MVA_AI_API_BASE_URL") and os.getenv("MVA_AI_API_KEY") and os.getenv("MVA_AI_MODEL")),
        "provider": os.getenv("MVA_AI_PROVIDER", "openai-compatible"),
        "model_set": bool(os.getenv("MVA_AI_MODEL")),
        "enabled": os.getenv("MVA_AI_ENABLED", "false").lower() == "true",
    }


@app.get("/api/ai/status")
def ai_status() -> dict[str, Any]:
    cfg = ai_configuration()
    return {
        "configured": cfg["configured"],
        "enabled": cfg["enabled"] and cfg["configured"],
        "provider": cfg["provider"],
        "message": "اتصال هوش مصنوعی فقط با تنظیم صریح کلید و نشانی سرویس فعال می‌شود؛ در غیر این صورت مکالمه آزمایشی محلی است.",
        "sends_customer_data": False,
    }


@app.post("/api/ai/test")
def ai_test() -> dict[str, Any]:
    """Safe connectivity test with a fixed, non-personal prompt only."""
    cfg = ai_configuration()
    if not (cfg["enabled"] and cfg["configured"]):
        return {"ok": False, "enabled": False, "message": "سرویس AI فعال نیست؛ هیچ درخواستی ارسال نشد."}
    base = os.environ["MVA_AI_API_BASE_URL"].rstrip("/")
    url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    body = {
        "model": os.environ["MVA_AI_MODEL"],
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": "فقط به فارسی و در یک جمله پاسخ بده. این آزمون فنی است؛ اطلاعات شخصی دریافت نکن."},
            {"role": "user", "content": "فقط بنویس: اتصال آزمایشی برقرار است."},
        ],
    }
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + os.environ["MVA_AI_API_KEY"]
    }, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            result = json.loads(response.read().decode("utf-8"))
        choices = result.get("choices", [])
        reply = choices[0].get("message", {}).get("content", "") if choices else ""
        return {"ok": bool(reply), "enabled": True, "reply": str(reply)[:500], "message": "آزمون با پیام ثابت و بدون اطلاعات مشتری انجام شد."}
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, IndexError) as exc:
        # Do not reveal URL, key, or raw provider response in the API output.
        return {"ok": False, "enabled": True, "message": "اتصال آزمون برقرار نشد. نشانی سرویس، مدل و کلید را بررسی کنید.", "error_type": type(exc).__name__}


@app.post("/api/validate")
def validate_request(payload: InstallationRequest) -> dict[str, Any]:
    issues = validation_issues(payload)
    return {"valid": not issues, "issues": issues}


@app.post("/api/conversation/message")
def conversation_message(payload: ConversationMessage) -> dict[str, Any]:
    text = payload.message.strip()
    session_id = payload.session_id or str(uuid.uuid4())
    state = SESSIONS.setdefault(session_id, {"step": 0, "data": {"product": "تلویزیون"}})
    if text.lower() in ("شروع", "شروع مجدد", "از اول", "reset"):
        state = {"step": 0, "data": {"product": "تلویزیون"}}
        SESSIONS[session_id] = state
        return {"session_id": session_id, "reply": "سلام، وقتتون بخیر. برای ثبت درخواست نصب تلویزیون چند مورد کوتاه می‌پرسم. لطفاً نام و نام خانوادگی مشتری را بفرمایید.", "done": False, "data": state["data"]}

    step, data = state["step"], state["data"]
    if step == 0:
        if not text:
            reply = "سلام، وقتتون بخیر. برای ثبت درخواست نصب تلویزیون چند مورد کوتاه می‌پرسم. لطفاً نام و نام خانوادگی مشتری را بفرمایید."
        else:
            data["full_name"] = text
            state["step"] = 1
            reply = "ممنون. حالا کد ملی ۱۰ رقمی مشتری را بفرمایید."
    elif step == 1:
        data["national_id"] = text
        state["step"] = 2
        reply = "لطفاً شماره تلفن ثابت و همراه را بفرمایید؛ یا اگر تلفن ثابت ندارید، دو شماره همراه وارد کنید. شماره‌ها را با ویرگول جدا کنید."
    elif step == 2:
        parts = [p.strip() for p in re.split(r"[,،;/|]+", text) if p.strip()]
        if len(parts) >= 2:
            first, second = parts[0], parts[1]
            if valid_mobile(first) and valid_mobile(second):
                data["mobile"], data["mobile2"], data["phone"] = first, second, ""
            else:
                data["phone"], data["mobile"] = first, second
        else:
            data["mobile"] = text
        state["step"] = 3
        reply = "محصول این درخواست تلویزیون است. اگر درست است بگویید «تأیید»؛ در غیر این صورت نام محصول را بفرمایید."
    elif step == 3:
        data["product"] = "تلویزیون" if text in ("تأیید", "تایید", "بله", "آره", "درسته") else text
        state["step"] = 4
        reply = "لطفاً شماره سریال تلویزیون را بفرمایید."
    elif step == 4:
        data["serial_number"] = text
        state["step"] = 5
        reply = "نوع نصب را مشخص کنید: دیواری یا زمینی (روی پایه)."
    elif step == 5:
        normalized = text.replace("ي", "ی").replace("ك", "ک")
        if "دیواری" in normalized or "دیوار" in normalized:
            data["install_type"] = "دیواری"
        elif "زمینی" in normalized or "پایه" in normalized:
            data["install_type"] = "زمینی"
        else:
            return {"session_id": session_id, "reply": "لطفاً یکی از این دو گزینه را بفرمایید: نصب دیواری یا زمینی (روی پایه).", "done": False, "data": data}
        state["step"] = 6
        reply = (f"لطفاً بررسی کنید: نام: {data.get('full_name','')}؛ کد ملی: {data.get('national_id','')}؛ "
                 f"تلفن ثابت: {data.get('phone') or 'ثبت نشده'}؛ همراه اول: {data.get('mobile','')}؛ "
                 f"همراه دوم: {data.get('mobile2') or 'ثبت نشده'}؛ محصول: {data.get('product','تلویزیون')}؛ "
                 f"سریال: {data.get('serial_number','')}؛ نوع نصب: {data.get('install_type','')}. اگر درست است بگویید تأیید.")
    else:
        if text in ("تأیید", "تایید", "بله", "درسته", "تأیید می‌کنم"):
            data["consent_confirmed"] = True
            request = InstallationRequest(**data)
            issues = validation_issues(request)
            if issues:
                return {"session_id": session_id, "reply": "چند مورد نیاز به اصلاح دارد: " + " ".join(x["message"] for x in issues), "done": False, "data": data, "issues": issues}
            return {"session_id": session_id, "reply": "اطلاعات تأیید شد. این نسخه فقط خروجی آزمایشی محلی می‌سازد و درخواست را به سامانه یا مرکز تماس ارسال نمی‌کند.", "done": True, "data": data}
        return {"session_id": session_id, "reply": "برای جلوگیری از اشتباه، لطفاً بگویید «تأیید» یا «شروع مجدد».", "done": False, "data": data}
    return {"session_id": session_id, "reply": reply, "done": False, "data": data}


@app.post("/api/export")
def export_request(payload: ExportPayload) -> dict[str, Any]:
    if not payload.request.consent_confirmed:
        raise HTTPException(status_code=400, detail="ابتدا اطلاعات را تأیید کنید.")
    issues = validation_issues(payload.request)
    if issues:
        raise HTTPException(status_code=422, detail=issues)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    request_id = str(uuid.uuid4())
    record = {"id": request_id, "created_at": datetime.now(timezone.utc).isoformat(), "project": "MVA",
              "source": payload.source, "status": "local_reviewed_not_submitted", "request": payload.request.model_dump()}
    (OUTPUTS / f"installation-request-{request_id}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "id": request_id, "status": record["status"], "message": "درخواست فقط به‌صورت محلی ذخیره شد و به سامانه بیرونی ارسال نشده است."}
