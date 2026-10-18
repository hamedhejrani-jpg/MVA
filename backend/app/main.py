from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import re
import uuid
import os
import urllib.request
import urllib.error
import asyncio

from fastapi import FastAPI, HTTPException, Request, Response
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
    model: str = ""
    address: str = ""
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


def normalize_spoken_digits(value: str) -> str:
    """Convert Persian digit words spoken one by one to ASCII digits."""
    text = (value or "").replace("ي", "ی").replace("ك", "ک").replace("\u200c", " ").strip()
    words = {"صفر":"0", "یک":"1", "اول":"1", "دو":"2", "سه":"3", "چهار":"4",
             "پنج":"5", "شش":"6", "شیش":"6", "هفت":"7", "هشت":"8", "نه":"9"}
    tokens = re.findall(r"[آ-ی]+|[0-9۰-۹٠-٩]+", text)
    if tokens and all(token in words or re.fullmatch(r"[0-9۰-۹٠-٩]+", token) for token in tokens):
        return "".join(words.get(token, normalize_digits(token)) for token in tokens)
    return normalize_digits(text)


def valid_mobile(value: str) -> bool:
    number = normalize_spoken_digits(value)
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




class SpeechSynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


@app.post("/api/speech/synthesize")
def synthesize_speech(payload: SpeechSynthesisRequest) -> Response:
    token = os.getenv("OPENAI_API_KEY", "").strip()
    if not token:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured.")
    body = json.dumps({
        "model": "gpt-4o-mini-tts",
        "voice": "alloy",
        "input": payload.text.strip(),
        "instructions": "Speak naturally and clearly in Persian (Farsi), with an Iranian Persian accent. Keep a warm, professional call-center tone. Read numbers clearly.",
        "response_format": "mp3",
    }, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/audio/speech",
        data=body,
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            audio = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise HTTPException(status_code=502, detail="OpenAI API key is invalid or lacks speech access.") from None
        if exc.code == 429:
            raise HTTPException(status_code=503, detail="OpenAI API quota or rate limit reached.") from None
        raise HTTPException(status_code=502, detail="Persian speech generation failed.") from None
    except (urllib.error.URLError, TimeoutError):
        raise HTTPException(status_code=502, detail="Could not connect to the speech service.") from None
    if not audio:
        raise HTTPException(status_code=502, detail="The speech service returned empty audio.")
    return Response(content=audio, media_type="audio/mpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/speech/status")
def speech_status() -> dict[str, Any]:
    configured = bool(os.getenv("OPENAI_API_KEY", "").strip())
    return {
        "configured": configured,
        "provider": "OpenAI",
        "model": "gpt-4o-mini-transcribe",
        "message": "Speech transcription uses the OpenAI API key.",
        "stores_audio": False,
    }


def _transcribe_audio(audio: bytes, content_type: str, token: str) -> dict[str, Any]:
    boundary = "----MVAOpenAIBoundary7MA4YWxk"
    filename = {
        "audio/webm": "recording.webm",
        "audio/ogg": "recording.ogg",
        "audio/mp4": "recording.mp4",
        "audio/mpeg": "recording.mp3",
        "audio/wav": "recording.wav",
        "audio/x-wav": "recording.wav",
    }.get(content_type, "recording.webm")

    parts = [
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="model"\r\n\r\n'
        "gpt-4o-mini-transcribe\r\n",
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="language"\r\n\r\n'
        "fa\r\n",
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="prompt"\r\n\r\n'
        "گفتار فارسی محاوره‌ای ایران است. نام افراد را به فارسی بنویس و متن را ترجمه نکن. اصطلاحات و مدل‌های محصول را دقیق حفظ کن. مدل‌های تلویزیون ممکن است ترکیبی از اعداد و حروف لاتین باشند؛ مثلاً TCL 65C755 را به‌صورت 65C755 بنویس، نه «سی 30» یا تبدیل مشابه. عبارت‌های رایج: میثم، حامد، تاجیک، هجرانی، آراسته؛ نصب دیواری، روی دیوار، زمینی، روی میز، روی پایه. اگر مدل دقیق واضح نیست، نزدیک‌ترین متن شنیده‌شده را بنویس و برای تأیید سؤال کن.\r\n",
    ]
    body = "".join(parts).encode("utf-8")
    body += (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    body += audio
    body += f"\r\n--{boundary}--\r\n".encode("utf-8")

    request = urllib.request.Request(
        "https://api.openai.com/v1/audio/transcriptions",
        data=body,
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise HTTPException(
                status_code=502,
                detail="OpenAI API key is invalid or lacks access."
            ) from None
        if exc.code == 429:
            raise HTTPException(
                status_code=503,
                detail="OpenAI API quota or rate limit reached."
            ) from None
        if exc.code in (400, 413, 415, 422):
            raise HTTPException(
                status_code=422,
                detail="Audio format or request was rejected. Please record again."
            ) from None
        raise HTTPException(
            status_code=502,
            detail="Speech transcription service returned an error."
        ) from None
    except (urllib.error.URLError, TimeoutError, ValueError):
        raise HTTPException(
            status_code=502,
            detail="Could not connect to the speech transcription service."
        ) from None

    transcript = str(result.get("text", "")).strip() if isinstance(result, dict) else ""
    if not transcript:
        raise HTTPException(
            status_code=502,
            detail="No speech was detected. Please speak clearly and try again."
        )
    return {"text": transcript[:2000], "provider": "openai", "stored": False}


@app.post("/api/speech/transcribe")
async def transcribe_speech(request: Request) -> dict[str, Any]:
    token = os.getenv("OPENAI_API_KEY", "").strip()
    if not token:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured in the server environment."
        )

    content_type = request.headers.get(
        "content-type", "audio/webm"
    ).split(";")[0].strip().lower()

    allowed_types = {
        "audio/webm", "audio/ogg", "audio/mp4",
        "audio/mpeg", "audio/wav", "audio/x-wav"
    }
    if content_type not in allowed_types:
        raise HTTPException(status_code=415, detail="Unsupported audio format.")

    audio = await request.body()
    if not audio:
        raise HTTPException(status_code=400, detail="Received an empty audio file.")
    if len(audio) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio must be smaller than 10 MB.")

    return await asyncio.to_thread(
        _transcribe_audio, audio, content_type, token
    )


@app.post("/api/validate")
def validate_request(payload: InstallationRequest) -> dict[str, Any]:
    issues = validation_issues(payload)
    return {"valid": not issues, "issues": issues}


def _ai_conversation_turn(message: str, current_data: dict[str, Any]) -> dict[str, Any]:
    token = os.getenv("OPENAI_API_KEY", "").strip()
    if not token:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured.")
    instructions = (
        "You are a Persian-only Iranian Farsi call-center assistant collecting a TV installation request. "
        "STRICT LANGUAGE RULE: understand and respond only to Persian (Farsi). Every reply must be entirely in Persian. "
        "Do not reject an utterance merely because speech recognition may misspell a Persian word or name. "
        "Treat imperfect Persian transcripts as potential answers. If a transcript plausibly represents Persian speech, extract what is reasonably clear. "
        "For a likely customer name, store the recognized Persian words in full_name even if spelling is uncertain, then ask the customer to confirm the spelling in Persian. "
        "Ask for repetition only when the utterance is genuinely unintelligible or clearly unrelated; never classify a plausible Persian name as a foreign language. "
        "Keep all assistant replies in Persian. Preserve product model codes, phone numbers, and serial numbers when clearly stated. "
        "Extract only values clearly stated; never guess or invent. Merge with existing values and never erase them, except when the user explicitly rejects a value such as a name. "
        "If one utterance includes multiple answers, fill all matching fields. Ask only for the next missing item. "
        "For installation type, دیواری means دیواری/روی دیوار/به دیوار; زمینی means زمینی/روی میز/روی پایه/پایه‌دار/روی میز تلویزیون. "
        "Treat these everyday Persian phrases as clear answers, not as a reason to repeat the question. Normalize install_type to exactly دیواری or زمینی. "
        "Ask the installation question exactly as «نوع نصب را می‌خواهید دیواری باشد یا روی میز/پایه؟» if still missing. "
        "When all fields are present, summarize in Persian and ask for explicit confirmation. Set confirmed=true only if the user explicitly confirms the final summary. "
        "Return JSON only with keys fields, reply, confirmed. fields must contain the full merged object. "
        "Allowed fields are exactly the keys in current_fields. Keep replies brief and natural in Persian."
    )
    body = {
        "model": os.getenv("MVA_CONVERSATION_MODEL", "gpt-4o-mini"),
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": json.dumps({
                "current_fields": current_data,
                "latest_utterance": message,
            }, ensure_ascii=False)},
        ],
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        parsed = json.loads(result["choices"][0]["message"]["content"])
        if not isinstance(parsed, dict) or not isinstance(parsed.get("fields"), dict):
            raise ValueError("Invalid AI result")
        return parsed
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise HTTPException(status_code=502, detail="OpenAI API key is invalid or lacks chat access.") from None
        if exc.code == 429:
            raise HTTPException(status_code=503, detail="OpenAI API quota or rate limit reached.") from None
        raise HTTPException(status_code=502, detail="AI could not process this answer.") from None
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError):
        raise HTTPException(status_code=502, detail="AI response could not be processed. Please repeat your answer.") from None


@app.post("/api/conversation/message")
def conversation_message(payload: ConversationMessage) -> dict[str, Any]:
    text = payload.message.strip()
    session_id = payload.session_id or str(uuid.uuid4())
    if text.lower() in ("شروع", "شروع مجدد", "از اول", "reset"):
        state = {"step": 0, "data": {"product": "تلویزیون"}}
        SESSIONS[session_id] = state
        return {
            "session_id": session_id,
            "reply": "سلام! نام و نام خانوادگی؟ اگر آماده است، مدل تلویزیون را هم بگویید.",
            "done": False,
            "data": state["data"],
        }

    state = SESSIONS.setdefault(session_id, {"step": 0, "data": {"product": "تلویزیون"}})
    data = state["data"]
    normalized_text = text.replace("ي", "ی").replace("ك", "ک").replace("؟", "").replace("?", "").replace(".", "").replace("،", "").strip()
    negative_answers = {"نه", "خیر", "درست نیست", "اشتباهه", "اشتباه است", "من نیستم", "نه درست نیست"}
    if state.get("awaiting_name_confirmation"):
        # If the customer rejects the proposed name and says the correction in the same turn,
        # use the words after the rejection as the new candidate instead of losing them.
        correction = normalized_text
        for prefix in ("نه", "خیر", "اشتباهه", "اشتباه است", "درست نیست", "من نیستم"):
            if correction.startswith(prefix):
                correction = correction[len(prefix):].strip(" ،,؛:ـ-")
                break
        if normalized_text in negative_answers or not correction:
            data.pop("full_name", None)
            state["awaiting_name_confirmation"] = False
            state["awaiting_name_input"] = True
            return {"session_id": session_id, "reply": "نام صحیح چیست؟", "done": False, "data": data}
        if correction != normalized_text and len(correction.split()) >= 2:
            data["full_name"] = correction
            state["awaiting_name_confirmation"] = True
            state["awaiting_name_input"] = False
            return {"session_id": session_id, "reply": f"نام صحیح «{correction}» است؟", "done": False, "data": data}
    if state.get("awaiting_name_input"):
        if len(normalized_text.split()) >= 2 and len(normalized_text) <= 100:
            data["full_name"] = normalized_text
            state["awaiting_name_input"] = False
            state["awaiting_name_confirmation"] = True
            return {"session_id": session_id, "reply": f"نام «{normalized_text}» درست است؟", "done": False, "data": data}
        return {"session_id": session_id, "reply": "نام صحیح چیست؟", "done": False, "data": data}
    parsed = _ai_conversation_turn(text, data)
    fields = parsed["fields"]
    allowed = {"full_name", "national_id", "phone", "mobile", "mobile2", "product", "serial_number", "install_type", "consent_confirmed"}
    for key in allowed:
        value = fields.get(key)
        if value is not None and str(value).strip():
            data[key] = value if key == "consent_confirmed" else str(value).strip()

    # Deterministic normalization for common spoken Persian answers.
    # These phrases are unambiguous and should not depend on the model's wording.
    utterance = text.replace("ي", "ی").replace("ك", "ک").replace("\u200c", " ")
    if any(term in utterance for term in ("دیواری", "روی دیوار", "به دیوار", "نصب به دیوار")):
        data["install_type"] = "دیواری"
    elif any(term in utterance for term in ("زمینی", "روی میز", "روی پایه", "پایه دار", "پایه‌دار", "روی میز تلویزیون")):
        data["install_type"] = "زمینی"

    install = str(data.get("install_type", "")).replace("ي", "ی").replace("ك", "ک")
    if "دیواری" in install or "دیوار" in install:
        data["install_type"] = "دیواری"
    elif "زمینی" in install or "پایه" in install or "میز" in install:
        data["install_type"] = "زمینی"

    reply = str(parsed.get("reply", "")).strip() or "متوجه شدم. لطفاً پاسخ را دوباره بفرمایید."
    if data.get("full_name") and "نام" in reply and any(word in reply for word in ("درست", "صحیح", "تأیید")):
        state["awaiting_name_confirmation"] = True
    if data.get("install_type") in ("دیواری", "زمینی") and any(x in reply for x in ("نوع نصب", "دیواری باشد یا")):
        reply = "ثبت شد. مورد بعدی را بفرمایید."
    if parsed.get("confirmed") is True:
        request_data = InstallationRequest(**{
            **{k: data.get(k, "") for k in ("full_name", "national_id", "phone", "mobile", "mobile2", "product", "serial_number", "install_type")},
            "consent_confirmed": True,
        })
        issues = validation_issues(request_data)
        if issues:
            return {"session_id": session_id, "reply": "برای تکمیل درخواست، لطفاً این موارد را اصلاح کنید: " + " ".join(i["message"] for i in issues), "done": False, "data": data, "issues": issues}
        data["consent_confirmed"] = True
        return {"session_id": session_id, "reply": "ممنون، اطلاعات تأیید شد. این نسخه فقط خروجی آزمایشی محلی می‌سازد و چیزی به مرکز تماس یا سامانه بیرونی ارسال نمی‌کند.", "done": True, "data": data}
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
