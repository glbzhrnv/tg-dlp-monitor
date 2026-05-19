from fastapi import FastAPI, Query, HTTPException, Form, Request, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from app.services.telegram import get_dialogs, get_messages, get_chat_info, disconnect, download_photo_bytes
from app.services.ocr import extract_text_from_image
from app.services.analyzer import LeakAnalyzer
from app.services.db import save_alert, get_recent_alerts, add_source, remove_source, get_sources, is_message_processed, mark_message_processed
from dotenv import load_dotenv
import os
import logging
import re
import json
from datetime import datetime, timezone
from apscheduler.schedulers.asyncio import AsyncIOScheduler

load_dotenv()

SESSION_SECRET = os.getenv("SESSION_SECRET")
if not SESSION_SECRET:
    raise RuntimeError("SESSION_SECRET не задан в .env! Добавьте строку SESSION_SECRET=ваш_ключ")

print("SESSION_SECRET загружен из .env:", SESSION_SECRET[:10] + "...")  # отладка

analyzer = LeakAnalyzer()

app = FastAPI(
    title="TG-DLP Monitor",
    description="Система мониторинга утечек в Telegram",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

logger = logging.getLogger("MainAPI")

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

def is_authenticated(request: Request) -> bool:
    session_token = request.cookies.get("session_token")
    print("[AUTH CHECK] Cookies:", request.cookies)  # отладка
    print("[AUTH CHECK] Полученный token:", session_token)
    print("[AUTH CHECK] Ожидаемый SESSION_SECRET:", SESSION_SECRET)
    is_auth = session_token == SESSION_SECRET
    print("[AUTH CHECK] Результат:", is_auth)
    return is_auth


@app.get("/login", include_in_schema=False)
async def login_page(request: Request):
    print("[GET /login] Показываем форму")
    if is_authenticated(request):
        print("[GET /login] Уже авторизован → редирект на dashboard")
        return Response(headers={"Location": "/dashboard"}, status_code=302)
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    print("[POST /login] Попытка входа:", username, password)
    
    correct_username = "admin"
    correct_password = "admin123"

    if username == correct_username and password == correct_password:
        print("[POST /login] Успешный логин")
        
        # Создаем HTML страницу с JavaScript редиректом
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Redirecting...</title>
        </head>
        <body>
            <script>
                document.cookie = "session_token=" + encodeURIComponent("SESSION_SECRET_PLACEHOLDER") + "; path=/; max-age=86400";
                window.location.href = "/dashboard";
            </script>
        </body>
        </html>
        """.replace("SESSION_SECRET_PLACEHOLDER", SESSION_SECRET)
        
        return HTMLResponse(content=html_content, status_code=200)
    else:
        print("[POST /login] Неверные данные")
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Неверный логин или пароль"
        })

@app.get("/logout")  # Важно: GET, а не POST
async def logout():
    print("[GET /logout] Выход — удаляем cookie")
    
    # Создаем ответ с редиректом
    response = Response(
        content="",
        status_code=302,
        headers={"Location": "/login"}
    )
    
    # Удаляем cookie
    response.delete_cookie(
        key="session_token",
        path="/"
    )
    
    print("[GET /logout] Cookie удален, редирект на /login")
    return response

def require_auth(request: Request):
    print("[REQUIRE AUTH] Проверка авторизации для пути:", request.url.path)
    if not is_authenticated(request):
        print("[REQUIRE AUTH] Не авторизован — редирект на /login")
        raise HTTPException(status_code=303, headers={"Location": "/login"})
    print("[REQUIRE AUTH] Авторизован успешно")

# Определения функций (выше использования)
def safe_clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    allowed = re.compile(r'[^а-яА-ЯёЁa-zA-Z0-9\s\.,!?()\[\]{}+*/=\\|"\':;@#$%^&*+-]')
    text = allowed.sub('?', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) > 1000 or not text.strip():
        return "[OCR текст повреждён или слишком длинный]"
    return text

def parse_telegram_date(date_str: str) -> datetime:
    try:
        if '.' in date_str:
            date_str = date_str.split('.')[0]
        if '+' in date_str:
            date_str = date_str.split('+')[0]
        if 'Z' in date_str:
            date_str = date_str.replace('Z', '')
        date_str = date_str.replace('T', ' ')
        dt = datetime.fromisoformat(date_str)
        return dt.replace(tzinfo=timezone.utc)
    except Exception as e:
        logger.error(f"Ошибка парсинга даты '{date_str}': {e}")
        return datetime.now(timezone.utc)

@app.get("/")
async def root():
    return {"message": "TG-DLP Monitor запущен", "status": "ok"}

@app.get("/dialogs")
async def dialogs(limit: int = Query(10, ge=1, le=50)):
    try:
        chats = await get_dialogs(limit=limit)
        return {"status": "success", "chats": chats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/messages/{chat_id}")
async def messages(chat_id: int, limit: int = Query(10, ge=1, le=100)):
    try:
        msgs = await get_messages(chat_id, limit=limit)
        
        analyzed_messages = []
        alerts = []
        
        for msg in msgs:
            text = msg.get("text", "")
            media = msg.get("media", False)
            full_text = text
            ocr_text = ""

            if media:
                try:
                    photo_bytes = await download_photo_bytes(msg)
                    if photo_bytes:
                        logger.info(f"Фото скачано: {len(photo_bytes)} байт")
                        ocr_text = extract_text_from_image(photo_bytes)
                        if ocr_text:
                            full_text = text + " [OCR] " + ocr_text if text else ocr_text
                            logger.info(f"OCR извлёк: {ocr_text[:100]}...")
                except Exception as e:
                    logger.error(f"Ошибка OCR: {e}")

            leaks, rubert_score = analyzer.scan_message_for_leaks(full_text)
            sx_score = analyzer.calculate_sx(leaks, rubert_score, text=full_text)

            safe_ocr_text = safe_clean_text(ocr_text)

            message_processed = is_message_processed(chat_id, msg["id"])

            msg_response = {
                "id": msg["id"],
                "date": msg["date"],
                "sender_id": msg["sender_id"],
                "text": text,
                "is_forward": msg["is_forward"],
                "out": msg["out"],
                "media": media,
                "regex_leaks": leaks,
                "rubert_score": rubert_score,
                "sx_score": round(sx_score, 3),
                "alert": sx_score > 0.55 and not message_processed,
                "has_ocr": bool(ocr_text),
                "ocr_text": safe_ocr_text,
                "sender": msg.get("sender", {})
            }

            analyzed_messages.append(msg_response)
            
            if msg_response["alert"]:
                mark_message_processed(chat_id, msg["id"])
                
                parsed_date = parse_telegram_date(msg["date"])
                
                alert_data = {
                    "chat_id": chat_id,
                    "message_id": msg["id"],
                    "sx_score": sx_score,
                    "rubert_score": rubert_score,
                    "text_preview": full_text[:150] + "..." if len(full_text) > 150 else full_text,
                    "leaks": json.dumps(leaks),
                    "ocr_text": safe_ocr_text,
                    "sender": json.dumps(msg.get("sender", {})),
                    "date": parsed_date
                }
                save_alert(alert_data)
                
                alerts.append({
                    "message_id": msg["id"],
                    "sx_score": sx_score,
                    "rubert_score": rubert_score,
                    "text_preview": text[:150] + "..." if len(text) > 150 else text,
                    "leaks": leaks,
                    "has_ocr": msg_response["has_ocr"],
                    "ocr_text": safe_ocr_text,
                    "sender": msg.get("sender", {}),
                    "date": msg["date"]
                })
        
        return {
            "status": "success",
            "chat_id": chat_id,
            "messages_count": len(analyzed_messages),
            "messages": analyzed_messages,
            "alerts": alerts
        }
    except Exception as e:
        logger.error(f"Ошибка /messages/{chat_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/{chat_id}")
async def chat_info(chat_id: int):
    try:
        info = await get_chat_info(chat_id)
        return {"status": "success", "chat": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/disconnect")
async def disconnect_client():
    try:
        await disconnect()
        return {"status": "success", "message": "Клиент отключен"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Фронтенд с авторизацией (проверка внутри)

@app.get("/", include_in_schema=False)
@app.get("/dashboard", include_in_schema=False)
async def dashboard(request: Request):
    require_auth(request)
    alerts = get_recent_alerts(20, filter_by_sources=True)
    sources = get_sources()
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "alerts": alerts,
        "sources_count": len(sources)
    })

@app.get("/sources", include_in_schema=False)
async def sources_page(request: Request):
    require_auth(request)
    sources = get_sources()
    return templates.TemplateResponse("sources.html", {
        "request": request,
        "sources": sources
    })

@app.post("/add_source")
async def add_source_endpoint(
    request: Request,
    chat_id: int = Form(...),
    title: str = Form(...),
    type: str = Form(...)
):
    require_auth(request)

    try:
        add_source(chat_id, title, type)

        response = RedirectResponse(
            url="/sources?status=added",
            status_code=303
        )

        return response

    except Exception as e:
        logger.error(f"Ошибка добавления источника: {e}")

        return RedirectResponse(
            url="/sources?status=error",
            status_code=303
        )


@app.post("/remove_source")
async def remove_source_endpoint(
    request: Request,
    chat_id: int = Form(...)
):
    require_auth(request)

    try:
        remove_source(chat_id)

        response = RedirectResponse(
            url="/sources?status=removed",
            status_code=303
        )

        return response

    except Exception as e:
        logger.error(f"Ошибка удаления источника: {e}")

        return RedirectResponse(
            url="/sources?status=error",
            status_code=303
        )

@app.get("/alerts", include_in_schema=False)
async def alerts_page(request: Request):
    require_auth(request)
    filter_by_sources = request.query_params.get("filter_by_sources", "false").lower() == "true"
    alerts = get_recent_alerts(50, filter_by_sources=filter_by_sources)
    return templates.TemplateResponse("alerts.html", {
        "request": request,
        "alerts": alerts,
        "filter_by_sources": filter_by_sources
    })

# Фоновая проверка

scheduler = AsyncIOScheduler()

async def background_check_sources():
    sources = get_sources()
    for source in sources:
        chat_id = source["chat_id"]
        try:
            await messages(chat_id, limit=5)
            logger.info(f"Фоновая проверка чата {chat_id} завершена")
        except Exception as e:
            logger.error(f"Ошибка фоновой проверки {chat_id}: {e}")

scheduler.add_job(background_check_sources, 'interval', minutes=5)
scheduler.start()