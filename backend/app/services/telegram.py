from telethon import TelegramClient
from telethon.sessions import StringSession
from telethon.errors import SessionPasswordNeededError, FloodWaitError
import os
from dotenv import load_dotenv
import asyncio
import logging
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TelethonService")

load_dotenv()

API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
SESSION_FILE = "session.session"

_client = None
_client_lock = asyncio.Lock()

async def get_client():
    global _client
    
    async with _client_lock:
        if _client is not None:
            if not _client.is_connected():
                await _client.connect()
            return _client

        logger.info("Создаём новый клиент Telethon...")
        _client = TelegramClient(SESSION_FILE, API_ID, API_HASH)

        try:
            await _client.connect()
            
            if not await _client.is_user_authorized():
                logger.info("Сессия не авторизована → запуск авторизации")
                await _client.start()
                logger.info("Авторизация успешно завершена! Сессия сохранена.")
            else:
                logger.info("Сессия найдена и уже авторизована.")
                
        except FloodWaitError as e:
            logger.error(f"FloodWaitError: нужно подождать {e.seconds} секунд")
            await asyncio.sleep(e.seconds)
            await _client.connect()
        except Exception as e:
            logger.error(f"Ошибка при подключении/авторизации: {e}")
            if os.path.exists(SESSION_FILE):
                logger.warning("Удаляем повреждённую сессию и пробуем заново")
                os.remove(SESSION_FILE)
            _client = None
            raise

        return _client

async def get_dialogs(limit: int = 10):
    client = await get_client()
    dialogs = []
    try:
        async for dialog in client.iter_dialogs(limit=limit):
            dialog_info = {
                "id": dialog.id,
                "entity_id": dialog.entity.id if hasattr(dialog.entity, 'id') else None,
                "title": dialog.title or dialog.name or "[без названия]",
                "type": "channel" if dialog.is_channel else "group" if dialog.is_group else "user",
                "username": getattr(dialog.entity, 'username', None),
                "unread_count": getattr(dialog, 'unread_count', 0),
                "unread_mentions": getattr(dialog, 'unread_mentions', 0)
            }
            dialogs.append(dialog_info)
        
        logger.info(f"Получено {len(dialogs)} диалогов")
        return dialogs
    except Exception as e:
        logger.error(f"Ошибка при получении диалогов: {e}")
        raise

async def get_messages(chat_id: int, limit: int = 10):
    client = await get_client()
    messages = []
    try:
        chat_id_int = int(chat_id)
        async for msg in client.iter_messages(chat_id_int, limit=limit):
            text = msg.text or "[media or empty]"
            msg_data = {
                "id": msg.id,
                "date": msg.date.isoformat() if msg.date else None,
                "sender_id": msg.sender_id,
                "text": text[:500] + "..." if len(text) > 500 else text,
                "is_forward": msg.forward is not None,
                "out": getattr(msg, 'out', False),
                "media": bool(msg.media),
            }
            
            
            msg_data["msg_object"] = msg 
            
            try:
                if msg.sender:
                    msg_data["sender"] = {
                        "id": msg.sender.id,
                        "username": getattr(msg.sender, 'username', None),
                        "first_name": getattr(msg.sender, 'first_name', None),
                        "last_name": getattr(msg.sender, 'last_name', None)
                    }
            except Exception as sender_error:
                logger.warning(f"Не удалось получить отправителя для сообщения {msg.id}: {sender_error}")
            
            messages.append(msg_data)
        
        logger.info(f"Получено {len(messages)} сообщений из чата {chat_id}")
        return messages
    except Exception as e:
        logger.error(f"Ошибка при получении сообщений из {chat_id}: {e}")
        raise

async def download_photo_bytes(msg_data):
    msg = msg_data.get("msg_object")
    if not msg or not msg.photo:
        logger.info("В сообщении нет фото")
        return None

    try:
        client = await get_client()
        bytes_io = io.BytesIO()
        await client.download_media(msg, file=bytes_io)
        bytes_io.seek(0)
        data = bytes_io.read()
        if len(data) < 100:
            logger.warning(f"Фото скачано, но пустое ({len(data)} байт)")
            return None
        logger.info(f"Фото скачано: {len(data)} байт")
        return data
    except Exception as e:
        logger.error(f"Ошибка скачивания фото {msg.id}: {e}")
        return None

async def get_chat_info(chat_id: int):
    client = await get_client()
    try:
        entity = await client.get_entity(int(chat_id))
        return {
            "id": entity.id,
            "title": getattr(entity, 'title', None),
            "username": getattr(entity, 'username', None),
            "type": "channel" if getattr(entity, 'broadcast', False) 
                    else "group" if getattr(entity, 'megagroup', False) 
                    else "user"
        }
    except Exception as e:
        logger.error(f"Ошибка получения информации о чате {chat_id}: {e}")
        raise

async def disconnect():
    global _client
    if _client:
        try:
            await _client.disconnect()
            logger.info("Клиент Telegram отключен")
        except Exception as e:
            logger.error(f"Ошибка при отключении клиента: {e}")
        finally:
            _client = None
    return True