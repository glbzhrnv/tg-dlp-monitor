import sqlite3
from datetime import datetime
from clickhouse_driver import Client
import logging
import json

logger = logging.getLogger("DB")

# SQLite для источников и обработанных сообщений
SQLITE_DB = "monitor.db"

def init_sqlite():
    conn = sqlite3.connect(SQLITE_DB)
    c = conn.cursor()
    
    # Таблица источников
    c.execute('''CREATE TABLE IF NOT EXISTS sources
                 (chat_id INTEGER PRIMARY KEY,
                  title TEXT NOT NULL,
                  type TEXT NOT NULL,
                  added_at TEXT NOT NULL)''')
    
    # Таблица обработанных сообщений
    c.execute('''CREATE TABLE IF NOT EXISTS processed_messages
                 (chat_id INTEGER,
                  message_id INTEGER,
                  processed_at TEXT NOT NULL,
                  PRIMARY KEY (chat_id, message_id))''')
    
    conn.commit()
    conn.close()
    logger.info("SQLite таблицы sources и processed_messages готовы")

def add_source(chat_id: int, title: str, type_: str):
    conn = sqlite3.connect(SQLITE_DB)
    c = conn.cursor()
    added_at = datetime.now().isoformat()
    c.execute("INSERT OR REPLACE INTO sources VALUES (?, ?, ?, ?)",
              (chat_id, title, type_, added_at))
    conn.commit()
    conn.close()
    logger.info(f"Источник добавлен: {chat_id} ({title})")

def remove_source(chat_id: int):
    conn = sqlite3.connect(SQLITE_DB)
    c = conn.cursor()
    c.execute("DELETE FROM sources WHERE chat_id = ?", (chat_id,))
    conn.commit()
    conn.close()
    logger.info(f"Источник удалён: {chat_id}")

def get_sources():
    conn = sqlite3.connect(SQLITE_DB)
    c = conn.cursor()
    c.execute("SELECT chat_id, title, type, added_at FROM sources ORDER BY added_at DESC")
    rows = c.fetchall()
    conn.close()
    return [{"chat_id": r[0], "title": r[1], "type": r[2], "added_at": r[3]} for r in rows]

def is_message_processed(chat_id: int, message_id: int) -> bool:
    """Проверяет, было ли сообщение уже обработано"""
    conn = sqlite3.connect(SQLITE_DB)
    c = conn.cursor()
    c.execute("SELECT 1 FROM processed_messages WHERE chat_id = ? AND message_id = ?", (chat_id, message_id))
    result = c.fetchone()
    conn.close()
    return result is not None

def mark_message_processed(chat_id: int, message_id: int):
    """Отмечает сообщение как обработанное"""
    conn = sqlite3.connect(SQLITE_DB)
    c = conn.cursor()
    processed_at = datetime.now().isoformat()
    c.execute("INSERT OR IGNORE INTO processed_messages VALUES (?, ?, ?)", 
              (chat_id, message_id, processed_at))
    conn.commit()
    conn.close()
    logger.info(f"Сообщение {message_id} в чате {chat_id} отмечено как обработанное")

# ClickHouse для алертов
ch_client = Client(host='localhost', port=9000, database='monitor')

def init_clickhouse():
    ch_client.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            chat_id Int64,                      
            message_id UInt64,
            sx_score Float32,
            rubert_score Float32,
            text_preview String,
            leaks String,
            ocr_text String,
            sender String,
            date DateTime64(3),
            created_at DateTime64(3) DEFAULT now64()
        ) ENGINE = MergeTree()
        ORDER BY (created_at, chat_id)
        PARTITION BY toYYYYMM(created_at)
    ''')
    logger.info("ClickHouse таблица alerts готова")

def save_alert(alert_data):
    try:
        ch_client.execute('''
            INSERT INTO alerts (
                chat_id, message_id, sx_score, rubert_score, text_preview,
                leaks, ocr_text, sender, date
            ) VALUES
        ''', [alert_data])
        logger.info(f"Алерт сохранён в ClickHouse: message_id={alert_data.get('message_id')}")
    except Exception as e:
        logger.error(f"Ошибка сохранения алерта в ClickHouse: {e}", exc_info=True)

def get_recent_alerts(limit=50, filter_by_sources=False):
    try:
        query = '''
            SELECT chat_id, message_id, sx_score, rubert_score, text_preview,
                   leaks, ocr_text, sender, date, created_at
            FROM alerts
            ORDER BY created_at DESC
            LIMIT %(limit)s
        '''
        params = {'limit': limit}
        
        if filter_by_sources:
            sources = get_sources()
            if sources:
                chat_ids = [s["chat_id"] for s in sources]
                query = '''
                    SELECT chat_id, message_id, sx_score, rubert_score, text_preview,
                           leaks, ocr_text, sender, date, created_at
                    FROM alerts
                    WHERE chat_id IN %(chat_ids)s
                    ORDER BY created_at DESC
                    LIMIT %(limit)s
                '''
                params['chat_ids'] = tuple(chat_ids)
            else:
                return []
        
        rows = ch_client.execute(query, params)
        
        return [
            {
                "chat_id": r[0],
                "message_id": r[1],
                "sx_score": float(r[2]),
                "rubert_score": float(r[3]),
                "text_preview": r[4],
                "leaks": json.loads(r[5]) if r[5] else [],
                "ocr_text": r[6],
                "sender": json.loads(r[7]) if r[7] else {},
                "date": r[8],
                "created_at": r[9]
            } for r in rows
        ]
    except Exception as e:
        logger.error(f"Ошибка чтения алертов из ClickHouse: {e}", exc_info=True)
        return []

# Инициализация при запуске
init_sqlite()
init_clickhouse()