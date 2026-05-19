import re
import logging
from typing import List, Dict, Any, Tuple, Set
from transformers import pipeline
import joblib
import pandas as pd
import os
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LeakAnalyzer")

class LeakAnalyzer:
    def __init__(self):
        # Whitelist
        self.whitelist = {
            "8.8.8.8", "8.8.4.4", "1.1.1.1", "1.0.0.1", "127.0.0.1", "0.0.0.0", "255.255.255.255",
            "localhost", "example.com", "test.com", "test@test.com", "admin@example.com",
            "192.168.0.1", "192.168.1.1"
        }

        # Загрузка RuBERT 
        self.rubert_classifier = None
        try:
            model_path = "C:/Users/minez/tg-dlp-monitor/backend/models/rubert-leak/best"
            self.rubert_classifier = pipeline("text-classification", model=model_path)
            logger.info(f"RuBERT загружен из {model_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки RuBERT: {e}. rubert_score будет 0.")
            self.rubert_classifier = None

        # Загрузка CatBoost 
        self.catboost_model = None
        catboost_path = "C:/Users/minez/tg-dlp-monitor/backend/models/catboost_leak.pkl"
        if os.path.exists(catboost_path):
            try:
                self.catboost_model = joblib.load(catboost_path)
                logger.info(f"CatBoost загружен из {catboost_path}")
            except Exception as e:
                logger.error(f"Ошибка загрузки CatBoost: {e}. Используем только RuBERT+regex.")
        else:
            logger.warning(f"CatBoost не найден по пути {catboost_path}. Используем только RuBERT+regex.")

        # Иерархия типов утечек 
        self.type_hierarchy = {
            'passport_rf': 10,
            'login_password_pair': 10,
            'pdn_auth_combo': 10,
            'fio_phone_combo': 10,
            'fio_address_combo': 10,
            'passport_fio_combo': 10,
            'jwt_token': 9,
            'private_key': 9,
            'seed_phrase': 9,
            'api_key': 9,
            'passport_series_number': 8,
            'passport_series_number_request': 8,
            'card_number': 8,
            'password': 7,
            'password_loose': 7,
            'config_fragment': 7,
            'inn': 6,
            'snils': 6,
            'passport_number': 6,
            'free_email_combo': 5,
            'ipv4_private': 5,
            'phone_ru': 4,
            'address': 4,
            'email': 3,
            'fio': 3,
            'birth_date': 3,
            'passport_mention': 2,  
        }
        
        # Группировка связанных типов
        self.type_groups = {
            'passport': ['passport_rf', 'passport_series_number', 'passport_series_number_request', 
                        'passport_number', 'passport_mention', 'passport_fio_combo'],
            'auth': ['login_password_pair', 'password', 'password_loose', 'jwt_token', 'api_key', 'private_key'],
            'person': ['fio', 'fio_phone_combo', 'fio_address_combo', 'birth_date'],
            'documents': ['inn', 'snils', 'passport_rf', 'passport_series_number', 'passport_number'],
            'contacts': ['email', 'phone_ru', 'address'],
            'financial': ['card_number', 'seed_phrase'],
        }
        
        # Карта для быстрого определения группы
        self.type_to_group = {}
        for group, types in self.type_groups.items():
            for t in types:
                self.type_to_group[t] = group

        self.patterns = [
            {
                "name": "passport_series_number_request",
                "pattern": r'(?i)(пришли|покажи|отправь|сфоткай|фото|скинь|серию и номер|серия и номер паспорта|серия номер паспорта|серия паспорта|номер паспорта|серия|номер)\s*(и|паспорта)?',
                "severity": 0.94,
                "context_boost": ["паспорт", "паспортные данные", "серия", "номер", "сотрудника", "работника", "мой", "твой", "документ", "паспорт", "серия паспорта", "номер паспорта"],
                "priority": 8
            },
            {
                "name": "passport_series_number",
                "pattern": r'\b\d{4}\s{0,2}[№#]?\s{0,2}\d{6}\b|\bсерия\s*\d{4}\s*(и|номер)?\s*\d{6}\b|\bномер\s*\d{6}\s*(и|серия)?\s*\d{4}\b',
                "severity": 0.96,
                "context_boost": ["паспорт", "паспортные данные", "серия", "номер", "сотрудник", "работник", "мой", "пришли", "покажи", "фото", "документа"],
                "priority": 8
            },
            {
                "name": "passport_rf",
                "pattern": r'(?i)(паспорт|паспортные данные|паспорт рф|паспорт гражданина рф)\s*[:=]?\s*["\']?(\d{4}\s?\d{6})["\']?',
                "severity": 1.0,
                "context_boost": ["мой", "сотрудника", "работника", "фио", "фамилия", "имя", "отчество", "дата рождения", "прописка", "адрес", "инн", "снилс", "логин", "пароль"],
                "priority": 10
            },
            {
                "name": "passport_number",
                "pattern": r'\b\d{4}\s?\d{6}\b',
                "severity": 0.93,
                "context_boost": ["паспорт", "серия", "номер", "пдн", "персональные", "сотрудник", "работник", "фио", "дата рождения", "инн", "снилс", "пришли", "покажи"],
                "priority": 6
            },
            {
                "name": "passport_mention",
                "pattern": r'(?i)\b(паспорт|паспортные данные|паспорт рф|серия паспорта|номер паспорта|серия|номер)\b',
                "severity": 0.45,
                "context_boost": ["мой", "сотрудника", "работника", "фио", "инн", "снилс", "логин", "пароль", "пришли", "покажи", "фото", "документа"],
                "priority": 2
            },

            # ИНН, СНИЛС
            {"name": "inn", "pattern": r'\b(\d{10}|\d{12})\b', "severity": 0.9, "context_boost": ["инн", "налоговый номер", "сотрудника", "работника", "фио"], "priority": 6},
            {"name": "snils", "pattern": r'\b\d{3}-\d{3}-\d{3}\s?\d{2}\b', "severity": 0.9, "context_boost": ["снилс", "страховой номер", "пенсионный", "сотрудника"], "priority": 6},

            # ФИО
            {
                "name": "fio",
                "pattern": r'(?i)\b([А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ][а-яё]{2,})\b',
                "severity": 0.45,
                "context_boost": ["сотрудник", "работник", "паспорт", "инн", "снилс", "адрес", "телефон"],
                "priority": 3
            },

            # Адрес
            {"name": "address", "pattern": r'(?i)(ул\.|улица|пр\.|проспект|пер\.|переулок|дом|д\.|кв\.|квартира|прописка|адрес|г\.|город)\s*[\w\s\d.,-]+', "severity": 0.8, "context_boost": ["мой", "сотрудника", "работника", "фио", "паспорт", "инн", "снилс"], "priority": 4},

            # Дата рождения
            {"name": "birth_date", "pattern": r'\b(0[1-9]|[12][0-9]|3[01])[./-]?(0[1-9]|1[0-2])[./-]?(\d{2,4})\b', "severity": 0.65, "context_boost": ["дата рождения", "др", "родился", "сотрудника"], "priority": 3},

            # Логины и пароли
            {"name": "login_password_pair", "pattern": r'(?i)(логин|login|username|user)\s*[:=]\s*["\']?([^"\s@]+)["\']?\s*[,;]?\s*(пароль|password|pass|pwd|secret|ключ|токен)\s*[:=]\s*["\']?([^"\s]{4,})["\']?', "severity": 1.0, "context_boost": ["мой", "введи", "забыл", "вход", "аккаунт"], "priority": 10},
            {"name": "password", "pattern": r'(?i)(пароль|password|pass|pwd|secret|key|credential|токен|auth|ключ|секрет)\s*[:=]\s*["\']?([^"\s]{4,})["\']?', "severity": 0.95, "context_boost": ["мой", "введи", "забыл", "вход", "логин", "аккаунт"], "priority": 7},
            {"name": "password_loose", "pattern": r'(?i)(пароль|password|pass|pwd|secret|key)\s+([^\s]{4,})', "severity": 0.8, "context_boost": ["мой", "введи", "забыл", "вход", "логин"], "priority": 7},

            # Самые опасные комбинации
            {"name": "pdn_auth_combo", "pattern": r'(?i)(паспорт|инн|снилс).{0,80}?(логин|пароль|login|password)', "severity": 1.0, "context_boost": ["мой", "сотрудника", "работника", "фио"], "priority": 10},
            {"name": "fio_phone_combo", "pattern": r'(?i)([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+).{0,80}?(\+7|8|7)\s?[\(\s-]?\d{3}[\)\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}', "severity": 1.0, "context_boost": ["сотрудник", "работник", "паспорт", "инн", "снилс", "адрес"], "priority": 10},
            {"name": "fio_address_combo", "pattern": r'(?i)([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+).{0,80}?(ул\.|улица|дом|кв\.|квартира|прописка|адрес)', "severity": 1.0, "context_boost": ["сотрудник", "работник", "паспорт", "инн", "снилс"], "priority": 10},
            {"name": "passport_fio_combo", "pattern": r'(?i)([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+).{0,80}?(паспорт|паспортные данные|серия|номер)', "severity": 1.0, "context_boost": ["сотрудник", "работник", "паспорт", "инн", "снилс"], "priority": 10},

            # Другие утечки
            {"name": "email", "pattern": r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', "severity": 0.65, "context_boost": ["почта", "email", "mail", "логин", "аккаунт"], "priority": 3},
            {"name": "free_email_combo","pattern": r'(?i)[a-zA-Z0-9_.+-]+@(mail\.ru|yandex\.ru|bk\.ru|gmail\.com|inbox\.ru|list\.ru|rambler\.ru|ya\.ru|proton\.me|tutanota\.com|hotmail\.com|outlook\.com)', "severity": 0.85, "context_boost": ["почта", "email", "mail", "логин", "пароль", "мой", "пришли", "скинь", "телефон", "номер", "фио", "паспорт", "инн", "снилс"], "priority": 5},
            {"name": "phone_ru", "pattern": r'(\+7|8|7)\s?[\(\s-]?\d{3}[\)\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}', "severity": 0.75, "context_boost": ["телефон", "номер", "мобильный", "сотрудника"], "priority": 4},
            {"name": "ipv4_private", "pattern": r'\b(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2[0-9]|3[0-1])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b', "severity": 0.85, "context_boost": ["ip", "адрес", "роутер", "wifi", "сеть"], "priority": 5},
            {"name": "jwt_token", "pattern": r'eyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*', "severity": 1.0, "context_boost": ["bearer", "token", "jwt", "auth"], "priority": 9},
            {"name": "api_key", "pattern": r'(?i)(api[-_]?key|access[-_]?key|secret[-_]?key|auth[-_]?token)\s*[:=]\s*["\']?([A-Za-z0-9+/=._-]{20,})["\']?', "severity": 1.0, "context_boost": ["api", "secret", "key"], "priority": 9},
            {"name": "private_key", "pattern": r'-----BEGIN (RSA|OPENSSH|DSA|EC|PGP) PRIVATE KEY-----[\s\S]*?-----END (RSA|OPENSSH|DSA|EC|PGP) PRIVATE KEY-----', "severity": 1.0, "context_boost": ["private", "key", "ssh", "rsa", "pgp"], "priority": 9},
            {"name": "card_number", "pattern": r'\b(?:\d[ -]*?){13,16}\b', "severity": 0.9, "context_boost": ["карта", "card", "visa", "mastercard", "сбер", "тинькофф", "альфа", "втб", "мир"], "priority": 8},
            {"name": "seed_phrase", "pattern": r'(?i)(seed|мнемоника|фразы|wallet|кошелёк)\s*[:=]?\s*["\']?([a-zA-Z]+(?:\s+[a-zA-Z]+){11,23})["\']?', "severity": 1.0, "context_boost": ["seed", "мнемоника", "крипто", "wallet"], "priority": 9},
            {"name": "config_fragment", "pattern": r'(\{.*(password|key|token|secret|pass|pwd|api_key).*:\s*["\']?[^"\']+["\']?\})|(- \w+: \w+)', "severity": 0.85, "context_boost": ["config", "yaml", "json", "settings", "env", "docker"], "priority": 7}
        ]

    def deduplicate_findings(self, findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Интеллектуальная дедупликация находок"""
        if not findings:
            return []
        
        # Сортируем по приоритету и уверенности
        findings.sort(key=lambda x: (-self.type_hierarchy.get(x["type"], 0), -x["confidence"]))
        
        unique_by_type = {}
        unique_by_group = defaultdict(list)
        
        for finding in findings:
            # Проверяем точное совпадение значения
            type_key = finding["type"]
            
            # Если тип уже есть, проверяем не лучше ли текущий
            if type_key in unique_by_type:
                existing = unique_by_type[type_key]
                # Оставляем с большей уверенностью
                if finding["confidence"] > existing["confidence"]:
                    unique_by_type[type_key] = finding
                continue
            
            # Проверяем группы
            group = self.type_to_group.get(type_key, type_key)
            is_duplicate = False
            
            for existing_finding in unique_by_group[group]:
                # Если значения пересекаются (одна утечка покрывает другую)
                if self._is_overlapping(finding, existing_finding):
                    is_duplicate = True
                    # Если текущая имеет больший приоритет, заменяем
                    if self.type_hierarchy.get(type_key, 0) > self.type_hierarchy.get(existing_finding["type"], 0):
                        unique_by_group[group].remove(existing_finding)
                        unique_by_group[group].append(finding)
                        # Обновляем в unique_by_type
                        if existing_finding["type"] in unique_by_type:
                            del unique_by_type[existing_finding["type"]]
                        unique_by_type[type_key] = finding
                    break
            
            if not is_duplicate:
                unique_by_group[group].append(finding)
                unique_by_type[type_key] = finding
        
        # Конвертируем обратно в список
        result = list(unique_by_type.values())
        
        # Сортируем по позиции в тексте для логичности
        result.sort(key=lambda x: x.get("position", 0))
        
        logger.info(f"Дедупликация: было {len(findings)} находок, стало {len(result)}")
        return result

    def _is_overlapping(self, finding1: Dict, finding2: Dict) -> bool:
        """Проверяет, перекрываются ли две находки"""
        pos1 = finding1.get("position", 0)
        pos2 = finding2.get("position", 0)
        
        # Если позиции не указаны, считаем по типу
        if pos1 == 0 or pos2 == 0:
            # Одинаковые или связанные типы
            group1 = self.type_to_group.get(finding1["type"], finding1["type"])
            group2 = self.type_to_group.get(finding2["type"], finding2["type"])
            return group1 == group2 and abs(finding1["confidence"] - finding2["confidence"]) < 0.3
        
        # Если позиции есть, проверяем расстояние
        distance = abs(pos1 - pos2)
        return distance < 50  

    def get_rubert_score(self, text: str) -> float:
        if not self.rubert_classifier or not text.strip():
            return 0.0
        try:
            result = self.rubert_classifier(text)[0]
            if result['label'] == 'LABEL_1':
                return round(result['score'], 3)
            else:
                return round(1 - result['score'], 3)
        except Exception as e:
            logger.error(f"RuBERT ошибка: {e}")
            return 0.0

    def scan_message_for_leaks(self, text: str | None) -> Tuple[List[Dict[str, Any]], float]:
        if not text or len(text.strip()) == 0:
            return [], 0.0

        findings = []
        lower_text = text.lower()
        
        # Собираем все совпадения
        raw_findings = []
        
        for p in self.patterns:
            for match in re.finditer(p["pattern"], text):
                value = match.group(0).strip()
                
                # Пропускаем слишком короткие значения
                if len(value) < 3 and p["name"] not in ["inn", "snils"]:
                    continue
                
                # Проверка белого списка
                if value in self.whitelist or value.lower() in self.whitelist:
                    continue
                
                # Вычисляем confidence
                confidence = p["severity"]
                
                context_start = max(0, match.start() - 100)
                context_end = min(len(text), match.end() + 100)
                context = lower_text[context_start:context_end]
                boost_count = sum(1 for word in p.get("context_boost", []) if word in context)
                confidence = min(1.0, confidence + boost_count * 0.15)
                
                # Маскируем значение
                if len(value) > 12:
                    masked = value[:6] + "..." + value[-6:]
                elif len(value) > 8:
                    masked = value[:4] + "..." + value[-4:]
                else:
                    masked = value
                
                raw_findings.append({
                    "type": p["name"],
                    "value": masked,
                    "confidence": round(confidence, 3),
                    "severity": p["severity"],
                    "position": match.start(),
                    "priority": p.get("priority", 5),
                    "full_value": value  # Сохраняем для дедупликации
                })
        
        # Дедупликация
        findings = self.deduplicate_findings(raw_findings)
        
        # Удаляем full_value перед возвратом
        for f in findings:
            f.pop("full_value", None)
        
        rubert_score = self.get_rubert_score(text)
        
        return findings, rubert_score

    def calculate_sx(self, leaks: List[Dict[str, Any]], rubert_score: float = 0.0, text: str = "") -> float:
        if not leaks and rubert_score < 0.5:
            return 0.0

        # Используем уникальные типы после дедупликации
        unique_types = set(l["type"] for l in leaks)
        num_types = len(unique_types)
        max_conf = max(l["confidence"] for l in leaks) if leaks else 0.0
        
        # Группируем по категориям
        categories = set()
        for leak in leaks:
            category = self.type_to_group.get(leak["type"], leak["type"])
            categories.add(category)
        num_categories = len(categories)

        # Весовые коэффициенты типов
        type_weights = {
            "login_password_pair": 0.45,
            "password": 0.38,
            "password_loose": 0.32,
            "passport_rf": 0.48,
            "passport_number": 0.42,
            "passport_mention": 0.25,
            "passport_series_number_request": 0.52,
            "passport_series_number": 0.50,
            "passport_fio_combo": 0.52,
            "pdn_auth_combo": 0.52,
            "fio_phone_combo": 0.52,
            "fio_address_combo": 0.52,
            "inn": 0.38,
            "snils": 0.38,
            "ipv4_private": 0.28,
            "jwt_token": 0.38,
            "api_key": 0.37,
            "private_key": 0.45,
            "free_email_combo": 0.40,
            "email": 0.18,
            "phone_ru": 0.25,
            "card_number": 0.42,
            "seed_phrase": 0.48,
            "config_fragment": 0.30,
            "address": 0.35,
            "fio": 0.25,
            "birth_date": 0.30
        }

        weighted_sum = 0.0
        for leak in leaks:
            weight = type_weights.get(leak["type"], 0.15)
            weighted_sum += leak["confidence"] * weight

        # Бонус за разнообразие категорий
        category_combo_bonus = 0.0
        if num_categories >= 3:
            category_combo_bonus = 0.40
        elif num_categories >= 2:
            category_combo_bonus = 0.25

        free_email_types = {"free_email_combo", "email"}
        has_free_email = any(l["type"] in free_email_types for l in leaks)
        if has_free_email and num_categories >= 2:
            category_combo_bonus += 0.15

        count_bonus = min(0.45, (len(leaks) - 1) * 0.09)

        regex_sx = (max_conf * 0.50) + (weighted_sum * 0.30) + category_combo_bonus + count_bonus
        regex_sx = min(1.0, max(0.0, regex_sx))

        # Усиленный вес для паспортных утечек
        has_passport = any("passport" in l["type"] for l in leaks)
        rubert_weight = 0.40 if has_passport else 0.35
        if not leaks:
            rubert_weight = 0.20  
        if len(text) < 20:
            rubert_weight *= 0.7

        # Специальный бонус
        document_bonus = 0.0
        if has_passport and rubert_score > 0.8:
            document_bonus = 0.22

        # Финальный sx
        final_sx = 0.50 * regex_sx + rubert_weight * rubert_score + document_bonus + 0.10 * (len(leaks) / 10.0)
        final_sx = min(1.0, max(0.0, final_sx))

        return float(round(final_sx, 3))


# Глобальный экземпляр
analyzer = LeakAnalyzer()
