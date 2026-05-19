# rubert_inference.py — загрузка и получение rubert_score

from transformers import pipeline

class RubertLeakDetector:
    def __init__(self, model_path="C:/Users/minez/tg-dlp-monitor/backend/models/rubert-leak/best"):
        print(f"Загружаю RuBERT из {model_path}...")
        self.classifier = pipeline("text-classification", model=model_path)
        print("RuBERT готов")

    def get_score(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0
        result = self.classifier(text)[0]
        # LABEL_1 — утечка, LABEL_0 — норм
        if result['label'] == 'LABEL_1':
            return round(result['score'], 3)
        else:
            return round(1 - result['score'], 3)

# Глобальный экземпляр (загружается один раз при импорте)
detector = RubertLeakDetector()

def get_rubert_score(text: str) -> float:
    return detector.get_score(text)