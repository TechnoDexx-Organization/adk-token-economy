class EmotionAnalyzer:
    def __init__(self):
        print("⏳ Загрузка модели эмоционального анализа...")
        try:
            from transformers import pipeline
            self.pipeline = pipeline(
                "text-classification",
                model="Aniemore/rubert-tiny2-russian-emotion-detection",
            )
            print("✅ Модель эмоционального анализа загружена")
        except Exception as e:
            self.pipeline = None
            print(f"⚠️ Не удалось загрузить модель: {e}")

    def analyze(self, text: str) -> str:
        if not self.pipeline:
            return "neutral"
        try:
            result = self.pipeline(text, truncation=True, max_length=512)[0]
            return result["label"]
        except Exception:
            return "neutral"
