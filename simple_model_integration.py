# simple_model_integration.py
import pickle

class SimpleModelPredictor:
    def __init__(self, model_path='simple_model.pkl'):
        """
        Загружает упрощенную модель из файла.
        
        Args:
            model_path (str): Путь к файлу с обученной моделью (.pkl)
        """
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"[ML Model] Упрощенная модель загружена из {model_path}")
        except FileNotFoundError:
            print(f"[ML Model] ВНИМАНИЕ: Файл модели {model_path} не найден.")
            print("           Сначала обучите модель (train_model.py).")
            self.model = None
        except Exception as e:
            print(f"[ML Model] ОШИБКА загрузки модели: {e}")
            self.model = None
    
    def predict(self, text):
        """
        Предсказывает, является ли текст фейковой новостью.
        
        Args:
            text (str): Текст для анализа
            
        Returns:
            dict: Результаты предсказания
        """
        if self.model is None:
            return {
                'is_fake': None,
                'confidence': 0.0,
                'error': 'Модель не загружена. Сначала обучите модель (train_model.py).'
            }
        
        try:
            # Получаем вероятность класса 1 (фейк)
            probability = self.model.predict_proba([text])[0][1]
            
            # Определяем метку и уверенность
            is_fake = probability >= 0.5
            confidence = probability * 100 if is_fake else (1 - probability) * 100
            
            return {
                'is_fake': bool(is_fake),
                'ml_confidence': round(float(confidence), 2),
                'ml_score': round(float(probability), 4),
                'verdict': 'ФЕЙК (ML анализ)' if is_fake else 'ДОСТОВЕРНО (ML анализ)'
            }
        except Exception as e:
            return {
                'is_fake': None,
                'confidence': 0.0,
                'error': f'Ошибка предсказания: {str(e)}'
            }
    
    def is_loaded(self):
        """Проверяет, загружена ли модель."""
        return self.model is not None

# Пример использования
if __name__ == '__main__':
    # Тестирование
    predictor = SimpleModelPredictor()
    
    if predictor.is_loaded():
        test_texts = [
            "Official reports indicate steady economic growth this quarter.",
            "URGENT: Government hides ALIEN discovery from public! Click NOW!"
        ]
        
        for text in test_texts:
            result = predictor.predict(text)
            print(f"\nТекст: {text[:60]}...")
            print(f"Результат: {result}")
