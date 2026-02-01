import re
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

logger = logging.getLogger(__name__)

class StyleAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_accuracy = 0.85
        
        # Списки слов для анализа
        self.clickbait_words = ['шок', 'сенсация', 'тайна', 'скандал', 'разоблачение', 
                               'ужас', 'чудо', 'невероятно', 'потрясающе', 'срочно',
                               'эксклюзив', 'секрет', 'правда', 'ложь', 'обман']
        
        self.certainty_words = ['точно', 'абсолютно', 'несомненно', 'безусловно',
                               'конечно', 'явно', 'очевидно', 'наверняка',
                               'гарантированно', 'стопроцентно', 'доказано']
        
        self.formal_words = ['сообщил', 'заявил', 'отметил', 'подчеркнул',
                            'указал', 'добавил', 'по данным', 'согласно']
        
        self.source_indicators = ['по данным', 'согласно', 'как сообщает',
                                 'по информации', 'по словам', 'по сведениям']
    
    def extract_features(self, text):
        """Извлечение признаков из текста"""
        features = {}
        text_lower = text.lower()
        
        # Базовые метрики
        features['length'] = len(text)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = re.findall(r'\b\w+\b', text_lower)
        
        features['sentence_count'] = len(sentences)
        features['word_count'] = len(words)
        
        if sentences:
            features['avg_words_per_sentence'] = len(words) / len(sentences)
        else:
            features['avg_words_per_sentence'] = 0
        
        # Эмоциональность
        try:
            blob = TextBlob(text)
            features['emotional_score'] = abs(blob.sentiment.polarity)
            features['subjectivity'] = blob.sentiment.subjectivity
        except:
            features['emotional_score'] = 0
            features['subjectivity'] = 0
        
        # Кликбейт
        clickbait_count = sum(1 for word in self.clickbait_words if word in text_lower)
        features['clickbait_score'] = min(clickbait_count / 3, 1.0)
        
        # Категоричность
        certainty_count = sum(1 for word in self.certainty_words if word in text_lower)
        features['certainty_score'] = min(certainty_count / 3, 1.0)
        
        # Формальность
        formal_count = sum(1 for word in self.formal_words if word in text_lower)
        features['formality_score'] = min(formal_count / 3, 1.0)
        
        # Источники
        source_count = sum(1 for word in self.source_indicators if word in text_lower)
        features['source_indicator_score'] = min(source_count / 2, 1.0)
        
        # Баланс
        balance_words = ['однако', 'тем не менее', 'с другой стороны']
        balance_count = sum(1 for word in balance_words if word in text_lower)
        features['balance_score'] = min(balance_count / 2, 1.0)
        
        # Пунктуация
        features['exclamation_count'] = text.count('!')
        features['exclamation_density'] = features['exclamation_count'] / max(len(sentences), 1)
        
        # Регистр
        caps_count = sum(1 for c in text if c.isupper())
        features['caps_ratio'] = caps_count / len(text) if len(text) > 0 else 0
        
        # Числа и проценты
        features['has_percentages'] = 1 if '%' in text or 'процент' in text_lower else 0
        
        return features
    
    def generate_synthetic_data(self, n_samples=600):
        """Генерация синтетических данных для обучения"""
        data = []
        
        fake_templates = [
            "ШОК! {subject} скрывают ПРАВДУ о {topic}! Все в УЖАСЕ!",
            "СЕНСАЦИЯ! {subject} РАЗОБЛАЧИЛИ {topic}! Это НЕВЕРОЯТНО!",
            "ТАЙНА {topic} РАСКРЫТА! {subject} в ШОКЕ!",
            "{subject} ОБМАНЫВАЮТ нас! ПРАВДА о {topic}!",
            "ВСКРЫЛАСЬ ТАЙНА! {topic} - это ОПАСНОСТЬ!"
        ]
        
        real_templates = [
            "По данным {source}, {topic} составил {number}%",
            "Эксперты отмечают, что {topic} демонстрирует рост",
            "Согласно отчету {source}, {topic} стабилизировался",
            "Аналитики прогнозируют изменение {topic}",
            "Статистика показывает, что {topic} соответствует норме"
        ]
        
        subjects = ["ученые", "власти", "врачи", "политики"]
        topics = ["инфляции", "экономике", "климате", "технологиях"]
        sources = ["Центробанка", "Минздрава", "Росстата", "экспертов"]
        
        # Фейковые новости
        for _ in range(n_samples // 2):
            template = np.random.choice(fake_templates)
            text = template.format(
                subject=np.random.choice(subjects),
                topic=np.random.choice(topics),
                source=np.random.choice(sources),
                number=np.random.randint(1, 99)
            )
            
            features = self.extract_features(text)
            features['text'] = text
            features['is_fake'] = 1
            data.append(features)
        
        # Настоящие новости
        for _ in range(n_samples // 2):
            template = np.random.choice(real_templates)
            text = template.format(
                source=np.random.choice(sources),
                topic=np.random.choice(topics),
                number=np.random.randint(1, 99)
            )
            
            features = self.extract_features(text)
            features['text'] = text
            features['is_fake'] = 0
            data.append(features)
        
        return pd.DataFrame(data)
    
    def load_or_train_model(self):
        """Загрузка или обучение модели"""
        model_path = 'models/fake_news_model.pkl'
        scaler_path = 'models/scaler.pkl'
        
        # Создаем папку models если её нет
        if not os.path.exists('models'):
            os.makedirs('models', exist_ok=True)
        
        try:
            # Пробуем загрузить существующую модель
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                logger.info("✅ Модель загружена из файла")
                
                # Проверяем точность на тестовых данных
                test_df = self.generate_synthetic_data(100)
                X_test, y_test = self._prepare_features(test_df)
                X_test_scaled = self.scaler.transform(X_test)
                accuracy = self.model.score(X_test_scaled, y_test)
                self.model_accuracy = accuracy
                logger.info(f"📊 Точность загруженной модели: {accuracy:.2%}")
                return
        except Exception as e:
            logger.warning(f"Не удалось загрузить модель: {e}")
        
        # Если не удалось загрузить, обучаем новую
        logger.info("🎓 Начинаю обучение новой модели...")
        self.train_model()
    
    def train_model(self):
        """Обучение модели"""
        # Генерация данных
        df = self.generate_synthetic_data(600)
        
        # Подготовка признаков
        X, y = self._prepare_features(df)
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Масштабирование
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Обучение модели
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Оценка
        train_accuracy = self.model.score(X_train_scaled, y_train)
        test_accuracy = self.model.score(X_test_scaled, y_test)
        self.model_accuracy = test_accuracy
        
        logger.info(f"📊 Точность на обучении: {train_accuracy:.2%}")
        logger.info(f"📊 Точность на тесте: {test_accuracy:.2%}")
        
        # Сохранение модели
        joblib.dump(self.model, 'models/fake_news_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        logger.info("💾 Модель сохранена")
    
    def _prepare_features(self, df):
        """Подготовка признаков для модели"""
        feature_cols = [
            'clickbait_score', 'emotional_score', 'certainty_score',
            'formality_score', 'source_indicator_score', 'balance_score',
            'avg_words_per_sentence', 'exclamation_density',
            'caps_ratio', 'has_percentages', 'subjectivity'
        ]
        
        # Заполнение пропусков
        X = df[feature_cols].fillna(0)
        y = df['is_fake']
        
        return X, y
    
    def predict(self, features):
        """Предсказание на основе признаков"""
        if self.model is None:
            return self._rule_based_predict(features)
        
        try:
            # Подготовка признаков для модели
            feature_values = [
                features.get('clickbait_score', 0),
                features.get('emotional_score', 0),
                features.get('certainty_score', 0),
                features.get('formality_score', 0),
                features.get('source_indicator_score', 0),
                features.get('balance_score', 0),
                features.get('avg_words_per_sentence', 0),
                features.get('exclamation_density', 0),
                features.get('caps_ratio', 0),
                features.get('has_percentages', 0),
                features.get('subjectivity', 0)
            ]
            
            # Масштабирование и предсказание
            X_scaled = self.scaler.transform([feature_values])
            proba = self.model.predict_proba(X_scaled)[0]
            
            # Корректировка на основе контекста
            adjusted_proba = proba[1]
            if features.get('has_percentages', 0):
                adjusted_proba *= 0.8
            if features.get('source_indicator_score', 0) > 0.3:
                adjusted_proba *= 0.7
            
            return {
                'is_fake': adjusted_proba > 0.5,
                'fake_probability': min(adjusted_proba, 0.99)
            }
            
        except Exception as e:
            logger.error(f"Ошибка ML предсказания: {e}")
            return self._rule_based_predict(features)
    
    def _rule_based_predict(self, features):
        """Rule-based предсказание как fallback"""
        fake_score = (
            features.get('clickbait_score', 0) * 0.3 +
            features.get('emotional_score', 0) * 0.2 +
            features.get('certainty_score', 0) * 0.2 +
            features.get('exclamation_density', 0) * 0.1 +
            features.get('caps_ratio', 0) * 0.1 +
            (1 - features.get('formality_score', 0)) * 0.1
        )
        
        # Корректировка на основе контекста
        if features.get('has_percentages', 0):
            fake_score *= 0.7
        if features.get('source_indicator_score', 0) > 0.3:
            fake_score *= 0.6
        
        return {
            'is_fake': fake_score > 0.5,
            'fake_probability': min(fake_score, 0.95)
        }
    
    def highlight_text(self, text):
        """Подсветка подозрительных фраз в тексте"""
        highlighted = text
        
        # Выделение кликбейт-слов
        for word in self.clickbait_words:
            pattern = re.compile(f'\\b{word}\\b', re.IGNORECASE)
            highlighted = pattern.sub(
                f'<span class="highlight clickbait">{word.upper()}</span>',
                highlighted
            )
        
        # Выделение слов категоричности
        for word in self.certainty_words:
            pattern = re.compile(f'\\b{word}\\b', re.IGNORECASE)
            highlighted = pattern.sub(
                f'<span class="highlight certainty">{word.upper()}</span>',
                highlighted
            )
        
        # Выделение источников
        for word in self.source_indicators:
            pattern = re.compile(f'\\b{word}\\b', re.IGNORECASE)
            highlighted = pattern.sub(
                f'<span class="highlight source">{word.upper()}</span>',
                highlighted
            )
        
        # Выделение чисел и процентов
        highlighted = re.sub(
            r'(\d+%?)',
            r'<span class="highlight number">\1</span>',
            highlighted
        )
        
        return highlighted
    
    def calculate_reliability_score(self, features, prediction, text):
        """Расчет общего балла достоверности"""
        base_score = 100 - (prediction['fake_probability'] * 100)
        
        # Корректировки
        adjustments = 0
        
        # Положительные корректировки
        if features.get('source_indicator_score', 0) > 0.3:
            adjustments += 10
        if features.get('formality_score', 0) > 0.4:
            adjustments += 8
        if features.get('has_percentages', 0):
            adjustments += 5
        
        # Отрицательные корректировки
        if features.get('clickbait_score', 0) > 0.4:
            adjustments -= 10
        if features.get('certainty_score', 0) > 0.5:
            adjustments -= 8
        if features.get('exclamation_density', 0) > 0.5:
            adjustments -= 5
        
        final_score = base_score + adjustments
        return max(0, min(100, round(final_score)))
    
    def generate_explanations(self, features, text):
        """Генерация объяснений результатов"""
        explanations = []
        
        if features.get('clickbait_score', 0) > 0.3:
            explanations.append("⚠️ Обнаружены кликбейт-слова")
        
        if features.get('emotional_score', 0) > 0.5:
            explanations.append("😠 Высокая эмоциональность текста")
        
        if features.get('certainty_score', 0) > 0.4:
            explanations.append("🎯 Избыточная категоричность")
        
        if features.get('exclamation_density', 0) > 0.3:
            explanations.append("❗ Много восклицательных знаков")
        
        if features.get('source_indicator_score', 0) > 0.2:
            explanations.append("✅ Упоминаются источники информации")
        
        if features.get('formality_score', 0) > 0.3:
            explanations.append("📝 Формальный стиль изложения")
        
        if features.get('has_percentages', 0):
            explanations.append("📊 Приведены статистические данные")
        
        # Если нет объяснений, добавить нейтральное
        if not explanations:
            explanations.append("📊 Текст не содержит явных стилистических маркеров")
        
        return explanations
