import re
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import random
from collections import Counter
import math

class StyleAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_accuracy = 0.0
        
        # Расширенные списки признаков с весами
        self.clickbait_words = {
            'шок': 2.0, 'сенсация': 2.0, 'тайна': 1.5, 'скандал': 1.8, 'разоблачение': 1.7,
            'ужас': 1.8, 'чудо': 1.3, 'невероятно': 1.6, 'потрясающе': 1.4, 'срочно': 1.5,
            'эксклюзив': 1.6, 'секрет': 1.4, 'правда': 1.2, 'ложь': 1.5, 'обман': 1.7,
            'вскрылось': 1.8, 'оказалось': 1.3, 'выяснилось': 1.3, 'вот что': 1.2,
            'шокирующий': 2.0, 'сенсационный': 2.0, 'невероятный': 1.7, 'потрясающий': 1.5,
            'удивительный': 1.4, 'жуткий': 1.8, 'страшный': 1.7, 'опасный': 1.5, 'ужасный': 1.8,
            'критический': 1.3, 'катастрофический': 1.9, 'революционный': 1.4,
            'разоблачили': 1.8, 'обнаружили': 1.3, 'открыли': 1.2, 'признали': 1.3
        }
        
        self.certainty_words = {
            'точно': 1.8, 'абсолютно': 1.9, 'несомненно': 1.8, 'безусловно': 1.7,
            'конечно': 1.5, 'явно': 1.6, 'очевидно': 1.7, 'наверняка': 1.5,
            'гарантированно': 1.9, 'стопроцентно': 1.9, 'доказано': 1.8,
            'факт': 1.7, 'истина': 1.6, 'правда': 1.4, 'установлено': 1.6,
            'подтверждено': 1.7, 'проверено': 1.5, 'документально': 1.4,
            'официально': 1.3, 'научно': 1.3, 'несомненный': 1.8,
            'бесспорно': 1.8, 'неопровержимо': 1.9
        }
        
        self.formal_words = {
            'сообщил': 1.0, 'заявил': 1.0, 'отметил': 0.8, 'подчеркнул': 0.9,
            'указал': 0.8, 'добавил': 0.7, 'по данным': 1.2, 'согласно': 1.1,
            'в соответствии': 1.2, 'на основании': 1.1, 'отчет': 1.0,
            'исследование': 1.0, 'анализ': 1.0, 'статистика': 1.1,
            'эксперт': 0.9, 'специалист': 0.9, 'аналитик': 0.9,
            'доклад': 1.0, 'конференция': 0.8, 'пресс-релиз': 1.0
        }
        
        self.source_indicators = {
            'по данным': 1.3, 'согласно': 1.2, 'как сообщает': 1.3,
            'по информации': 1.2, 'по словам': 1.1, 'по сведениям': 1.1,
            'в интервью': 0.9, 'на пресс-конференции': 1.0,
            'в заявлении': 1.0, 'в докладе': 1.0, 'в исследовании': 1.0,
            'статистика': 1.0, 'отчет': 1.0, 'анализ': 1.0,
            'цифры': 0.8, 'результаты': 0.8, 'опрос': 0.8
        }
        
        self.balance_indicators = {
            'с одной стороны': 1.2, 'с другой стороны': 1.2,
            'однако': 0.8, 'тем не менее': 0.9, 'впрочем': 0.7,
            'хотя': 0.6, 'несмотря на': 0.7, 'по мнению': 0.8,
            'по оценкам': 0.9, 'возможно': 0.7, 'вероятно': 0.7,
            'предположительно': 0.6, 'по-видимому': 0.6,
            'согласно мнению': 0.9, 'исходя из': 0.8
        }
        
        self.emotional_intensifiers = {
            'очень': 1.2, 'крайне': 1.5, 'чрезвычайно': 1.6, 'невероятно': 1.7,
            'ужасно': 1.8, 'жутко': 1.8, 'страшно': 1.7, 'необычайно': 1.5,
            'совершенно': 1.4, 'абсолютно': 1.6, 'полностью': 1.3,
            'сильно': 1.2, 'глубоко': 1.1, 'необычно': 1.2
        }
        
        self.news_sources = {
            'центробанк': 1.4, 'правительство': 1.3, 'минздрав': 1.2,
            'роспотребнадзор': 1.2, 'росстат': 1.3, 'оон': 1.1,
            'всемирный банк': 1.1, 'мвф': 1.1, 'эксперты': 0.9,
            'аналитики': 0.9, 'ученые': 0.9, 'исследователи': 0.9,
            'журналисты': 0.7, 'корреспонденты': 0.7, 'редакция': 0.7,
            'министерство': 1.1, 'ведомство': 1.0, 'агентство': 1.0
        }
        
        self.credibility_indicators = {
            'процент': 0.8, 'статистика': 0.9, 'данные': 0.8,
            'исследование': 0.9, 'анализ': 0.9, 'отчет': 0.9,
            'цифры': 0.8, 'показатель': 0.8, 'тенденция': 0.7,
            'динамика': 0.7, 'результаты': 0.8, 'методология': 0.6
        }
        
    def extract_features(self, text):
        """Извлечение улучшенных стилистических признаков из текста"""
        features = {}
        text_lower = text.lower()
        
        # Базовые метрики
        features['length'] = len(text)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = re.findall(r'\b\w+\b', text_lower)
        features['word_count'] = len(words)
        
        if sentences:
            features['sentence_count'] = len(sentences)
            features['avg_sentence_length'] = sum(len(s) for s in sentences) / len(sentences)
            features['avg_words_per_sentence'] = len(words) / len(sentences) if len(sentences) > 0 else 0
            # Стандартное отклонение длины предложений
            if len(sentences) > 1:
                sentence_lengths = [len(s) for s in sentences]
                features['sentence_length_std'] = np.std(sentence_lengths)
            else:
                features['sentence_length_std'] = 0
        else:
            features['sentence_count'] = 0
            features['avg_sentence_length'] = 0
            features['avg_words_per_sentence'] = 0
            features['sentence_length_std'] = 0
        
        # Эмоциональность (сентимент-анализ с учетом контекста)
        try:
            blob = TextBlob(text)
            features['emotional_score'] = min(abs(blob.sentiment.polarity) * 1.5, 1.0)
            features['subjectivity'] = blob.sentiment.subjectivity
        except:
            features['emotional_score'] = 0.0
            features['subjectivity'] = 0.0
        
        # Усилители эмоциональности
        intensifier_score = sum(self.emotional_intensifiers.get(word, 0) for word in words)
        features['intensifier_score'] = min(intensifier_score / 3, 1.0)
        
        # Кликбейт индекс с учетом контекста и весов
        clickbait_score = 0
        clickbait_words_found = []
        for word, weight in self.clickbait_words.items():
            if word in text_lower:
                clickbait_score += weight
                clickbait_words_found.append(word)
        
        # Нормализация с учетом длины текста
        features['clickbait_score'] = min(clickbait_score / max(len(words) / 50, 1), 1.5)
        features['clickbait_words'] = clickbait_words_found
        
        # Категоричность с контекстом и весами
        certainty_score = 0
        certainty_words_found = []
        for word, weight in self.certainty_words.items():
            if word in text_lower:
                certainty_score += weight
                certainty_words_found.append(word)
        
        # Учитываем длину предложений
        sentence_penalty = 1.0
        if features['avg_words_per_sentence'] < 12:
            sentence_penalty = 1.3
        elif features['avg_words_per_sentence'] > 30:
            sentence_penalty = 0.7
            
        features['certainty_score'] = min(certainty_score * sentence_penalty / 4, 1.5)
        features['certainty_words'] = certainty_words_found
        
        # Формальность стиля
        formality_score = sum(self.formal_words.get(word, 0) for word in words)
        features['formality_score'] = min(formality_score / 5, 1.0)
        
        # Индикаторы источников
        source_score = sum(self.source_indicators.get(word, 0) for word in words)
        features['source_indicator_score'] = min(source_score / 3, 1.0)
        
        # Сбалансированность изложения
        balance_score = sum(self.balance_indicators.get(word, 0) for word in words)
        features['balance_score'] = min(balance_score / 2, 1.0)
        
        # Наличие упоминаний официальных источников
        news_source_score = sum(self.news_sources.get(word, 0) for word in words)
        features['news_source_score'] = min(news_source_score / 2, 1.0)
        
        # Показатели достоверности (цифры, статистика)
        credibility_score = sum(self.credibility_indicators.get(word, 0) for word in words)
        features['credibility_indicator_score'] = min(credibility_score / 3, 1.0)
        
        # Пунктуация и регистр
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['exclamation_density'] = features['exclamation_count'] / max(len(sentences), 1)
        
        # Анализ регистра
        caps_count = sum(1 for c in text if c.isupper())
        features['caps_count'] = caps_count
        features['caps_ratio'] = caps_count / len(text) if len(text) > 0 else 0
        
        # Наличие CAPS LOCK фраз (слова полностью заглавными)
        caps_words = re.findall(r'\b[A-ZА-Я]{2,}\b', text)
        features['caps_lock_words'] = len(caps_words)
        
        # Количественные данные
        numbers = re.findall(r'\b\d+\b', text)
        features['number_count'] = len(numbers)
        features['has_percentages'] = 1 if ('%' in text or 'процент' in text_lower or 'процентов' in text_lower) else 0
        features['has_dates'] = 1 if bool(re.search(r'\d{1,2}[./]\d{1,2}[./]\d{2,4}', text)) else 0
        features['has_currency'] = 1 if bool(re.search(r'\d+\s*(руб|р|₽|usd|\$|€|евро)', text_lower)) else 0
        
        # Лексическое разнообразие
        if words:
            unique_words = set(words)
            features['lexical_diversity'] = len(unique_words) / len(words)
            features['unique_word_count'] = len(unique_words)
            # Частотность слов
            word_freq = Counter(words)
            most_common = word_freq.most_common(3)
            features['most_common_words'] = [word for word, count in most_common]
            features['word_repetition_score'] = min(sum(count for _, count in most_common) / len(words) * 3, 1.0)
        else:
            features['lexical_diversity'] = 0
            features['unique_word_count'] = 0
            features['most_common_words'] = []
            features['word_repetition_score'] = 0
        
        # Структурные особенности
        features['paragraph_count'] = text.count('\n\n') + 1
        features['has_lists'] = 1 if bool(re.search(r'\d+\.\s|\-\s|\*\s', text)) else 0
        
        # Контекстные признаки
        features['is_question'] = 1 if text.strip().endswith('?') else 0
        features['title_case_ratio'] = self._calculate_title_case_ratio(text)
        
        # Сложность текста (приблизительная оценка)
        features['complexity_score'] = self._calculate_complexity_score(text, words, sentences)
        
        return features
    
    def _calculate_title_case_ratio(self, text):
        """Рассчитывает отношение слов с заглавной буквы к общему числу слов"""
        words = re.findall(r'\b[A-ZА-Я][a-zа-я]*\b', text)
        all_words = re.findall(r'\b[a-zA-Zа-яА-Я]+\b', text)
        if all_words:
            return len(words) / len(all_words)
        return 0
    
    def _calculate_complexity_score(self, text, words, sentences):
        """Рассчитывает сложность текста"""
        if not words or not sentences:
            return 0.0
            
        # Средняя длина слова
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Доля длинных слов (более 6 символов)
        long_words = sum(1 for word in words if len(word) > 6)
        long_word_ratio = long_words / len(words)
        
        # Индекс удобочитаемости Flesch-Kincaid (адаптированный)
        avg_sentence_length = len(words) / len(sentences)
        
        # Комбинированная оценка сложности
        complexity = (
            min(avg_word_length / 10, 1.0) * 0.4 +
            min(long_word_ratio * 2, 1.0) * 0.3 +
            min(avg_sentence_length / 30, 1.0) * 0.3
        )
        
        return min(complexity, 1.0)
    
    def generate_synthetic_data(self, n_samples=1500):
        """Генерация улучшенного синтетического датасета"""
        data = []
        
        # Шаблоны для разных типов текстов
        fake_templates = [
            "ШОК! {subject} скрывают ПРАВДУ о {topic}! Все в УЖАСЕ! АБСОЛЮТНО точно!",
            "СЕНСАЦИЯ! {subject} РАЗОБЛАЧИЛИ {topic}! Это НЕВЕРОЯТНО! СРОЧНО!",
            "ТАЙНА {topic} РАСКРЫТА! {subject} в ШОКЕ! ДОКАЗАНО научно!",
            "{subject} ОБМАНЫВАЮТ нас! ЖУТКАЯ ПРАВДА о {topic}!",
            "ВСКРЫЛАСЬ УЖАСНАЯ ТАЙНА! {topic} - это ОПАСНОСТЬ!",
            "НЕВЕРОЯТНО! {subject} СКРЫВАЮТ {topic}! Это ФАКТ!",
            "ШОКИРУЮЩЕЕ ОТКРЫТИЕ! {topic} УБИВАЕТ! СЕНСАЦИЯ!",
            "{subject} в ПАНИКЕ! {topic} УГРОЖАЕТ всем! СКАНДАЛ!",
            "МИР ПЕРЕВЕРНУЛСЯ! {topic} оказался ОБМАНОМ!",
            "ЖУТЬ! {subject} МОЛЧАТ о {topic}! Это КАТАСТРОФА!"
        ]
        
        # Реальные новости с эмоциональными заголовками
        real_emotional_templates = [
            "СЕНСАЦИЯ: {source} повысил ставку до {number}%",
            "ВАЖНО: По данным {source}, {topic} достиг {number}%",
            "{source} сообщает о росте {topic} на {number}%",
            "ЭКСКЛЮЗИВ: {source} опубликовал данные по {topic}",
            "Согласно отчету {source}, {topic} составил {number}%",
            "{source} заявил об изменении {topic} до {number}%",
            "По информации {source}, {topic} демонстрирует рост",
            "{source} обнародовал статистику по {topic}",
            "В соответствии с данными {source}, {topic} стабилизировался",
            "Эксперты {source} прокомментировали ситуацию с {topic}"
        ]
        
        # Нейтральные новости
        neutral_templates = [
            "По данным {source}, {topic} составил {number}% за отчетный период.",
            "Эксперты отмечают, что {topic} демонстрирует устойчивый рост.",
            "В соответствии с отчетом {source}, {topic} остался на прежнем уровне.",
            "Аналитики прогнозируют умеренное изменение {topic} в ближайшее время.",
            "Согласно исследованию, {topic} имеет тенденцию к постепенному улучшению.",
            "Статистика показывает, что {topic} соответствует средним значениям.",
            "В ходе конференции обсуждались перспективы развития {topic}.",
            "Доклад содержит данные о текущем состоянии {topic} в регионе.",
            "Экономические индикаторы свидетельствуют о стабилизации {topic}.",
            "По мнению специалистов, {topic} требует дальнейшего изучения."
        ]
        
        # Дезинформация с элементами правды
        mixed_templates = [
            "Эксперты предупреждают: {topic} может достигнуть {number}%",
            "Новые данные о {topic}: специалисты обсуждают последствия",
            "Аналитики спорят о влиянии {topic} на экономику",
            "В связи с {topic} возможны изменения в законодательстве",
            "Дискуссия о {topic} продолжается среди экспертов"
        ]
        
        subjects = ["ученые", "власти", "врачи", "политики", "журналисты", "банки", "корпорации"]
        topics = ["инфляции", "экономике", "климате", "технологиях", "здоровье", "финансах", "образовании", "вакцинации"]
        sources = ["Центробанка", "Минздрава", "Росстата", "ООН", "ВОЗ", "экспертов", "аналитиков", "исследователей"]
        
        # Генерация данных
        n_fake = n_samples // 3
        n_real_emotional = n_samples // 3
        n_neutral = n_samples // 4
        n_mixed = n_samples - n_fake - n_real_emotional - n_neutral
        
        # Фейковые новости
        for _ in range(n_fake):
            template = random.choice(fake_templates)
            text = template.format(
                subject=random.choice(subjects),
                topic=random.choice(topics),
                source=random.choice(sources),
                number=random.randint(1, 99)
            )
            
            # Добавляем вариативность
            if random.random() > 0.6:
                text = text.upper()
            if random.random() > 0.4:
                text = text + " " + "!" * random.randint(1, 3)
            if random.random() > 0.8:
                text = "ВНИМАНИЕ! " + text
            
            features = self.extract_features(text)
            features['text'] = text
            features['is_fake'] = 1
            data.append(features)
        
        # Реальные новости с эмоциональными заголовками
        for _ in range(n_real_emotional):
            template = random.choice(real_emotional_templates)
            text = template.format(
                source=random.choice(sources),
                topic=random.choice(topics),
                number=random.randint(1, 99)
            )
            
            features = self.extract_features(text)
            features['text'] = text
            features['is_fake'] = 0
            data.append(features)
        
        # Нейтральные новости
        for _ in range(n_neutral):
            template = random.choice(neutral_templates)
            text = template.format(
                source=random.choice(sources),
                topic=random.choice(topics),
                number=random.randint(1, 99)
            )
            
            # Иногда добавляем немного формальности
            if random.random() > 0.7:
                text = "Официально: " + text
            
            features = self.extract_features(text)
            features['text'] = text
            features['is_fake'] = 0
            data.append(features)
        
        # Смешанные (дезинформация)
        for _ in range(n_mixed):
            template = random.choice(mixed_templates)
            text = template.format(
                topic=random.choice(topics),
                number=random.randint(1, 99)
            )
            
            # Случайным образом определяем как фейк или правду
            is_fake = random.random() > 0.6
            
            features = self.extract_features(text)
            features['text'] = text
            features['is_fake'] = 1 if is_fake else 0
            data.append(features)
        
        df = pd.DataFrame(data)
        
        # Добавляем шум для реалистичности
        for i in range(len(df)):
            if random.random() < 0.05:  # 5% шума
                df.loc[i, 'is_fake'] = 1 - df.loc[i, 'is_fake']
        
        return df
    
    def load_or_train_model(self):
        """Загрузка существующей модели или обучение новой"""
        model_path = 'models/fake_news_model.pkl'
        scaler_path = 'models/scaler.pkl'
        
        # Создаем директорию для моделей
        os.makedirs('models', exist_ok=True)
        
        try:
            # Пробуем загрузить существующую модель
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                print("✅ Модель загружена из файла")
                
                # Оцениваем точность на тестовых данных
                df = self.generate_synthetic_data(300)
                X, y = self._prepare_features(df)
                X_scaled = self.scaler.transform(X)
                accuracy = self.model.score(X_scaled, y)
                self.model_accuracy = accuracy
                print(f"📊 Точность загруженной модели: {accuracy:.2%}")
                return
        except Exception as e:
            print(f"⚠️ Не удалось загрузить модель: {e}")
        
        # Если не удалось загрузить, обучаем новую
        print("🔧 Обучение новой модели...")
        self.train_model()
    
    def train_model(self):
        """Обучение улучшенной ML модели"""
        # Генерация данных
        df = self.generate_synthetic_data(1500)
        print(f"📈 Сгенерировано {len(df)} примеров для обучения")
        
        # Подготовка признаков
        X, y = self._prepare_features(df)
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📚 Размер тренировочной выборки: {len(X_train)}")
        print(f"📚 Размер тестовой выборки: {len(X_test)}")
        
        # Масштабирование
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Обучение ансамблевой модели
        self.model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            subsample=0.8,
            max_features='sqrt'
        )
        
        # Кросс-валидация
        print("🔬 Проводим кросс-валидацию...")
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"📊 Кросс-валидация: {cv_scores.mean():.2%} (+/- {cv_scores.std() * 2:.2%})")
        
        # Обучение на всех тренировочных данных
        print("🎓 Обучаем модель...")
        self.model.fit(X_train_scaled, y_train)
        
        # Оценка на тестовых данных
        train_accuracy = self.model.score(X_train_scaled, y_train)
        test_accuracy = self.model.score(X_test_scaled, y_test)
        self.model_accuracy = test_accuracy
        
        print(f"📈 Точность на тренировочных данных: {train_accuracy:.2%}")
        print(f"📈 Точность на тестовых данных: {test_accuracy:.2%}")
        
        # Важность признаков
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🏆 Топ-10 важных признаков:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        # Сохранение модели
        joblib.dump(self.model, 'models/fake_news_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        print(f"💾 Модель сохранена в models/fake_news_model.pkl")
        
        return self.model
    
    def _prepare_features(self, df):
        """Подготовка признаков для модели"""
        feature_cols = [
            'clickbait_score', 'emotional_score', 'certainty_score',
            'formality_score', 'source_indicator_score', 'balance_score',
            'news_source_score', 'credibility_indicator_score',
            'avg_words_per_sentence', 'exclamation_density',
            'caps_ratio', 'caps_lock_words', 'has_percentages',
            'lexical_diversity', 'intensifier_score', 'subjectivity',
            'word_repetition_score', 'complexity_score', 'sentence_length_std'
        ]
        
        # Заполнение пропусков
        X = df[feature_cols].fillna(0)
        y = df['is_fake']
        
        return X, y
    
    def predict(self, features):
        """Предсказание на основе извлеченных признаков"""
        if self.model is None:
            # Fallback на rule-based подход
            return self._rule_based_predict(features)
        
        try:
            # Подготовка признаков для модели
            feature_values = self._prepare_prediction_features(features)
            
            # Масштабирование и предсказание
            X_scaled = self.scaler.transform([feature_values])
            proba = self.model.predict_proba(X_scaled)[0]
            
            # Корректировка на основе контекстных признаков
            adjusted_proba = self._adjust_probability(proba[1], features)
            
            return {
                'is_fake': adjusted_proba > 0.5,
                'fake_probability': adjusted_proba,
                'raw_probability': proba[1]
            }
        except Exception as e:
            print(f"⚠️ Ошибка ML предсказания: {e}")
            return self._rule_based_predict(features)
    
    def _prepare_prediction_features(self, features):
        """Подготовка признаков для предсказания"""
        return [
            features.get('clickbait_score', 0),
            features.get('emotional_score', 0),
            features.get('certainty_score', 0),
            features.get('formality_score', 0),
            features.get('source_indicator_score', 0),
            features.get('balance_score', 0),
            features.get('news_source_score', 0),
            features.get('credibility_indicator_score', 0),
            features.get('avg_words_per_sentence', 0),
            features.get('exclamation_density', 0),
            features.get('caps_ratio', 0),
            features.get('caps_lock_words', 0),
            features.get('has_percentages', 0),
            features.get('lexical_diversity', 0),
            features.get('intensifier_score', 0),
            features.get('subjectivity', 0),
            features.get('word_repetition_score', 0),
            features.get('complexity_score', 0),
            features.get('sentence_length_std', 0)
        ]
    
    def _adjust_probability(self, probability, features):
        """Корректировка вероятности на основе контекста"""
        adjusted = probability
        
        # Положительные корректировки (снижают вероятность фейка)
        if features.get('source_indicator_score', 0) > 0.3:
            adjusted *= 0.7
        if features.get('credibility_indicator_score', 0) > 0.3:
            adjusted *= 0.8
        if features.get('formality_score', 0) > 0.4:
            adjusted *= 0.6
        if features.get('has_percentages', 0) > 0:
            adjusted *= 0.8
        if features.get('news_source_score', 0) > 0.3:
            adjusted *= 0.7
        if features.get('balance_score', 0) > 0.3:
            adjusted *= 0.9
        
        # Отрицательные корректировки (повышают вероятность фейка)
        if features.get('clickbait_score', 0) > 0.5:
            adjusted = min(adjusted * 1.5, 0.95)
        if features.get('emotional_score', 0) > 0.6:
            adjusted = min(adjusted * 1.3, 0.95)
        if features.get('certainty_score', 0) > 0.5:
            adjusted = min(adjusted * 1.4, 0.95)
        if features.get('exclamation_density', 0) > 0.5:
            adjusted = min(adjusted * 1.2, 0.95)
        if features.get('caps_ratio', 0) > 0.3:
            adjusted = min(adjusted * 1.3, 0.95)
        if features.get('intensifier_score', 0) > 0.5:
            adjusted = min(adjusted * 1.2, 0.95)
        
        # Учитываем сложность текста
        if features.get('complexity_score', 0) < 0.2 and len(features.get('most_common_words', [])) > 0:
            # Очень простой текст с повторами
            adjusted = min(adjusted * 1.1, 0.95)
        
        return min(max(adjusted, 0.01), 0.99)
    
    def _rule_based_predict(self, features):
        """Rule-based предсказание как fallback"""
        fake_score = (
            features.get('clickbait_score', 0) * 0.25 +
            features.get('emotional_score', 0) * 0.20 +
            features.get('certainty_score', 0) * 0.15 +
            features.get('exclamation_density', 0) * 0.15 +
            features.get('caps_ratio', 0) * 0.10 +
            (1 - features.get('formality_score', 0)) * 0.10 +
            (1 - features.get('source_indicator_score', 0)) * 0.05
        )
        
        # Корректировка на основе контекста
        if features.get('has_percentages', 0) > 0:
            fake_score *= 0.8
        if features.get('news_source_score', 0) > 0.3:
            fake_score *= 0.7
        if features.get('credibility_indicator_score', 0) > 0.3:
            fake_score *= 0.8
        
        return {
            'is_fake': fake_score > 0.55,
            'fake_probability': min(fake_score, 0.95)
        }
    
    def highlight_text(self, text):
        """Улучшенное выделение подозрительных фраз в тексте"""
        if not text:
            return ""
            
        highlighted = text
        
        # Сначала выделяем числа и проценты (зеленым)
        highlighted = re.sub(
            r'(\d+%?)',
            r'<span class="highlight number" title="Статистические данные">\1</span>',
            highlighted
        )
        
        # Выделение источников (синим)
        for source in list(self.news_sources.keys()) + list(self.source_indicators.keys()):
            pattern = re.compile(f'(?<!\\w)({source})(?!\\w)', re.IGNORECASE)
            highlighted = pattern.sub(
                f'<span class="highlight source" title="Упоминание источника">\\1</span>',
                highlighted
            )
        
        # Выделение кликбейт-слов с учетом контекста
        for word in self.clickbait_words:
            pattern = re.compile(f'(?<!\\w)({word})(?!\\w)', re.IGNORECASE)
            matches = list(pattern.finditer(highlighted))
            
            for match in reversed(matches):
                start, end = match.span()
                
                # Проверяем контекст вокруг слова
                context_start = max(0, start - 30)
                context_end = min(len(highlighted), end + 30)
                context = highlighted[context_start:context_end].lower()
                
                # Если рядом есть числа или источники - менее подозрительно
                has_numbers_nearby = bool(re.search(r'\d+%?', highlighted[max(0, start-20):end+20]))
                has_sources_nearby = any(src in context for src in self.source_indicators)
                
                if has_numbers_nearby and has_sources_nearby:
                    replacement = f'<span class="highlight clickbait-context" title="Эмоциональное слово в контексте данных">\\1</span>'
                else:
                    replacement = f'<span class="highlight clickbait" title="Кликбейт-слово">\\1</span>'
                
                highlighted = highlighted[:start] + replacement + highlighted[end:]
        
        # Выделение слов категоричности
        for word in self.certainty_words:
            pattern = re.compile(f'(?<!\\w)({word})(?!\\w)', re.IGNORECASE)
            highlighted = pattern.sub(
                f'<span class="highlight certainty" title="Категоричное утверждение">\\1</span>',
                highlighted
            )
        
        # Выделение восклицательных и вопросительных знаков
        sentences = re.split(r'([.!?]+)', highlighted)
        for i in range(len(sentences)):
            if '!' in sentences[i]:
                excl_count = sentences[i].count('!')
                if excl_count > 2:
                    sentences[i] = f'<span class="highlight exclamation-high" title="Много восклицаний ({excl_count})">{sentences[i]}</span>'
                elif excl_count > 1:
                    sentences[i] = f'<span class="highlight exclamation-medium" title="Несколько восклицаний">{sentences[i]}</span>'
                else:
                    sentences[i] = f'<span class="highlight exclamation" title="Восклицание">{sentences[i]}</span>'
            elif '?' in sentences[i]:
                sentences[i] = f'<span class="highlight question" title="Вопрос">{sentences[i]}</span>'
        
        highlighted = ''.join(sentences)
        
        # Выделение ВСЕХ ЗАГЛАВНЫХ СЛОВ
        words = highlighted.split()
        for i, word in enumerate(words):
            if len(word) > 2 and word.isupper() and word.isalpha():
                if len(word) > 5:
                    words[i] = f'<span class="highlight caps-high" title="Длинное слово заглавными">{word}</span>'
                else:
                    words[i] = f'<span class="highlight caps" title="Слово заглавными">{word}</span>'
        
        highlighted = ' '.join(words)
        
        return highlighted
    
    def calculate_reliability_score(self, features, prediction, text):
        """Расчет комплексного балла достоверности"""
        # Базовый балл от ML модели
        base_score = 100 - (prediction['fake_probability'] * 100)
        
        # Корректировки на основе контекста
        adjustments = 0
        
        # Положительные корректировки (повышают достоверность)
        if features.get('source_indicator_score', 0) > 0.3:
            adjustments += 12
        if features.get('credibility_indicator_score', 0) > 0.3:
            adjustments += 10
        if features.get('formality_score', 0) > 0.4:
            adjustments += 15
        if features.get('has_percentages', 0) > 0:
            adjustments += 8
        if features.get('news_source_score', 0) > 0.3:
            adjustments += 15
        if features.get('balance_score', 0) > 0.3:
            adjustments += 10
        if features.get('has_dates', 0) > 0:
            adjustments += 5
        if features.get('has_currency', 0) > 0:
            adjustments += 5
        
        # Отрицательные корректировки (снижают достоверность)
        if features.get('clickbait_score', 0) > 0.6:
            adjustments -= 20
        elif features.get('clickbait_score', 0) > 0.3:
            adjustments -= 10
            
        if features.get('certainty_score', 0) > 0.6:
            adjustments -= 15
        elif features.get('certainty_score', 0) > 0.3:
            adjustments -= 8
            
        if features.get('exclamation_density', 0) > 0.5:
            adjustments -= 12
        elif features.get('exclamation_density', 0) > 0.2:
            adjustments -= 6
            
        if features.get('caps_ratio', 0) > 0.3:
            adjustments -= 15
        elif features.get('caps_ratio', 0) > 0.15:
            adjustments -= 8
            
        if features.get('intensifier_score', 0) > 0.5:
            adjustments -= 10
            
        if features.get('caps_lock_words', 0) > 2:
            adjustments -= 8
            
        if features.get('word_repetition_score', 0) > 0.4:
            adjustments -= 5
        
        # Учитываем длину и сложность текста
        text_length = len(text)
        if text_length > 800:
            adjustments += 10  # Длинные аналитические тексты
        elif text_length > 300:
            adjustments += 5
        elif text_length < 100:
            adjustments -= 10  # Слишком короткие тексты
            
        if features.get('complexity_score', 0) > 0.5:
            adjustments += 5  # Сложные тексты обычно более достоверны
        
        final_score = base_score + adjustments
        return max(0, min(100, round(final_score)))
    
    def generate_explanations(self, features, text):
        """Генерация объяснений результатов анализа"""
        explanations = []
        
        # Собираем статистику
        total_flags = 0
        warning_flags = 0
        positive_flags = 0
        
        # Положительные признаки
        positive_points = []
        if features.get('source_indicator_score', 0) > 0.2:
            positive_points.append("📋 Упоминаются источники информации")
            positive_flags += 1
        if features.get('formality_score', 0) > 0.3:
            positive_points.append("📝 Формальный стиль изложения")
            positive_flags += 1
        if features.get('balance_score', 0) > 0.2:
            positive_points.append("⚖️ Сбалансированное изложение")
            positive_flags += 1
        if features.get('has_percentages', 0) > 0:
            positive_points.append("📊 Приведены статистические данные")
            positive_flags += 1
        if features.get('news_source_score', 0) > 0.2:
            positive_points.append("🏛️ Упоминаются официальные источники")
            positive_flags += 1
        if features.get('credibility_indicator_score', 0) > 0.2:
            positive_points.append("🔬 Научный/аналитический подход")
            positive_flags += 1
        if features.get('has_dates', 0) > 0:
            positive_points.append("📅 Указаны даты/сроки")
            positive_flags += 1
        if features.get('has_currency', 0) > 0:
            positive_points.append("💰 Указаны финансовые данные")
            positive_flags += 1
        
        # Отрицательные признаки
        warning_points = []
        if features.get('clickbait_score', 0) > 0.3:
            clickbait_words = features.get('clickbait_words', [])[:3]
            clickbait_str = ", ".join(clickbait_words) if clickbait_words else "эмоциональные слова"
            warning_points.append(f"🚨 Кликбейт-слова: {clickbait_str}")
            warning_flags += 1
            total_flags += 1
        if features.get('emotional_score', 0) > 0.5:
            warning_points.append("😠 Высокая эмоциональность текста")
            warning_flags += 1
            total_flags += 1
        if features.get('certainty_score', 0) > 0.4:
            certainty_words = features.get('certainty_words', [])[:2]
            certainty_str = ", ".join(certainty_words) if certainty_words else "категоричные слова"
            warning_points.append(f"⚠️ Избыточная категоричность: {certainty_str}")
            warning_flags += 1
            total_flags += 1
        if features.get('exclamation_density', 0) > 0.3:
            warning_points.append(f"❗ Высокая плотность восклицаний ({features['exclamation_count']} знаков)")
            warning_flags += 1
            total_flags += 1
        if features.get('caps_ratio', 0) > 0.2:
            warning_points.append("🔠 Много заглавных букв")
            warning_flags += 1
            total_flags += 1
        if features.get('intensifier_score', 0) > 0.4:
            warning_points.append("💥 Много эмоциональных усилителей")
            warning_flags += 1
            total_flags += 1
        if features.get('caps_lock_words', 0) > 1:
            warning_points.append(f"🆘 Слова заглавными: {features['caps_lock_words']} шт.")
            warning_flags += 1
            total_flags += 1
        if features.get('word_repetition_score', 0) > 0.4:
            common_words = features.get('most_common_words', [])[:3]
            if common_words:
                warning_points.append(f"🔄 Повторы слов: {', '.join(common_words)}")
                warning_flags += 1
                total_flags += 1
        
        # Общая оценка
        if warning_flags == 0 and positive_flags >= 3:
            explanations.append("✅ Высокая вероятность достоверности")
            explanations.append("Текст содержит множество признаков качественной журналистики")
        elif warning_flags <= 2 and positive_flags >= 2:
            explanations.append("⚠️ Средняя вероятность достоверности")
            explanations.append("Текст имеет как сильные, так и слабые стороны")
        elif warning_flags >= 3:
            explanations.append("🚨 Низкая вероятность достоверности")
            explanations.append("Обнаружено много признаков фейковых новостей")
        else:
            explanations.append("📊 Нейтральная оценка")
            explanations.append("Текст не содержит явных маркеров достоверности или фейковости")
        
        # Добавляем детали
        if warning_points:
            explanations.append("\n🚩 **Обнаружены тревожные признаки:**")
            explanations.extend([f"• {point}" for point in warning_points])
        
        if positive_points:
            explanations.append("\n✅ **Признаки достоверности:**")
            explanations.extend([f"• {point}" for point in positive_points])
        
        # Особые случаи
        if features.get('clickbait_score', 0) > 0.3 and features.get('has_percentages', 0) > 0:
            explanations.append("\n💡 **Важно:** Несмотря на эмоциональный заголовок, текст содержит статистические данные. Рекомендуется проверить источник данных.")
        
        if len(text) < 150:
            explanations.append("\n📝 **Внимание:** Текст очень короткий. Короткие сообщения часто не содержат достаточного контекста.")
        
        if features.get('complexity_score', 0) < 0.2:
            explanations.append("\n📚 **Замечание:** Текст очень простой. Сложные темы обычно требуют более детального изложения.")
        
        # Рекомендации
        recommendations = []
        if warning_flags > positive_flags:
            recommendations.append("🔍 Проверьте информацию в независимых источниках")
            recommendations.append("📰 Обратите внимание на стиль изложения - он может быть манипулятивным")
        elif positive_flags > 0:
            recommendations.append("✅ Информация выглядит правдоподобно, но всегда проверяйте факты")
        
        if recommendations:
            explanations.append("\n🎯 **Рекомендации:**")
            explanations.extend([f"• {rec}" for rec in recommendations])
        
        return explanations
    
    def assess_credibility(self, features, text):
        """Комплексная оценка достоверности"""
        assessment = {
            'style_analysis': {},
            'content_analysis': {},
            'risk_factors': [],
            'confidence_level': 'medium'
        }
        
        # Анализ стиля
        style_score = 0
        max_style_score = 6
        
        if features.get('formality_score', 0) > 0.3:
            style_score += 1
            assessment['style_analysis']['formality'] = 'Соответствует'
        
        if features.get('balance_score', 0) > 0.2:
            style_score += 1
            assessment['style_analysis']['balance'] = 'Сбалансированный'
        
        if features.get('exclamation_density', 0) < 0.3:
            style_score += 1
            assessment['style_analysis']['punctuality'] = 'Умеренная'
        
        if features.get('caps_ratio', 0) < 0.2:
            style_score += 1
            assessment['style_analysis']['case_usage'] = 'Корректное'
        
        if features.get('lexical_diversity', 0) > 0.6:
            style_score += 1
            assessment['style_analysis']['vocabulary'] = 'Разнообразный'
        
        if features.get('complexity_score', 0) > 0.3:
            style_score += 1
            assessment['style_analysis']['complexity'] = 'Достаточная'
        
        assessment['style_analysis']['score'] = f"{style_score}/{max_style_score}"
        assessment['style_analysis']['percentage'] = round((style_score / max_style_score) * 100)
        
        # Анализ содержания
        content_score = 0
        max_content_score = 6
        
        if features.get('source_indicator_score', 0) > 0.2:
            content_score += 1
            assessment['content_analysis']['sources'] = 'Есть ссылки'
        
        if features.get('has_percentages', 0) > 0:
            content_score += 1
            assessment['content_analysis']['data'] = 'Есть статистика'
        
        if features.get('has_dates', 0) > 0:
            content_score += 1
            assessment['content_analysis']['dates'] = 'Есть датировка'
        
        if features.get('news_source_score', 0) > 0.2:
            content_score += 1
            assessment['content_analysis']['official_sources'] = 'Упомянуты'
        
        if features.get('credibility_indicator_score', 0) > 0.2:
            content_score += 1
            assessment['content_analysis']['methodology'] = 'Научный подход'
        
        if features.get('has_currency', 0) > 0:
            content_score += 1
            assessment['content_analysis']['financial_data'] = 'Есть'
        
        assessment['content_analysis']['score'] = f"{content_score}/{max_content_score}"
        assessment['content_analysis']['percentage'] = round((content_score / max_content_score) * 100)
        
        # Факторы риска
        if features.get('clickbait_score', 0) > 0.4:
            assessment['risk_factors'].append({
                'factor': 'Кликбейт',
                'severity': 'high' if features['clickbait_score'] > 0.6 else 'medium',
                'description': 'Эмоциональные слова для привлечения внимания'
            })
        
        if features.get('certainty_score', 0) > 0.5:
            assessment['risk_factors'].append({
                'factor': 'Категоричность',
                'severity': 'high',
                'description': 'Избыточная уверенность в утверждениях'
            })
        
        if features.get('exclamation_density', 0) > 0.5:
            assessment['risk_factors'].append({
                'factor': 'Эмоциональность',
                'severity': 'high',
                'description': 'Высокая плотность восклицаний'
            })
        
        if features.get('caps_ratio', 0) > 0.3:
            assessment['risk_factors'].append({
                'factor': 'Кричащий стиль',
                'severity': 'medium',
                'description': 'Много заглавных букв'
            })
        
        # Уровень уверенности
        total_score = style_score + content_score
        max_total_score = max_style_score + max_content_score
        confidence_percentage = (total_score / max_total_score) * 100
        
        if confidence_percentage >= 70:
            assessment['confidence_level'] = 'high'
        elif confidence_percentage >= 40:
            assessment['confidence_level'] = 'medium'
        else:
            assessment['confidence_level'] = 'low'
        
        assessment['overall_score'] = round(confidence_percentage)
        
        return assessment
