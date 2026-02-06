# app.py - Полная версия для GitHub + Render с упрощенной ML-моделью
from flask import Flask, render_template, request, jsonify
import re
import os
import pickle
import numpy as np
from textblob import TextBlob

app = Flask(__name__)

# ============================================================================
# 1. RULE-BASED ДЕТЕКТОР (Ваш существующий код с улучшениями)
# ============================================================================
class FakeNewsDetector:
    def __init__(self):
        # Существующие и расширенные словари
        self.clickbait_words = ['шок', 'сенсация', 'тайна', 'скандал', 'разоблачение',
                               'ужас', 'чудо', 'невероятно', 'потрясающе', 'срочно',
                               'эксклюзив', 'секрет', 'правда', 'ложь', 'обман',
                               'шокирующее', 'жуткий', 'адский', 'чудовищный', 'гнусный',
                               'шокирующее открытие', 'целенаправленно', 'адский заговор',
                               'жуткая статистика', 'замалчивали', 'срочно', 'эксклюзивно']

        self.certainty_words = ['точно', 'абсолютно', 'несомненно', 'безусловно',
                               'конечно', 'явно', 'очевидно', 'наверняка',
                               'гарантированно', 'стопроцентно', 'доказано', 'полностью']

        self.formal_words = ['сообщил', 'заявил', 'отметил', 'подчеркнул',
                            'указал', 'добавил', 'по данным', 'согласно', 'заметил']

        self.source_indicators = ['по данным', 'согласно', 'как сообщает',
                                 'по информации', 'по словам', 'по сведениям',
                                 'источник', 'эксперт', 'ученый', 'исследование']

        self.news_sources = ['центробанк', 'правительство', 'минздрав', 'роспотребнадзор',
                            'росстат', 'оон', 'всемирный банк', 'мвф', 'эксперты', 'аналитики',
                            'университет', 'институт', 'лаборатория', 'исследователи']

        # Новые категории для улучшенного анализа
        self.conspiracy_words = ['заговор', 'глобалисты', 'мировое правительство',
                                'мафия', 'тайный альянс', 'сильные мира сего',
                                'система', 'агенты', 'куплены', 'скрывают',
                                'замалчивают', 'сокрытие', 'правду скрывают',
                                'мегакорпоративный заговор', 'кремниевая мафия',
                                'фармацевтические гиганты', 'агенты системы']

        self.pseudo_science = ['токсины мышления', 'нейроны выжигает', 'излучение',
                              'волны', 'программа уничтожения', 'дегенеративный',
                              'оружение массового', 'нейро-щит', 'блокирует',
                              'токсины мышления', 'нейроны выжигает', 'нейро-щит',
                              'блокирует 99%', 'волны буквально выжигают',
                              'излучение смартфонов', 'программа уничтожения']

        self.fake_stat_words = ['на 300% выше', 'на 47% снижается', '99% токсинов',
                               'доказано фактами', 'статистика которую скрывают',
                               'невероятные но доказанные']

        self.anonymous_sources = ['псевдоним', 'имя изменено', 'пожелавший остаться анонимным',
                                 'наш источник', 'некоторые эксперты', 'ученый который',
                                 'доктор', 'эксперт под прикрытием']

        self.emotional_manipulation = ['спасите', 'пока не поздно', 'проснитесь', 'борьба началась',
                                       'немедленно', 'требуйте', 'распространите', 'выкиньте', 'замените',
                                       'критическое мышление', 'полное уничтожение', 'безнадёжными', 'глупыми']

    def analyze_text(self, text):
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Основные метрики
        clickbait_count = sum(1 for word in self.clickbait_words if word in text_lower)
        clickbait_score = min(clickbait_count / 2, 1.0)

        # Анализ эмоциональности через TextBlob
        try:
            blob = TextBlob(text)
            emotional_score = abs(blob.sentiment.polarity)
            if emotional_score > 0.5:
                emotional_score = min(emotional_score * 1.5, 1.0)
        except:
            emotional_score = 0

        certainty_count = sum(1 for word in self.certainty_words if word in text_lower)
        certainty_score = min(certainty_count / 2, 1.0)

        formal_count = sum(1 for word in self.formal_words if word in text_lower)
        formality_score = min(formal_count / 3, 1.0)

        source_count = sum(1 for word in self.source_indicators if word in text_lower)
        source_score = min(source_count / 2, 1.0)

        news_source_count = sum(1 for source in self.news_sources if source in text_lower)
        news_source_score = min(news_source_count / 2, 1.0)

        exclamation_count = text.count('!')
        exclamation_density = exclamation_count / max(len(sentences), 1)
        if exclamation_density > 0.5:
            exclamation_density = 1.0

        caps_count = sum(1 for c in text if c.isupper())
        caps_ratio = caps_count / max(len(text), 1)
        if caps_ratio > 0.1:
            caps_ratio = min(caps_ratio * 2, 1.0)

        # Новые метрики
        conspiracy_count = sum(1 for word in self.conspiracy_words if word in text_lower)
        conspiracy_score = min(conspiracy_count / 2, 1.0)

        pseudo_science_count = 0
        for phrase in self.pseudo_science:
            if phrase in text_lower:
                pseudo_science_count += 1
        pseudo_science_score = min(pseudo_science_count / 1, 1.0)

        fake_stat_count = sum(1 for word in self.fake_stat_words if word in text_lower)
        fake_stat_score = min(fake_stat_count / 1, 1.0)

        anonymous_count = sum(1 for word in self.anonymous_sources if word in text_lower)
        anonymous_score = min(anonymous_count / 1, 1.0)

        action_urgency_count = sum(1 for word in self.emotional_manipulation if word in text_lower)
        urgency_score = min(action_urgency_count / 2, 1.0)

        vague_source_penalty = 0
        if ('доктор' in text_lower or 'ученый' in text_lower or 'источник' in text_lower) and \
           any(word in text_lower for word in ['доказано', 'точно', 'абсолютно']):
            vague_source_penalty = 0.3

        only_exclam_quest = all(any(c in s for c in '!?') for s in sentences if s)
        structure_penalty = 0.4 if only_exclam_quest and len(sentences) > 3 else 0

        # Расчет общего балла фейковости
        fake_score = (
            clickbait_score * 0.15 +
            emotional_score * 0.20 +
            certainty_score * 0.10 +
            exclamation_density * 0.15 +
            caps_ratio * 0.05 +
            (1 - formality_score) * 0.03 +
            conspiracy_score * 0.12 +
            pseudo_science_score * 0.15 +
            fake_stat_score * 0.10 +
            anonymous_score * 0.08 +
            urgency_score * 0.10 +
            vague_source_penalty +
            structure_penalty
        )

        # Корректировки
        fake_score = min(max(fake_score, 0), 1)

        if news_source_score > 0.3 and anonymous_score < 0.3:
            fake_score *= 0.7
        elif anonymous_score > 0.3:
            fake_score *= 1.2

        # Достоверность и вердикт
        reliability_score = max(0, min(100, round(100 - (fake_score * 100))))

        if reliability_score >= 80:
            verdict = "ВЫСОКАЯ ДОСТОВЕРНОСТЬ"
            is_fake = False
        elif reliability_score >= 60:
            verdict = "СРЕДНЯЯ ДОСТОВЕРНОСТЬ"
            is_fake = fake_score > 0.6
        else:
            verdict = "НИЗКАЯ ДОСТОВЕРНОСТЬ"
            is_fake = True

        # Формируем метрики для отображения
        metrics = {
            'clickbait_score': round(clickbait_score * 100),
            'emotional_score': round(emotional_score * 100),
            'certainty_score': round(certainty_score * 100),
            'formality_score': round(formality_score * 100),
            'source_score': round(source_score * 100),
            'news_source_score': round(news_source_score * 100),
            'exclamation_density': round(exclamation_density * 100),
            'caps_ratio': round(caps_ratio * 100),
            'conspiracy_score': round(conspiracy_score * 100),
            'pseudo_science_score': round(pseudo_science_score * 100),
            'fake_stat_score': round(fake_stat_score * 100),
            'anonymous_score': round(anonymous_score * 100),
            'urgency_score': round(urgency_score * 100)
        }

        # Вектор признаков для возможной интеграции
        ml_feature_vector = [
            metrics['clickbait_score'] / 100,
            metrics['emotional_score'] / 100,
            metrics['conspiracy_score'] / 100,
            metrics['pseudo_science_score'] / 100,
            metrics['exclamation_density'] / 100,
            metrics['caps_ratio'] / 100,
            metrics['certainty_score'] / 100,
            metrics['anonymous_score'] / 100,
            fake_score,
            min(len(text) / 1000, 1.0)
        ]

        return {
            'reliability_score': reliability_score,
            'fake_score': round(fake_score * 100, 1),
            'is_fake': is_fake,
            'verdict': verdict,
            'metrics': metrics,
            'ml_feature_vector': ml_feature_vector,
            'details': {
                'clickbait_words': list(set([w for w in self.clickbait_words if w in text_lower])),
                'certainty_words': list(set([w for w in self.certainty_words if w in text_lower])),
                'conspiracy_words': list(set([w for w in self.conspiracy_words if w in text_lower])),
                'pseudo_science_phrases': list(set([p for p in self.pseudo_science if p in text_lower])),
                'exclamation_count': exclamation_count,
                'has_percentages': bool('%' in text or 'процент' in text_lower),
                'word_count': len(words),
                'sentence_count': len(sentences),
                'anonymous_sources_detected': anonymous_count > 0
            }
        }

    def highlight_text(self, text):
        """Подсветка ключевых слов в тексте"""
        highlighted = text
        
        # Подсветка кликбейта
        for word in self.clickbait_words:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            highlighted = pattern.sub(
                f'<span class="highlight clickbait">{word.upper()}</span>',
                highlighted
            )
        
        # Подсветка категоричности
        for word in self.certainty_words:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            highlighted = pattern.sub(
                f'<span class="highlight certainty">{word.upper()}</span>',
                highlighted
            )
        
        # Подсветка конспирологии
        for word in self.conspiracy_words:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            highlighted = pattern.sub(
                f'<span class="highlight conspiracy">{word.upper()}</span>',
                highlighted
            )
        
        return highlighted

    def generate_explanations(self, analysis):
        """Генерация пояснений для анализа"""
        explanations = []
        metrics = analysis['metrics']
        details = analysis['details']
        
        if metrics['clickbait_score'] > 20:
            explanations.append(f"⚠️ Высокий кликбейт-индекс ({metrics['clickbait_score']}%)")
        
        if metrics['emotional_score'] > 30:
            explanations.append(f"😠 Высокая эмоциональность ({metrics['emotional_score']}%)")
        
        if metrics['conspiracy_score'] > 20:
            explanations.append(f"🕵️ Признаки теории заговора ({metrics['conspiracy_score']}%)")
        
        if metrics['pseudo_science_score'] > 10:
            explanations.append(f"🔬 Псевдонаучные утверждения ({metrics['pseudo_science_score']}%)")
        
        if metrics['exclamation_density'] > 20:
            explanations.append(f"❗ Много восклицаний ({details['exclamation_count']} шт.)")
        
        if metrics['caps_ratio'] > 10:
            explanations.append(f"🔠 Много заглавных букв ({metrics['caps_ratio']}%)")
        
        if not explanations:
            explanations.append("✅ Текст не содержит явных стилистических маркеров фейковых новостей")
        
        return explanations

# ============================================================================
# 2. ML МОДЕЛЬ (Упрощенная версия для Render)
# ============================================================================
class SimpleMLPredictor:
    def __init__(self, model_path='simple_model.pkl'):
        self.model = None
        self.model_loaded = False
        
        # Пытаемся загрузить модель
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.model_loaded = True
                print(f"[ML] Модель успешно загружена из {model_path}")
            except Exception as e:
                print(f"[ML] Ошибка загрузки модели: {e}")
        else:
            print(f"[ML] Файл модели {model_path} не найден. ML анализ недоступен.")
    
    def predict(self, text):
        """Предсказание с использованием ML модели"""
        if not self.model_loaded or self.model is None:
            return {
                'is_fake': None,
                'confidence': 0,
                'error': 'ML модель не загружена',
                'available': False
            }
        
        try:
            # Предсказание вероятности
            prediction = self.model.predict_proba([text])[0]
            
            # Определяем класс (1 - фейк, 0 - достоверно)
            is_fake = self.model.predict([text])[0] == 1
            
            # Уверенность в предсказании
            confidence = prediction[1] if is_fake else prediction[0]
            
            return {
                'is_fake': bool(is_fake),
                'confidence': round(float(confidence * 100), 2),
                'ml_score': round(float(prediction[1]), 4),
                'verdict': 'ФЕЙК (ML анализ)' if is_fake else 'ДОСТОВЕРНО (ML анализ)',
                'available': True
            }
        except Exception as e:
            return {
                'is_fake': None,
                'confidence': 0,
                'error': f'Ошибка предсказания: {str(e)}',
                'available': False
            }
    
    def is_loaded(self):
        return self.model_loaded

# ============================================================================
# 3. ИНИЦИАЛИЗАЦИЯ КОМПОНЕНТОВ
# ============================================================================
print("=" * 60)
print("FAKE NEWS DETECTOR - Simplified Version for Render")
print("=" * 60)

# Создаем экземпляры детекторов
rule_detector = FakeNewsDetector()
ml_predictor = SimpleMLPredictor()

print(f"[System] Rule-based детектор: ✅ Загружен")
print(f"[System] ML модель: {'✅ Загружена' if ml_predictor.is_loaded() else '⚠️ Не загружена'}")
print("=" * 60)

# ============================================================================
# 4. FLASK ROUTES (Маршруты приложения)
# ============================================================================
@app.route('/')
def index():
    """Главная страница"""
    ml_status = "✅ Активна" if ml_predictor.is_loaded() else "⚠️ Недоступна"
    return render_template('index.html', 
                         ml_status=ml_status,
                         title="Детектор фейковых новостей")

@app.route('/analyze', methods=['POST'])
def analyze_rule_based():
    """Только rule-based анализ"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        # Проверка входных данных
        if not text:
            return jsonify({'error': 'Введите текст для анализа', 'success': False})
        
        if len(text) < 20:
            return jsonify({'error': 'Текст слишком короткий (минимум 20 символов)', 'success': False})
        
        if len(text) > 5000:
            return jsonify({'error': 'Текст слишком длинный (максимум 5000 символов)', 'success': False})
        
        # Выполняем анализ
        analysis = rule_detector.analyze_text(text)
        highlighted = rule_detector.highlight_text(text)
        explanations = rule_detector.generate_explanations(analysis)
        
        # Формируем ответ
        response = {
            'success': True,
            'type': 'rule_based',
            'reliability_score': analysis['reliability_score'],
            'fake_score': analysis['fake_score'],
            'is_fake': analysis['is_fake'],
            'verdict': analysis['verdict'],
            'highlighted_text': highlighted,
            'explanations': explanations,
            'metrics': analysis['metrics'],
            'details': analysis['details'],
            'text_length': len(text)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Внутренняя ошибка сервера: {str(e)}'
        })

@app.route('/analyze_hybrid', methods=['POST'])
def analyze_hybrid():
    """Гибридный анализ: rule-based + ML"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        # Проверка входных данных
        if not text:
            return jsonify({'error': 'Введите текст для анализа', 'success': False})
        
        if len(text) < 20:
            return jsonify({'error': 'Текст слишком короткий (минимум 20 символов)', 'success': False})
        
        if len(text) > 5000:
            return jsonify({'error': 'Текст слишком длинный (максимум 5000 символов)', 'success': False})
        
        # Rule-based анализ
        rule_analysis = rule_detector.analyze_text(text)
        highlighted = rule_detector.highlight_text(text)
        explanations = rule_detector.generate_explanations(rule_analysis)
        
        # ML анализ
        ml_result = ml_predictor.predict(text)
        
        # Комбинируем результаты
        if ml_result['available']:
            # Используем ML как основной результат
            final_verdict = ml_result['verdict']
            final_is_fake = ml_result['is_fake']
            final_confidence = ml_result['confidence']
            ml_available = True
        else:
            # Используем rule-based как основной результат
            final_verdict = rule_analysis['verdict']
            final_is_fake = rule_analysis['is_fake']
            final_confidence = rule_analysis['reliability_score']
            ml_available = False
        
        # Формируем ответ
        response = {
            'success': True,
            'type': 'hybrid',
            'ml_available': ml_available,
            
            # Итоговые результаты
            'is_fake': final_is_fake,
            'verdict': final_verdict,
            'confidence': final_confidence,
            
            # Rule-based детали
            'rule_based_score': rule_analysis['reliability_score'],
            'highlighted_text': highlighted,
            'explanations': explanations,
            'details': rule_analysis['details'],
            
            # ML детали (если доступны)
            'ml_result': ml_result if ml_available else None,
            
            # Статистика
            'text_length': len(text),
            'processing_time': 'мгновенно'
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Внутренняя ошибка сервера: {str(e)}'
        })

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка состояния сервиса"""
    return jsonify({
        'status': 'healthy',
        'service': 'Fake News Detector API',
        'version': '2.0-simple',
        'timestamp': os.times().user,
        'components': {
            'rule_based_detector': 'active',
            'ml_model': 'loaded' if ml_predictor.is_loaded() else 'not_loaded',
            'ml_type': 'LogisticRegression + TF-IDF'
        },
        'endpoints': {
            'home': '/',
            'rule_based_analysis': '/analyze (POST)',
            'hybrid_analysis': '/analyze_hybrid (POST)',
            'health_check': '/health (GET)'
        }
    })

@app.route('/api/status', methods=['GET'])
def api_status():
    """Статус API"""
    return jsonify({
        'ml_model_loaded': ml_predictor.is_loaded(),
        'rule_based_active': True,
        'total_endpoints': 4,
        'max_text_length': 5000,
        'min_text_length': 20
    })

# ============================================================================
# 5. ЗАПУСК СЕРВЕРА
# ============================================================================
if __name__ == '__main__':
    # Получаем порт из переменных окружения (для Render)
    port = int(os.environ.get('PORT', 5000))
    
    # Запускаем сервер
    print(f"\n[Server] Запуск сервера на порту {port}")
    print(f"[Server] Доступно по адресу: http://localhost:{port}")
    print(f"[Server] Health check: http://localhost:{port}/health")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=False)
