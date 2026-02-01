from flask import Flask, render_template, request, jsonify
import re
import os
from textblob import TextBlob
import nltk

# Загрузка данных для nltk (совместимо с Python 3.9.0)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

app = Flask(__name__)

class FakeNewsDetector:
    def __init__(self):
        # Существующие словари
        self.clickbait_words = ['шок', 'сенсация', 'тайна', 'скандал', 'разоблачение',
                               'ужас', 'чудо', 'невероятно', 'потрясающе', 'срочно',
                               'эксклюзив', 'секрет', 'правда', 'ложь', 'обман',
                               'шокирующее', 'жуткий', 'адский', 'чудовищный', 'гнусный']
        
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
        
        # Новые категории
        self.conspiracy_words = ['заговор', 'глобалисты', 'мировое правительство',
                                'мафия', 'тайный альянс', 'сильные мира сего',
                                'система', 'агенты', 'куплены', 'скрывают',
                                'замалчивают', 'сокрытие', 'правду скрывают']
        
        self.pseudo_science = ['токсины мышления', 'нейроны выжигает', 'излучение',
                              'волны', 'программа уничтожения', 'дегенеративный',
                              'оружие массового', 'нейро-щит', 'блокирует']
        
        self.fake_stat_words = ['на 300% выше', 'на 47% снижается', '99% токсинов',
                               'доказано фактами', 'статистика которую скрывают',
                               'невероятные но доказанные']
        
        self.anonymous_sources = ['псевдоним', 'имя изменено', 'пожелавший остаться анонимным',
                                 'наш источник', 'некоторые эксперты', 'ученый который',
                                 'доктор', 'эксперт под прикрытием']
    
    def analyze_text(self, text):
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 1. Кликбейт
        clickbait_count = sum(1 for word in self.clickbait_words if word in text_lower)
        clickbait_score = min(clickbait_count / 2, 1.0)
        
        # 2. Эмоциональность
        try:
            blob = TextBlob(text)
            emotional_score = abs(blob.sentiment.polarity)
            if emotional_score > 0.5:
                emotional_score = min(emotional_score * 1.5, 1.0)
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            emotional_score = 0
        
        # 3. Категоричность
        certainty_count = sum(1 for word in self.certainty_words if word in text_lower)
        certainty_score = min(certainty_count / 2, 1.0)
        
        # 4. Формальность
        formal_count = sum(1 for word in self.formal_words if word in text_lower)
        formality_score = min(formal_count / 3, 1.0)
        
        # 5. Источники
        source_count = sum(1 for word in self.source_indicators if word in text_lower)
        source_score = min(source_count / 2, 1.0)
        
        # 6. Официальные источники
        news_source_count = sum(1 for source in self.news_sources if source in text_lower)
        news_source_score = min(news_source_count / 2, 1.0)
        
        # 7. Пунктуация
        exclamation_count = text.count('!')
        exclamation_density = exclamation_count / max(len(sentences), 1)
        if exclamation_density > 0.5:
            exclamation_density = 1.0
        
        # 8. Регистр
        caps_count = sum(1 for c in text if c.isupper())
        caps_ratio = caps_count / max(len(text), 1)
        if caps_ratio > 0.1:
            caps_ratio = min(caps_ratio * 2, 1.0)
        
        # 9. Числа и статистика
        has_percentages = 1 if ('%' in text or 'процент' in text_lower) else 0
        has_numbers = 1 if bool(re.search(r'\d+', text)) else 0
        
        # 10. Конспирология
        conspiracy_count = sum(1 for word in self.conspiracy_words if word in text_lower)
        conspiracy_score = min(conspiracy_count / 2, 1.0)
        
        # 11. Псевдонаука
        pseudo_science_count = 0
        for phrase in self.pseudo_science:
            if phrase in text_lower:
                pseudo_science_count += 1
        pseudo_science_score = min(pseudo_science_count / 1, 1.0)
        
        # 12. Фейковая статистика
        fake_stat_count = sum(1 for word in self.fake_stat_words if word in text_lower)
        fake_stat_score = min(fake_stat_count / 1, 1.0)
        
        # 13. Анонимные источники
        anonymous_count = sum(1 for word in self.anonymous_sources if word in text_lower)
        anonymous_score = min(anonymous_count / 1, 1.0)
        
        # Расчет балла фейковости
        fake_score = (
            clickbait_score * 0.20 +
            emotional_score * 0.25 +
            certainty_score * 0.15 +
            exclamation_density * 0.20 +
            caps_ratio * 0.10 +
            (1 - formality_score) * 0.05 +
            conspiracy_score * 0.15 +
            pseudo_science_score * 0.20 +
            fake_stat_score * 0.15 +
            anonymous_score * 0.10
        )
        
        # Корректировки
        if has_percentages and fake_stat_score > 0:
            fake_score *= 1.2
        
        if news_source_score > 0.3 and anonymous_score < 0.3:
            fake_score *= 0.6
        elif anonymous_score > 0.3:
            fake_score *= 1.3
        
        # Комбинированные признаки
        if clickbait_score > 0.5 and exclamation_density > 0.5:
            fake_score += 0.2
        if conspiracy_score > 0.5 and pseudo_science_score > 0.3:
            fake_score += 0.3
        
        # Ограничение 0-1
        fake_score = min(max(fake_score, 0), 1)
        
        # Достоверность (0-100%)
        reliability_score = max(0, min(100, round(100 - (fake_score * 100))))
        
        # Вердикт
        if reliability_score >= 80:
            verdict = "ВЫСОКАЯ ДОСТОВЕРНОСТЬ"
            is_fake = False
        elif reliability_score >= 60:
            verdict = "СРЕДНЯЯ ДОСТОВЕРНОСТЬ"
            is_fake = fake_score > 0.6
        else:
            verdict = "НИЗКАЯ ДОСТОВЕРНОСТЬ"
            is_fake = True
        
        return {
            'reliability_score': reliability_score,
            'fake_score': round(fake_score * 100, 1),
            'is_fake': is_fake,
            'verdict': verdict,
            'metrics': {
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
                'anonymous_score': round(anonymous_score * 100)
            },
            'details': {
                'clickbait_words': list(set([w for w in self.clickbait_words if w in text_lower])),
                'certainty_words': list(set([w for w in self.certainty_words if w in text_lower])),
                'conspiracy_words': list(set([w for w in self.conspiracy_words if w in text_lower])),
                'pseudo_science_phrases': list(set([p for p in self.pseudo_science if p in text_lower])),
                'exclamation_count': exclamation_count,
                'has_percentages': bool(has_percentages),
                'has_numbers': bool(has_numbers),
                'word_count': len(words),
                'sentence_count': len(sentences),
                'anonymous_sources_detected': anonymous_count > 0
            }
        }
    
    def highlight_text(self, text):
        highlighted = text
        
        # Кликбейт
        for word in self.clickbait_words:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            highlighted = pattern.sub(
                f'<span class="highlight clickbait">{word.upper()}</span>',
                highlighted
            )
        
        # Категоричность
        for word in self.certainty_words:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            highlighted = pattern.sub(
                f'<span class="highlight certainty">{word.upper()}</span>',
                highlighted
            )
        
        # Конспирология
        for word in self.conspiracy_words:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            highlighted = pattern.sub(
                f'<span class="highlight conspiracy">{word.upper()}</span>',
                highlighted
            )
        
        # Псевдонаучные фразы
        for phrase in self.pseudo_science:
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            highlighted = pattern.sub(
                f'<span class="highlight pseudo">{phrase.upper()}</span>',
                highlighted
            )
        
        # Источники
        for word in self.source_indicators + self.news_sources + self.anonymous_sources:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            highlighted = pattern.sub(
                f'<span class="highlight source">{word.upper()}</span>',
                highlighted
            )
        
        # Числа
        highlighted = re.sub(r'(\d+%?)', r'<span class="highlight number">\1</span>', highlighted)
        
        # Восклицания
        highlighted = re.sub(r'(!+)', r'<span class="highlight exclamation">\1</span>', highlighted)
        
        return highlighted
    
    def generate_explanations(self, analysis):
        explanations = []
        metrics = analysis['metrics']
        details = analysis['details']
        
        if metrics['clickbait_score'] > 20:
            explanations.append(f"⚠️ Высокий кликбейт-индекс ({metrics['clickbait_score']}%)")
            if details['clickbait_words']:
                words = details['clickbait_words'][:3]
                explanations.append(f"   Найдены слова: {', '.join(words)}")
        
        if metrics['emotional_score'] > 30:
            explanations.append(f"😠 Высокая эмоциональность ({metrics['emotional_score']}%)")
        
        if metrics['certainty_score'] > 20:
            explanations.append(f"🎯 Избыточная категоричность ({metrics['certainty_score']}%)")
            if details['certainty_words']:
                words = details['certainty_words'][:3]
                explanations.append(f"   Слова: {', '.join(words)}")
        
        if metrics['conspiracy_score'] > 20:
            explanations.append(f"🕵️ Признаки теории заговора ({metrics['conspiracy_score']}%)")
            if details['conspiracy_words']:
                words = details['conspiracy_words'][:3]
                explanations.append(f"   Слова: {', '.join(words)}")
        
        if metrics['pseudo_science_score'] > 10:
            explanations.append(f"🔬 Псевдонаучные утверждения ({metrics['pseudo_science_score']}%)")
            if details['pseudo_science_phrases']:
                phrases = details['pseudo_science_phrases'][:2]
                explanations.append(f"   Фразы: {', '.join(phrases)}")
        
        if metrics['fake_stat_score'] > 10:
            explanations.append(f"📈 Сомнительная статистика ({metrics['fake_stat_score']}%)")
        
        if metrics['anonymous_score'] > 20:
            explanations.append(f"👤 Анонимные источники ({metrics['anonymous_score']}%)")
            explanations.append("   Нет конкретных имен и должностей")
        
        if metrics['exclamation_density'] > 20:
            explanations.append(f"❗ Много восклицаний ({details['exclamation_count']} шт.)")
        
        if metrics['caps_ratio'] > 10:
            explanations.append(f"🔠 Много заглавных букв ({metrics['caps_ratio']}%)")
        
        if metrics['source_score'] > 30:
            explanations.append(f"✅ Упоминаются источники ({metrics['source_score']}%)")
        else:
            explanations.append("❌ Источники не указаны или расплывчаты")
        
        if metrics['formality_score'] > 30:
            explanations.append(f"📝 Формальный стиль ({metrics['formality_score']}%)")
        
        if details['has_percentages']:
            if metrics['fake_stat_score'] > 20:
                explanations.append("⚠️ Статистика выглядит манипулятивной")
            else:
                explanations.append("📊 Есть статистические данные")
        
        if metrics['news_source_score'] > 20:
            explanations.append("🏛️ Упоминаются официальные источники")
        
        if not explanations:
            explanations.append("✅ Текст не содержит явных стилистических маркеров фейковых новостей")
        
        return explanations

detector = FakeNewsDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Введите текст для анализа'})
        
        if len(text) < 20:
            return jsonify({'error': 'Текст слишком короткий (минимум 20 символов)'})
        
        if len(text) > 5000:
            return jsonify({'error': 'Текст слишком длинный (максимум 5000 символов)'})
        
        analysis = detector.analyze_text(text)
        highlighted_text = detector.highlight_text(text)
        explanations = detector.generate_explanations(analysis)
        
        result = {
            'success': True,
            'reliability_score': analysis['reliability_score'],
            'fake_score': analysis['fake_score'],
            'is_fake': analysis['is_fake'],
            'verdict': analysis['verdict'],
            'highlighted_text': highlighted_text,
            'explanations': explanations,
            'metrics': analysis['metrics'],
            'details': analysis['details']
        }
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Error in analyze_text: {str(e)}")
        return jsonify({'error': f'Внутренняя ошибка: {str(e)}'})

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'ok',
        'version': '2.0',
        'algorithm': 'Rule-based стилистический анализ + конспирология',
        'python_version': '3.9.0'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
