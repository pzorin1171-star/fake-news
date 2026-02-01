from flask import Flask, render_template, request, jsonify
import re
import os
from textblob import TextBlob

app = Flask(__name__)

class FakeNewsDetector:
    def __init__(self):
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
        
        self.news_sources = ['центробанк', 'правительство', 'минздрав', 'роспотребнадзор',
                            'росстат', 'оон', 'всемирный банк', 'мвф', 'эксперты', 'аналитики']
    
    def analyze_text(self, text):
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 1. Кликбейт
        clickbait_count = sum(1 for word in self.clickbait_words if word in text_lower)
        clickbait_score = min(clickbait_count / 3, 1.0)
        
        # 2. Эмоциональность
        try:
            blob = TextBlob(text)
            emotional_score = abs(blob.sentiment.polarity)
        except:
            emotional_score = 0
        
        # 3. Категоричность
        certainty_count = sum(1 for word in self.certainty_words if word in text_lower)
        certainty_score = min(certainty_count / 3, 1.0)
        
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
        
        # 8. Регистр
        caps_count = sum(1 for c in text if c.isupper())
        caps_ratio = caps_count / len(text) if len(text) > 0 else 0
        
        # 9. Числа
        has_percentages = 1 if ('%' in text or 'процент' in text_lower) else 0
        has_numbers = 1 if bool(re.search(r'\d+', text)) else 0
        
        # Расчет балла фейковости
        fake_score = (
            clickbait_score * 0.25 +
            emotional_score * 0.20 +
            certainty_score * 0.15 +
            exclamation_density * 0.15 +
            caps_ratio * 0.10 +
            (1 - formality_score) * 0.10 +
            (1 - source_score) * 0.05
        )
        
        # Корректировки
        if has_percentages:
            fake_score *= 0.8
        if news_source_score > 0.3:
            fake_score *= 0.7
        
        # Достоверность (0-100%)
        reliability_score = max(0, min(100, round(100 - (fake_score * 100))))
        
        # Вердикт
        if reliability_score >= 75:
            verdict = "ВЫСОКАЯ ДОСТОВЕРНОСТЬ"
            is_fake = False
        elif reliability_score >= 50:
            verdict = "СРЕДНЯЯ ДОСТОВЕРНОСТЬ"
            is_fake = fake_score > 0.5
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
                'caps_ratio': round(caps_ratio * 100)
            },
            'details': {
                'clickbait_words': [w for w in self.clickbait_words if w in text_lower],
                'certainty_words': [w for w in self.certainty_words if w in text_lower],
                'exclamation_count': exclamation_count,
                'has_percentages': bool(has_percentages),
                'has_numbers': bool(has_numbers),
                'word_count': len(words),
                'sentence_count': len(sentences)
            }
        }
    
    def highlight_text(self, text):
        highlighted = text
        
        # Кликбейт
        for word in self.clickbait_words:
            pattern = re.compile(f'\\b{word}\\b', re.IGNORECASE)
            highlighted = pattern.sub(
                f'<span class="highlight clickbait">{word.upper()}</span>',
                highlighted
            )
        
        # Категоричность
        for word in self.certainty_words:
            pattern = re.compile(f'\\b{word}\\b', re.IGNORECASE)
            highlighted = pattern.sub(
                f'<span class="highlight certainty">{word.upper()}</span>',
                highlighted
            )
        
        # Источники
        for word in self.source_indicators + self.news_sources:
            pattern = re.compile(f'\\b{word}\\b', re.IGNORECASE)
            highlighted = pattern.sub(
                f'<span class="highlight source">{word.upper()}</span>',
                highlighted
            )
        
        # Числа
        highlighted = re.sub(r'(\d+%?)', r'<span class="highlight number">\1</span>', highlighted)
        
        # Восклицания
        if '!' in highlighted:
            parts = highlighted.split('!')
            highlighted = ''
            for i, part in enumerate(parts):
                highlighted += part
                if i < len(parts) - 1:
                    highlighted += '<span class="highlight exclamation">!</span>'
        
        return highlighted
    
    def generate_explanations(self, analysis):
        explanations = []
        metrics = analysis['metrics']
        details = analysis['details']
        
        if metrics['clickbait_score'] > 30:
            explanations.append(f"⚠️ Высокий кликбейт-индекс ({metrics['clickbait_score']}%)")
            if details['clickbait_words']:
                explanations.append(f"   Найдены слова: {', '.join(details['clickbait_words'])}")
        
        if metrics['emotional_score'] > 40:
            explanations.append(f"😠 Высокая эмоциональность ({metrics['emotional_score']}%)")
        
        if metrics['certainty_score'] > 30:
            explanations.append(f"🎯 Избыточная категоричность ({metrics['certainty_score']}%)")
            if details['certainty_words']:
                explanations.append(f"   Слова: {', '.join(details['certainty_words'])}")
        
        if metrics['exclamation_density'] > 30:
            explanations.append(f"❗ Много восклицаний ({details['exclamation_count']} шт.)")
        
        if metrics['caps_ratio'] > 20:
            explanations.append(f"🔠 Много заглавных букв ({metrics['caps_ratio']}%)")
        
        if metrics['source_score'] > 30:
            explanations.append(f"✅ Упоминаются источники ({metrics['source_score']}%)")
        
        if metrics['formality_score'] > 40:
            explanations.append(f"📝 Формальный стиль ({metrics['formality_score']}%)")
        
        if details['has_percentages']:
            explanations.append("📊 Есть статистические данные")
        
        if metrics['news_source_score'] > 30:
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
        return jsonify({'error': f'Внутренняя ошибка: {str(e)}'})

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'ok',
        'version': '1.0',
        'algorithm': 'Rule-based стилистический анализ'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
