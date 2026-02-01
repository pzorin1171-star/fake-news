from flask import Flask, render_template, request, jsonify
from style_analyzer import StyleAnalyzer
import json

app = Flask(__name__)
analyzer = StyleAnalyzer()

# Загружаем и обучаем модель при запуске
try:
    analyzer.load_or_train_model()
    print("✅ Детектор фейковых новостей успешно инициализирован")
    print(f"📊 Точность модели: {analyzer.model_accuracy:.2%}")
except Exception as e:
    print(f"❌ Ошибка при инициализации модели: {e}")

@app.route('/')
def index():
    """Главная страница с формой ввода"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Анализ текста и возврат результатов"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Введите текст для анализа'})
        
        if len(text) < 20:
            return jsonify({'error': 'Текст слишком короткий. Введите не менее 20 символов.'})
        
        if len(text) > 10000:
            return jsonify({'error': 'Текст слишком длинный. Максимум 10000 символов.'})
        
        # Анализ текста
        features = analyzer.extract_features(text)
        prediction = analyzer.predict(features)
        highlighted_text = analyzer.highlight_text(text)
        credibility_assessment = analyzer.assess_credibility(features, text)
        
        # Подготовка данных для визуализации
        visualization_data = {
            'clickbait_score': round(features['clickbait_score'] * 100),
            'emotional_score': round(features['emotional_score'] * 100),
            'certainty_score': round(features['certainty_score'] * 100),
            'formality_score': round(features['formality_score'] * 100),
            'source_score': round(features['source_indicator_score'] * 100),
            'balance_score': round(features['balance_score'] * 100),
        }
        
        # Расчет общего балла достоверности с учетом контекста
        reliability_score = analyzer.calculate_reliability_score(features, prediction, text)
        
        # Объяснение результатов
        explanations = analyzer.generate_explanations(features, text)
        
        # Статистика по тексту
        text_stats = {
            'length': len(text),
            'sentences': features.get('sentence_count', 0),
            'words': features.get('word_count', 0),
            'avg_sentence_length': round(features.get('avg_words_per_sentence', 0), 1)
        }
        
        result = {
            'success': True,
            'reliability_score': reliability_score,
            'is_fake': prediction['is_fake'],
            'fake_probability': round(prediction['fake_probability'] * 100, 1),
            'raw_fake_probability': round(prediction.get('raw_probability', 0) * 100, 1),
            'highlighted_text': highlighted_text,
            'features': features,
            'visualization_data': visualization_data,
            'explanations': explanations,
            'clickbait_words': features.get('clickbait_words', []),
            'certainty_words': features.get('certainty_words', []),
            'credibility_assessment': credibility_assessment,
            'text_stats': text_stats,
            'model_confidence': round(analyzer.model_accuracy * 100, 1)
        }
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Ошибка анализа: {str(e)}")
        return jsonify({'error': f'Внутренняя ошибка сервера: {str(e)}'})

@app.route('/features', methods=['GET'])
def get_feature_info():
    """Возвращает информацию о признаках для обучения"""
    features_info = {
        'features': [
            {'name': 'Кликбейт', 'description': 'Наличие слов, привлекающих внимание'},
            {'name': 'Эмоциональность', 'description': 'Сила эмоциональной окраски текста'},
            {'name': 'Категоричность', 'description': 'Степень уверенности в утверждениях'},
            {'name': 'Формальность', 'description': 'Официальность стиля изложения'},
            {'name': 'Источники', 'description': 'Упоминание источников информации'},
            {'name': 'Баланс', 'description': 'Сбалансированность изложения'}
        ],
        'model_info': {
            'name': 'Градиентный бустинг (Gradient Boosting)',
            'features_count': 15,
            'samples_trained': 1000,
            'accuracy': round(analyzer.model_accuracy * 100, 1)
        }
    }
    return jsonify(features_info)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка работоспособности сервиса"""
    return jsonify({
        'status': 'ok',
        'model_loaded': analyzer.model is not None,
        'model_accuracy': analyzer.model_accuracy if hasattr(analyzer, 'model_accuracy') else 0
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
