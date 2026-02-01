from flask import Flask, render_template, request, jsonify
from style_analyzer import StyleAnalyzer
import os
import logging

app = Flask(__name__)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация анализатора
analyzer = StyleAnalyzer()

@app.before_first_request
def initialize_model():
    """Инициализация модели при первом запросе"""
    try:
        # Создаем папку models если её нет
        if not os.path.exists('models'):
            os.makedirs('models', exist_ok=True)
            logger.info("📁 Создана папка models/")
        
        # Загружаем или обучаем модель
        analyzer.load_or_train_model()
        logger.info(f"✅ Модель инициализирована. Точность: {analyzer.model_accuracy:.2%}")
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации модели: {e}")

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Анализ текста"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Введите текст для анализа'})
        
        if len(text) < 20:
            return jsonify({'error': 'Текст слишком короткий (минимум 20 символов)'})
        
        if len(text) > 5000:
            return jsonify({'error': 'Текст слишком длинный (максимум 5000 символов)'})
        
        # Проверяем, готова ли модель
        if analyzer.model is None:
            return jsonify({'error': 'Модель еще загружается. Пожалуйста, подождите...'})
        
        # Анализ
        features = analyzer.extract_features(text)
        prediction = analyzer.predict(features)
        highlighted_text = analyzer.highlight_text(text)
        reliability_score = analyzer.calculate_reliability_score(features, prediction, text)
        explanations = analyzer.generate_explanations(features, text)
        
        # Подготовка данных для визуализации
        visualization_data = {
            'clickbait_score': round(features['clickbait_score'] * 100),
            'emotional_score': round(features['emotional_score'] * 100),
            'certainty_score': round(features['certainty_score'] * 100),
            'formality_score': round(features['formality_score'] * 100),
            'source_score': round(features['source_indicator_score'] * 100),
            'balance_score': round(features['balance_score'] * 100),
        }
        
        result = {
            'success': True,
            'reliability_score': reliability_score,
            'is_fake': prediction['is_fake'],
            'fake_probability': round(prediction['fake_probability'] * 100, 1),
            'highlighted_text': highlighted_text,
            'features': features,
            'visualization_data': visualization_data,
            'explanations': explanations,
            'model_accuracy': round(analyzer.model_accuracy * 100, 1),
            'text_stats': {
                'length': len(text),
                'sentences': features.get('sentence_count', 0),
                'words': features.get('word_count', 0),
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Ошибка анализа: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'})

@app.route('/health')
def health_check():
    """Проверка здоровья сервиса"""
    return jsonify({
        'status': 'ok',
        'model_loaded': analyzer.model is not None,
        'model_accuracy': analyzer.model_accuracy
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
