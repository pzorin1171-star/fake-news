# train_model.py
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def main():
    print("=" * 60)
    print("ОБУЧЕНИЕ УПРОЩЕННОЙ МОДЕЛИ ДЛЯ RENDER")
    print("=" * 60)
    
    # 1. Проверяем наличие датасета
    dataset_path = 'fake_news_dataset.csv'
    if not os.path.exists(dataset_path):
        print(f"[ОШИБКА] Файл {dataset_path} не найден.")
        print("Убедитесь, что сначала выполнили prepare_dataset.py")
        return
    
    # 2. Загрузка данных
    print("[1/4] Загрузка датасета...")
    try:
        df = pd.read_csv(dataset_path)
        # Предполагаем колонки 'text' и 'label'
        texts = df['text'].astype(str).fillna('').tolist()
        labels = df['label'].astype(int).values
        print(f"   Загружено записей: {len(texts)}")
    except Exception as e:
        print(f"[ОШИБКА] Не удалось загрузить датасет: {e}")
        return
    
    # 3. Разделение данных
    print("[2/4] Разделение данных...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"   Обучающая выборка: {len(X_train)} записей")
    print(f"   Тестовая выборка: {len(X_test)} записей")
    
    # 4. Создание и обучение модели
    print("[3/4] Обучение модели...")
    # Упрощенный пайплайн: векторизация + классификация
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=3000,      # Берем 3000 самых частых слов (для экономии памяти)
            stop_words='english',   # Убираем стоп-слова (для русского нужна своя настройка)
            ngram_range=(1, 2)      # Берем отдельные слова и пары слов
        )),
        ('clf', LogisticRegression(
            C=1.0,
            max_iter=1000,          # Увеличиваем итерации для надежности
            random_state=42,
            class_weight='balanced' # Балансируем классы
        ))
    ])
    
    # Обучение
    model_pipeline.fit(X_train, y_train)
    
    # 5. Оценка качества
    print("[4/4] Оценка модели...")
    train_score = model_pipeline.score(X_train, y_train)
    test_score = model_pipeline.score(X_test, y_test)
    
    print(f"   Точность на обучающих данных: {train_score:.4f}")
    print(f"   Точность на тестовых данных: {test_score:.4f}")
    
    # 6. Сохранение модели
    output_path = 'simple_model.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(model_pipeline, f)
    
    print("\n" + "=" * 60)
    print(f"✅ МОДЕЛЬ УСПЕШНО ОБУЧЕНА И СОХРАНЕНА В '{output_path}'")
    print("=" * 60)
    
    # Небольшой тест
    print("\n🧪 Тестовое предсказание:")
    test_samples = [
        "Breaking news: Scientists discover revolutionary new method",
        "SHOCKING: This one weird trick will make you rich overnight!"
    ]
    
    for sample in test_samples:
        prob = model_pipeline.predict_proba([sample])[0][1]
        label = model_pipeline.predict([sample])[0]
        verdict = "ФЕЙК" if label == 1 else "ДОСТОВЕРНО"
        print(f"   '{sample[:50]}...' → {verdict} (вероятность: {prob:.2%})")

if __name__ == '__main__':
    main()
