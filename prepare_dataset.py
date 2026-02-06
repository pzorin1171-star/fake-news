# prepare_dataset.py - Создание датасета для обучения ML модели
import pandas as pd
import numpy as np
import os
import sys

def create_dataset(fake_path='Fake.csv', true_path='True.csv', output_path='fake_news_dataset.csv'):
    """
    Создает объединенный датасет из файлов Fake.csv и True.csv
    
    Args:
        fake_path (str): Путь к файлу с фейковыми новостями
        true_path (str): Путь к файлу с достоверными новостями  
        output_path (str): Путь для сохранения итогового датасета
    
    Returns:
        pd.DataFrame: Объединенный датасет
    """
    
    print("=" * 60)
    print("ПОДГОТОВКА ДАТАСЕТА ДЛЯ ОБУЧЕНИЯ ML МОДЕЛИ")
    print("=" * 60)
    
    # Проверяем наличие исходных файлов
    print("[1/6] Проверка наличия исходных файлов...")
    if not os.path.exists(fake_path):
        print(f"   ❌ Файл с фейковыми новостями не найден: {fake_path}")
        print("   Скачайте датасет с Kaggle:")
        print("   https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        print("   И поместите файлы Fake.csv и True.csv в папку с проектом")
        return None
    
    if not os.path.exists(true_path):
        print(f"   ❌ Файл с достоверными новостями не найден: {true_path}")
        return None
    
    print(f"   ✅ Файл фейковых новостей: {fake_path}")
    print(f"   ✅ Файл достоверных новостей: {true_path}")
    
    try:
        # Загружаем данные
        print("[2/6] Загрузка данных...")
        df_fake = pd.read_csv(fake_path, encoding='utf-8')
        df_true = pd.read_csv(true_path, encoding='utf-8')
        
        print(f"   Загружено фейковых новостей: {len(df_fake)} записей")
        print(f"   Загружено достоверных новостей: {len(df_true)} записей")
        
        # Проверяем структуру данных
        print("[3/6] Проверка структуры данных...")
        print(f"   Колонки в Fake.csv: {list(df_fake.columns)}")
        print(f"   Колонки в True.csv: {list(df_true.columns)}")
        
        # Определяем колонку с текстом (в этом датасете обычно 'text')
        text_column = None
        for col in ['text', 'content', 'article', 'title']:
            if col in df_fake.columns and col in df_true.columns:
                text_column = col
                break
        
        if text_column is None:
            print("   ⚠️ Не найдена общая колонка с текстом. Ищу альтернативы...")
            # Берем первую подходящую колонку
            for col in df_fake.columns:
                if col in df_true.columns:
                    text_column = col
                    print(f"   Использую колонку: {text_column}")
                    break
        
        if text_column is None:
            print("   ❌ Не найдена общая колонка в обоих датасетах!")
            print("   Доступные колонки в Fake.csv:", list(df_fake.columns))
            print("   Доступные колонки в True.csv:", list(df_true.columns))
            return None
        
        print(f"   ✅ Используется колонка с текстом: '{text_column}'")
        
        # Добавляем метки классов
        print("[4/6] Добавление меток классов...")
        df_fake['label'] = 1  # 1 = Фейковая новость
        df_true['label'] = 0  # 0 = Достоверная новость
        
        # Переименовываем текстовую колонку для единообразия
        if text_column != 'text':
            df_fake = df_fake.rename(columns={text_column: 'text'})
            df_true = df_true.rename(columns={text_column: 'text'})
        
        # Оставляем только нужные колонки (текст и метка)
        df_fake = df_fake[['text', 'label']]
        df_true = df_true[['text', 'label']]
        
        # Объединяем датасеты
        print("[5/6] Объединение датасетов...")
        df_combined = pd.concat([df_fake, df_true], ignore_index=True)
        
        # Проверяем и очищаем данные
        print("   Проверка качества данных...")
        initial_count = len(df_combined)
        
        # Удаляем пустые тексты
        df_combined = df_combined.dropna(subset=['text'])
        df_combined['text'] = df_combined['text'].astype(str).str.strip()
        df_combined = df_combined[df_combined['text'].str.len() > 10]
        
        cleaned_count = len(df_combined)
        removed_count = initial_count - cleaned_count
        
        if removed_count > 0:
            print(f"   Удалено некорректных записей: {removed_count}")
        
        # Перемешиваем данные
        df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Сохраняем датасет
        print("[6/6] Сохранение датасета...")
        df_combined.to_csv(output_path, index=False, encoding='utf-8')
        
        # Выводим статистику
        print("\n" + "=" * 60)
        print("✅ ДАТАСЕТ УСПЕШНО СОЗДАН")
        print("=" * 60)
        print(f"Итоговый файл: {output_path}")
        print(f"Общее количество записей: {len(df_combined)}")
        print(f"Фейковых новостей (label=1): {df_combined['label'].sum()}")
        print(f"Достоверных новостей (label=0): {len(df_combined) - df_combined['label'].sum()}")
        
        # Выводим примеры данных
        print("\n📊 Примеры данных:")
        print("-" * 40)
        
        # Показываем по 2 примера каждого класса
        for label_value, label_name in [(0, "Достоверные"), (1, "Фейковые")]:
            print(f"\n{label_name} новости (label={label_value}):")
            examples = df_combined[df_combined['label'] == label_value].head(2)
            for i, (_, row) in enumerate(examples.iterrows()):
                text_preview = row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
                print(f"  {i+1}. {text_preview}")
        
        print("\n" + "=" * 60)
        print("Следующие шаги:")
        print("1. Обучите модель: python train_model.py")
        print("2. Запустите приложение: python app.py")
        print("=" * 60)
        
        return df_combined
        
    except Exception as e:
        print(f"\n❌ ОШИБКА при создании датасета: {str(e)}")
        print("\nВозможные причины:")
        print("1. Файлы CSV имеют другую структуру или кодировку")
        print("2. Не хватает памяти")
        print("3. Проблемы с правами доступа к файлам")
        
        # Детальная информация об ошибке
        import traceback
        print("\nДетали ошибки:")
        traceback.print_exc()
        
        return None

def verify_dataset(dataset_path='fake_news_dataset.csv'):
    """
    Проверяет созданный датасет
    
    Args:
        dataset_path (str): Путь к датасету для проверки
    """
    if not os.path.exists(dataset_path):
        print(f"❌ Датасет не найден: {dataset_path}")
        return False
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"✅ Датасет загружен: {dataset_path}")
        print(f"   Записей: {len(df)}")
        print(f"   Колонки: {list(df.columns)}")
        
        if 'text' not in df.columns or 'label' not in df.columns:
            print("❌ В датасете отсутствуют необходимые колонки 'text' или 'label'")
            return False
        
        # Проверяем баланс классов
        label_counts = df['label'].value_counts()
        print(f"\n📈 Распределение классов:")
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            label_name = "Фейковые" if label == 1 else "Достоверные"
            print(f"   {label_name} (label={label}): {count} записей ({percentage:.1f}%)")
        
        # Проверяем длину текстов
        df['text_length'] = df['text'].astype(str).apply(len)
        print(f"\n📏 Статистика длины текстов:")
        print(f"   Средняя длина: {df['text_length'].mean():.0f} символов")
        print(f"   Минимальная длина: {df['text_length'].min()} символов")
        print(f"   Максимальная длина: {df['text_length'].max()} символов")
        
        # Проверяем наличие пустых значений
        empty_texts = df['text'].isnull().sum()
        if empty_texts > 0:
            print(f"⚠️  Найдено пустых текстов: {empty_texts}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при проверке датасета: {str(e)}")
        return False

def create_small_test_dataset(original_path='fake_news_dataset.csv', 
                             output_path='test_dataset.csv',
                             sample_size=1000):
    """
    Создает небольшой тестовый датасет для быстрого тестирования
    
    Args:
        original_path (str): Путь к исходному датасету
        output_path (str): Путь для сохранения тестового датасета
        sample_size (int): Размер тестового датасета
    """
    if not os.path.exists(original_path):
        print(f"❌ Исходный датасет не найден: {original_path}")
        return None
    
    try:
        df = pd.read_csv(original_path)
        
        # Берем пропорциональную выборку по классам
        sample_per_class = sample_size // 2
        df_fake = df[df['label'] == 1].sample(sample_per_class, random_state=42)
        df_true = df[df['label'] == 0].sample(sample_per_class, random_state=42)
        
        df_test = pd.concat([df_fake, df_true], ignore_index=True)
        df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)
        
        df_test.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"✅ Создан тестовый датасет: {output_path}")
        print(f"   Размер: {len(df_test)} записей")
        print(f"   Фейковые: {df_test['label'].sum()} записей")
        print(f"   Достоверные: {len(df_test) - df_test['label'].sum()} записей")
        
        return df_test
        
    except Exception as e:
        print(f"❌ Ошибка при создании тестового датасета: {str(e)}")
        return None

# Основная функция для запуска из командной строки
def main():
    """Основная функция для запуска скрипта"""
    
    # Проверяем аргументы командной строки
    import argparse
    parser = argparse.ArgumentParser(description='Создание датасета для детекции фейковых новостей')
    parser.add_argument('--fake', default='Fake.csv', help='Путь к файлу с фейковыми новостями')
    parser.add_argument('--true', default='True.csv', help='Путь к файлу с достоверными новостями')
    parser.add_argument('--output', default='fake_news_dataset.csv', help='Путь для сохранения датасета')
    parser.add_argument('--test', action='store_true', help='Создать тестовый датасет')
    parser.add_argument('--verify', action='store_true', help='Проверить существующий датасет')
    
    args = parser.parse_args()
    
    if args.verify:
        # Режим проверки датасета
        verify_dataset(args.output)
        return
    
    if args.test:
        # Режим создания тестового датасета
        if not os.path.exists(args.output):
            print(f"❌ Для создания тестового датасета сначала создайте основной датасет")
            print(f"   Запустите: python prepare_dataset.py")
            return
        
        create_small_test_dataset(args.output, 'test_dataset.csv')
        return
    
    # Режим создания основного датасета
    dataset = create_dataset(args.fake, args.true, args.output)
    
    if dataset is not None:
        # После создания проверяем датасет
        print("\n" + "=" * 60)
        print("ПРОВЕРКА СОЗДАННОГО ДАТАСЕТА")
        print("=" * 60)
        verify_dataset(args.output)

if __name__ == '__main__':
    main()
