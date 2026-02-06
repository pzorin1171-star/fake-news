# create_small_datasets.py
from datasets import load_dataset
import pandas as pd
import os

def create_datasets(output_fake='Fake.csv', output_true='True.csv', target_size_mb=24):
    """
    Создает два CSV-файла с фейковыми и правдивыми новостями из датасета LIAR2.
    Файлы создаются такого размера, чтобы каждый был меньше target_size_mb МБ.
    """
    print("Загрузка датасета LIAR2...")
    try:
        dataset = load_dataset("chengxuphd/liar2")
        print("Датасет успешно загружен.")
    except Exception as e:
        print(f"Ошибка загрузки датасета: {e}")
        return

    all_fake_texts = []
    all_true_texts = []

    # Проходим по всем разделам датасета (train, validation, test)
    for split in ['train', 'validation', 'test']:
        print(f"Обработка раздела: {split}")
        for item in dataset[split]:
            label = item['label']
            # Основной текст утверждения
            text = str(item['statement']).strip()
            # Добавляем контекст (обоснование), чтобы текст был информативнее
            justification = str(item.get('justification', '')).strip()
            full_text = f"{text} {justification}".strip()

            # Преобразуем метки LIAR2 в бинарные:
            # 0, 1 ('pants-fire', 'false') -> ФЕЙК (1)
            # 2, 3, 4, 5 ('barely-true', 'half-true', 'mostly-true', 'true') -> ПРАВДА (0)
            # Это стандартное преобразование для этого датасета.
            if label in [0, 1]:
                all_fake_texts.append(full_text)
            else:
                all_true_texts.append(full_text)

    print(f"\nВсего собрано: {len(all_fake_texts)} фейковых и {len(all_true_texts)} правдивых утверждений.")

    # Контролируем размер файлов: берем только часть данных
    records_per_file = 8000  # Начните с этого числа. Каждая запись ~2-3 КБ, итого ~20 МБ.
    fake_subset = all_fake_texts[:records_per_file]
    true_subset = all_true_texts[:records_per_file]

    # Сохраняем в DataFrame и затем в CSV
    df_fake = pd.DataFrame({'text': fake_subset})
    df_true = pd.DataFrame({'text': true_subset})

    df_fake.to_csv(output_fake, index=False, encoding='utf-8')
    df_true.to_csv(output_true, index=False, encoding='utf-8')

    # Проверяем размер файлов
    size_fake_mb = os.path.getsize(output_fake) / (1024 * 1024)
    size_true_mb = os.path.getsize(output_true) / (1024 * 1024)

    print(f"\nФайлы созданы:")
    print(f"  {output_fake}: {len(df_fake)} записей, {size_fake_mb:.2f} МБ")
    print(f"  {output_true}: {len(df_true)} записей, {size_true_mb:.2f} МБ")

    if size_fake_mb > target_size_mb or size_true_mb > target_size_mb:
        print(f"\n⚠️  Внимание: один из файлов превысил {target_size_mb} МБ.")
        print("   Уменьшите параметр 'records_per_file' в скрипте и запустите его снова.")
    else:
        print(f"\n✅ Оба файла меньше {target_size_mb} МБ и готовы к загрузке в GitHub.")

    # Показываем первые записи для проверки
    print("\nПример фейковой новости:")
    print(fake_subset[0][:200], "...")
    print("\nПример правдивой новости:")
    print(true_subset[0][:200], "...")

if __name__ == '__main__':
    create_datasets(target_size_mb=24)
