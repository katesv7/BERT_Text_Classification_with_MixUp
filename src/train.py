import os
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments
from datasets import load_from_disk
from src.mixup import MixupTrainer


def tokenize(batch):
    """
    Токенизация данных.

    Args:
        batch (dict): Батч данных

    Returns:
        dict: Токенизированные данные
    """
    return tokenizer(batch['clean_text'], padding='max_length', truncation=True, max_length=512)


def train_model(batch_size: int, epochs: int) -> None:
    """
    Функция для запуска тренировки модели.
    Результат сохраняется в файл.

    Args:
        batch_size (int): Размер батча для обучения
        epochs (int): Количество эпох для обучения
    """
    global tokenizer
    # Загрузка токенизатора и данных
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    train_dataset = load_from_disk(os.path.join('data', 'train'))
    validation_dataset = load_from_disk(os.path.join('data', 'val'))

    # Токенизация данных
    train_dataset = train_dataset.map(tokenize, batched=True)
    validation_dataset = validation_dataset.map(tokenize, batched=True)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Загрузка модели
    model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)

    # Настройка аргументов тренировки
    training_args = TrainingArguments(
        output_dir=os.path.join('results'),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join('logs'),
        logging_steps=10,
        evaluation_strategy="epoch",
    )

    # Создание тренера с использованием Mixup
    trainer = MixupTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )

    # Тренировка и оценка модели
    trainer.train()
    eval_results = trainer.evaluate()
    print(eval_results)

    # Сохранение модели и токенизатора
    model.save_pretrained(os.path.join('model'))
    tokenizer.save_pretrained(os.path.join('model'))

if __name__ == "__main__":
    train_model(batch_size=8, epochs=3)
