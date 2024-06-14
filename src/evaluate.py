import os
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk


def tokenize(batch):
    """
    Токенизация данных.

    Args:
        batch (dict): Батч данных

    Returns:
        dict: Токенизированные данные
    """
    return tokenizer(batch['clean_text'], padding='max_length', truncation=True, max_length=512)


def evaluate_model() -> None:
    """
    Функция для оценки модели.
    Результат выводится на экран.
    """
    global tokenizer
    # Загрузка токенизатора и модели
    tokenizer = BertTokenizer.from_pretrained(os.path.join('model'))
    model = BertForSequenceClassification.from_pretrained(os.path.join('model'))

    # Загрузка данных
    test_dataset = load_from_disk(os.path.join('data', 'test'))

    # Токенизация данных
    test_dataset = test_dataset.map(tokenize, batched=True)
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Определение функции вычисления метрик
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = (preds == labels).mean()
        return {"accuracy": acc}

    # Настройка аргументов для оценки
    training_args = TrainingArguments(
        output_dir=os.path.join('results'),
        per_device_eval_batch_size=64,
    )

    # Создание и оценка модели с помощью Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    eval_results = trainer.evaluate()
    print(eval_results)

if __name__ == "__main__":
    evaluate_model()
