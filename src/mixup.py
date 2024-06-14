import numpy as np
import torch
import torch.nn.functional as F
from transformers import Trainer, BertTokenizer, BertForSequenceClassification
from datasets import load_from_disk


def mixup_embeddings(embeddings, labels, alpha=0.2):
    """
    Реализация MixUp для эмбеддингов.

    Args:
        embeddings (torch.Tensor): Эмбеддинги
        labels (torch.Tensor): Метки
        alpha (float): Параметр для распределения Бета

    Returns:
        Tuple: Смешанные эмбеддинги и метки
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = embeddings.size(0)
    index = torch.randperm(batch_size)
    mixed_embeddings = lam * embeddings + (1 - lam) * embeddings[index, :]
    labels_a, labels_b = labels, labels[index]
    return mixed_embeddings, labels_a, labels_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Критерий для MixUp.

    Args:
        criterion: Функция потерь
        pred (torch.Tensor): Предсказания
        y_a (torch.Tensor): Метки a
        y_b (torch.Tensor): Метки b
        lam (float): Коэффициент MixUp

    Returns:
        float: Значение потерь
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class MixupTrainer(Trainer):
    """
    Кастомный Trainer с поддержкой MixUp.
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        embeddings = model.bert.embeddings.word_embeddings(inputs['input_ids'])
        mixed_embeddings, labels_a, labels_b, lam = mixup_embeddings(embeddings, labels)

        outputs = model(inputs_embeds=mixed_embeddings, attention_mask=inputs['attention_mask'])
        logits = outputs.logits

        loss = mixup_criterion(F.cross_entropy, logits, labels_a, labels_b, lam)

        return (loss, outputs) if return_outputs else loss


def main() -> None:
    """
    Основная функция для тестирования MixUp.
    """
    # Загрузка токенизатора и модели
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)

    # Загрузка предобработанных данных
    dataset = load_from_disk('data/train')
    train_dataset = dataset.map(lambda batch: tokenizer(batch['clean_text'], padding='max_length', truncation=True, max_length=512), batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Выбор одного батча данных для тестирования MixUp
    sample_batch = next(iter(train_dataset))
    embeddings = model.bert.embeddings.word_embeddings(sample_batch['input_ids'])
    labels = sample_batch['label']
    
    mixed_embeddings, labels_a, labels_b, lam = mixup_embeddings(embeddings, labels)
    print(mixed_embeddings, labels_a, labels_b, lam)

if __name__ == "__main__":
    main()
