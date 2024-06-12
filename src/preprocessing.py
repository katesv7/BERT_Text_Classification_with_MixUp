import os
import string
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset, Dataset

def preprocess_text(text: str) -> str:
    """
    Предобработка текста: приведение к нижнему регистру, удаление пунктуации, лемматизация и удаление стоп-слов.

    Args:
        text (str): Исходный текст

    Returns:
        str: Очищенный текст
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    clean_text = ' '.join(tokens)
    return clean_text

def preprocess_data() -> None:
    """
    Основная функция для предобработки данных. Загружает данные, применяет предобработку текста и сохраняет предобработанные данные.
    """
    # Загрузка необходимых данных для NLTK
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    global lemmatizer, stop_words
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Загрузка данных
    dataset = load_dataset('rotten_tomatoes')

    # Предобработка данных
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    validation_df = pd.DataFrame(dataset['validation'])

    train_df['clean_text'] = train_df['text'].apply(preprocess_text)
    test_df['clean_text'] = test_df['text'].apply(preprocess_text)
    validation_df['clean_text'] = validation_df['text'].apply(preprocess_text)

    # Создание директорий и сохранение предобработанных данных
    stages = {'train': train_df, 'test': test_df, 'val': validation_df}
    for stage, df in stages.items():
        creation_path = os.path.join('data', stage)
        if not os.path.exists(creation_path):
            os.makedirs(creation_path)
        
        dataset = Dataset.from_pandas(df)
        dataset.save_to_disk(creation_path)

if __name__ == "__main__":
    preprocess_data()   
