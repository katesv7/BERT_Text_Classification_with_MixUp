from src.preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    # Предобработка данных
    preprocess_data()

    # Обучение модели
    train_model(batch_size=8, epochs=3)

    # Оценка модели
    evaluate_model()

if __name__ == "__main__":
    main()
