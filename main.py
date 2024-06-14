from src.preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model


def main():
    preprocess_data()
    train_model(batch_size=8, epochs=3)
    evaluate_model()


if __name__ == "__main__":
    main()
