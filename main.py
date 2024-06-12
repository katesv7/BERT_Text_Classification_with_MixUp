import cv2

from src.EDA import EDA
from src.train import train_model


def main():
    EDA()
    train_model(8, 100)


if __name__ == "__main__":
    main()

