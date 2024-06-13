## BERT Text Classification with MixUp Data Augmentation 

This repository implements a pipeline for fine-tuning `bert-base-cased` model on the Rotten Tomatoes dataset for text classification, using a MixUp data augmentation technique at the embedding level. The MixUp technique helps improve the model's generalization by creating virtual training examples.

### Project Structure

- `main.py`: Coordinates the preprocessing, training, and evaluation of the model.
- `src/preprocessing.py`: Contains functions for preprocessing the dataset.
- `src/train.py`: Handles the training of the model with MixUp augmentation.
- `src/evaluate.py`: Evaluates the trained model on the test set.
- `src/mixup.py`: Implements the MixUp data augmentation technique.
- `requirements.txt`: Lists the dependencies required for the project.

### Dataset

The Rotten Tomatoes dataset is used for this project. It can be accessed from [Hugging Face Datasets](https://huggingface.co/datasets/rotten_tomatoes).

#### Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/katesv7/BERT_Text_Classification_with_MixUp.git
    cd BERT_Text_Classification_with_MixUp
    ```
2. **Create and activate a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate 
    ```

3. **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```
### References
- [MixUp Data Augmentation](https://huggingface.co/collections/stereoplegic/data-augmentation-655385cce37d1644856eeeb8)
- [BERT Base Cased Model](https://huggingface.co/bert-base-cased)
- [Rotten Tomatoes Dataset](https://huggingface.co/datasets/rotten_tomatoes)
