# Plagarism-Detector
## Overview
This project is aimed at detecting plagiarism in text using both traditional machine learning models and deep learning models. The dataset used contains pairs of original and plagiarized sentences with corresponding labels indicating whether the text is plagiarized or not.

## Dataset
The dataset used in this project is sourced from the **SNLI dataset**. It consists of the following columns:
- **Original**: The original text.
- **Plagiarized**: The text suspected of being plagiarized.
- **Label**: Binary label indicating whether the text is plagiarized (1) or not (0).

### Data Preprocessing
Before training the models, the text data is preprocessed through the following steps:
- **Lowercasing**: Converting all text to lowercase.
- **Punctuation Removal**: Removing all punctuation marks.
- **Stopwords Removal**: Eliminating common English stopwords.
- **Stemming**: Reducing words to their base form using the PorterStemmer.

## Models Used

### 1. **Random Forest Classifier**
- **Description**: A traditional machine learning model that uses an ensemble of decision trees to classify the text.
- **Training Accuracy**: ~86%
- **Testing Accuracy**: ~70%
- **Evaluation**: The model's performance was evaluated using a confusion matrix and classification report.

### 2. **LSTM with Bidirectional Layers**
- **Description**: A deep learning model that uses Long Short-Term Memory (LSTM) units to capture the sequential dependencies in text data.
- **Architecture**:
  - **Embedding Layer**: Converts words into dense vectors.
  - **SpatialDropout1D**: Reduces overfitting by randomly setting a fraction of input units to zero at each update during training.
  - **Bidirectional LSTM**: Captures information from both forward and backward directions in the text.
  - **Dense Layers**: Fully connected layers with ReLU activation.
  - **Output Layer**: A sigmoid activated layer for binary classification.
- **Training Accuracy**: ~86%
- **Validation Accuracy**: ~85%
- **Evaluation**: Evaluated using a confusion matrix and F1-score.

## Results
- **Random Forest Classifier**: The model achieved a testing accuracy of approximately 70%, with a balanced performance across precision and recall metrics.
- **LSTM Model**: Achieved a validation accuracy of ~85%, showing the model's ability to generalize well to unseen data.

## Visualizations
- **Count Plot of Labels**: Visualizes the distribution of labels in the dataset.
- **Word Cloud**: Displays the most frequent words in the original text.
- **Confusion Matrices**: Generated for both models to visualize their performance in classifying plagiarized vs. original text.

## Installation

### Prerequisites
- Python 3.7+
- Required libraries: `pandas`, `nltk`, `scikit-learn`, `tensorflow`, `keras`, `matplotlib`, `seaborn`, `wordcloud`

### Steps
1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/plagiarism-detector.git
    ```
2. Navigate to the project directory:
    ```sh
    cd plagiarism-detector
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
4. Download NLTK resources:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

## Contributing
If you want to contribute to this project, feel free to submit pull requests. Please ensure your code follows the project guidelines and is well documented.
