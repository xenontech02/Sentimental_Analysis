# Sentiment Text Analysis

The Sentiment Analysis Project, implemented in Python, provides tools to classify the sentiment of text. By utilizing machine learning and natural language processing (NLP), this project allows you to analyze text and determine whether its sentiment is positive, negative, or neutral. This can be used for various applications, such as analyzing customer reviews or monitoring social media.

## Features
- Text preprocessing
  - Tokenization
  - Stopword removal
  - Stemming/Lemmatization
- Classification models
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machines
- Evaluation Metrics
  - Accuracy
  - Precision
  - Recall
  - F1-score

## Requirements
Ensure you have the following libraries installed. You can install them using pip from the `requirements.txt` file.
```sh
pip install -r requirements.txt
```
## Project Structure
+ `file_loader.py`: Contains the function to load a text file using a file dialog.
+ `preprocessing.py`: Contains the function to preprocess the text, including tokenization, stopword removal, and stemming.
+ `model_evaluation.py`: Contains the function to evaluate the models using accuracy, precision, recall, and F1-score.
+ `sentiment_analysis.py`: Main script that connects all components, loads the text file, preprocesses the text, trains classification models, and evaluates their performance.
+ `requirements.txt`: Lists the required Python libraries.

## How to Run
1. Clone the repository:
```sh
git clone https://github.com/xenontech02/Sentimental_Analysis.git
cd SentimentTextAnalysis
```
2. Install required libraries:
```sh
pip install -r requirements.txt
```
3. Run the main script:
```sh
python sentiment_analysis.py
```
A file dialog will appear allowing you to select a text file from your folder. The script will then perform sentiment analysis on the content of the selected file and print the evaluation results for Logistic Regression, Naive Bayes, and Support Vector Machines.

## Example Data
The script uses sample sentiment labels for the lines in the selected text file. You can modify the sentiment labels and the text preprocessing steps as needed.

## Contributing
Feel free to fork this repository and contribute by submitting pull requests. Any improvements and suggestions are welcome!

## Acknowledgments
- [NLTK](https://www.nltk.org/) for providing powerful text processing libraries.
- [Scikit-learn](https://scikit-learn.org/stable/) for offering robust machine learning tools.
- [Pandas](https://pandas.pydata.org/) for data manipulation and analysis.
   
