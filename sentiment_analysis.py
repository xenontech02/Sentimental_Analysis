import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import nltk

# Download required NLTK data
required_nltk_resources = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'wordnet', 'stopwords']
for resource in required_nltk_resources:
    try:
        nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else resource)
    except LookupError:
        nltk.download(resource)

from file_loader import load_text_file
from preprocessing import preprocess_text
from model_evaluation import evaluate_model

# Load text data from file
text_data = load_text_file()

# Sample data
data = {
    'text': text_data.splitlines(),
    'sentiment': [1 if i % 2 == 0 else 0 for i in range(len(text_data.splitlines()))]  # Sample sentiment labels
}

# Create DataFrame
df = pd.DataFrame(data)

# Apply preprocessing
df['processed_text'] = df['text'].apply(preprocess_text)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['sentiment'], test_size=0.3, random_state=42)

# Logistic Regression
logistic_regression_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression()),
])

logistic_regression_pipeline.fit(X_train, y_train)
print("Logistic Regression Evaluation:")
evaluate_model(logistic_regression_pipeline, X_test, y_test)

# Naive Bayes
naive_bayes_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

naive_bayes_pipeline.fit(X_train, y_train)
print("\nNaive Bayes Evaluation:")
evaluate_model(naive_bayes_pipeline, X_test, y_test)

# Support Vector Machines
svm_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SVC()),
])

svm_pipeline.fit(X_train, y_train)
print("\nSupport Vector Machines Evaluation:")
evaluate_model(svm_pipeline, X_test, y_test)