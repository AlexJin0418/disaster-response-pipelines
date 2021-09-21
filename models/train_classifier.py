import sys
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

import pickle

def load_data(database_filepath):
    """Load data from database"""
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disasterTable', engine)
    X = df['message']
    y = df.iloc[:, 4:]
    category_names = df.columns[4:]
    return X, y, category_names

def tokenize(text):
    """Text processing"""
    text = text.lower() # convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text) # remove punctuation characters
    tokens = word_tokenize(text) # tokenize
    words = [w for w in tokens if w not in stopwords.words('english')] # remove stop words
    lemmatizer = WordNetLemmatizer() # reduce words to their root form
    lemmed = [lemmatizer.lemmatize(w) for w in words]
    return lemmed

def build_model():
    """Build a machine learning pipeline"""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [100, 200],
        'clf__estimator__max_features': ['auto', 'sqrt']
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, y_test, category_names):
    """Test the model and report the results"""
    y_pred = model.predict(X_test)
    for i in range(y_test.shape[1]):
        print(category_names[i])
        print(classification_report(y_test.iloc[:, i], y_pred[:, i]))

def save_model(model, model_filepath):
    """Export the model as a pickle file"""
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()