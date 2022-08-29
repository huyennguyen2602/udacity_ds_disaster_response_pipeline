#!/usr/bin/env python
# coding: utf-8

### 1. Import libraries and load data from database.

# import libraries
import pandas as pd
import numpy as np
import re
import sys
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.simplefilter("ignore")

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
import pickle
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'



### 2. Write a function to load your data
def load_data(database_path):
    """load data from database
    INPUT
    database_path - Path to the database created in the data processes 
    OUTPUT
    X - features dataset
    y - Target variables dataset
    """
    name = 'sqlite:///' + database_path
    engine = create_engine(name)
   
    df = pd.read_sql_table('clean_dataset', con=engine).head(10000)
    X = df['message']
    y = df.drop(['original','genre','message','offer','request'], axis=1)
    y = y.astype(int)
    return X, y


### 3. Write a tokenization function to process your text data

def tokenize(text):
    """ Clean data and transform the text into tokens
    INPUT
    text - the data to be cleaned
    OUTPUT
    clean_tokens - the data cleaned and tokenized
    """
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


### 4. Build a class for extracting the starting verb of a text, creating a new feature for the ML classifier.
    
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


    
### 5. Build a machine learning pipeline

def model_pipeline():
    """
    We create a pipeline from where the model shall be run
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
    ])

    return pipeline


### 6. Build a machine learning model using the pipeline created before and grid searching

def build_model():
    """Return Grid Search model with pipeline and Classifier"""

    model = model_pipeline()

    parameters = {'clf__estimator__max_depth': [10, 50, None],
                  'clf__estimator__min_samples_leaf':[2, 5, 10]}

    model = GridSearchCV(model, parameters)
    
    return model


### 7. Write a function to obtain the f1 score, precision and recall for each output category of the dataset.

def get_results(y_test, y_pred):
    """ Obtain the metrics to evaluate the model
    INPUT
    y_test - the real values of the prediction
    y_pred - the prediction by the model
    OUTPUT
    results - an object with the f-score, precision and recall values
    """
    results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])
    num = 0
    for cat in y_test.columns:
        precision, recall, f_score, support = precision_recall_fscore_support(y_test[cat], y_pred[:,num], average='weighted')
        results.set_value(num+1, 'Category', cat)
        results.set_value(num+1, 'f_score', f_score)
        results.set_value(num+1, 'precision', precision)
        results.set_value(num+1, 'recall', recall)
        num += 1
    print('Aggregated f_score:', results['f_score'].mean())
    print('Aggregated precision:', results['precision'].mean())
    print('Aggregated recall:', results['recall'].mean())
    #return results


### 8.Create the main class

def main():
    """Load the data, run the model and save model"""
    if len(sys.argv) == 3:
        database_path, model_path = sys.argv[1:]
          
        print('Loading data...')
        X, y = load_data(database_path)
              
        print('Splitting data...')
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        y_pred = model.predict(X_test)
        get_results(y_test, y_pred)
        
        
        pickle.dump(model, open('models/classifier.pkl', 'wb'))
        print('Trained model saved!')
       
    else:
        print('Error, missing file paths')


if __name__ == '__main__':
    main()


