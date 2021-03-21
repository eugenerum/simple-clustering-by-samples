import numpy as np 
import pandas as pd

import sklearn 
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split

from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        return X[self.key]
    
class NumberSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        return X[[self.key]]
        
from nltk.stem.snowball import SnowballStemmer
import re
stemmer = SnowballStemmer("english") 

def stemming_tokenizer(text):
    text = re.split('\W+', text)
    text = [stemmer.stem(word) for word in text]
    return text

X = df[["column_1", "column_2"]]
y = df["preds"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

#Get all features
#Text feature
col1 = Pipeline([
                ('selector', TextSelector(key = 'column_1')),
                ('tfidf', TfidfVectorizer(max_df=0.35, tokenizer=stemming_tokenizer))
            ])

col1.fit_transform(df).todense()

#Numerical feature
col2 =  Pipeline([
                ('selector', NumberSelector(key = 'column_2')),
                ('standard', StandardScaler())
            ])

col2.fit_transform(df)

feats = FeatureUnion([('col1', col1),          
                      ('col2', col2)])

feature_processing = Pipeline([('feats', feats)])

feature_processing.fit_transform(X_train).todense()

pipeline_SVC = Pipeline([
    ('features', feats),
    ('classifier', SVC(random_state = 0)),
])

pipeline_SVC.fit(X_train, y_train)

preds = pipeline_SVC.predict(X_test)
print(classification_report(y_test, preds))