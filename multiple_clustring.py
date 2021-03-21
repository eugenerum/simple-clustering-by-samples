from sklearn.pipeline import FeatureUnion
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

#Get all features
#Text feature
col1 = Pipeline([
                ('selector', TextSelector(key = 'column_1')),
                ('tfidf', TfidfVectorizer(max_df=0.35, tokenizer=stemming_tokenizer))
            ])

col1.fit_transform(df) 

#Numerical feature
col2 =  Pipeline([
                ('selector', NumberSelector(key = 'column_2')),
                ('standard', StandardScaler())
            ])

col2.fit_transform(df)

feats = FeatureUnion([('col1', col1),          
                      ('col2', col2)])

feature_processing = Pipeline([('feats', feats)])

X = feature_processing.fit_transform(df).todense()
X_sparse_mtrix = feature_processing.fit_transform(df)

from sklearn.cluster import KMeans

pipeline = Pipeline([
    ('feats', feats),
    ('classifier', KMeans(n_clusters=5, random_state = 0)),
])

pipeline.fit(df)

preds = pipeline.predict(df)