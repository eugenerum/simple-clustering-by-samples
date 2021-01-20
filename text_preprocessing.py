import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

#Function to clean a column
def clean_data(df):
    df = df.str.lower()
    df.replace(r'[^a-zA-Z0-1]', ' ', regex=True, inplace=True)
    df = df.str.strip()
    stop_words = stopwords.words("english")
    df = df.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    return df
