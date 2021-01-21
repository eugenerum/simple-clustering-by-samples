import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer

#A vectorizer for count words in text
cvzer = CountVectorizer(stop_words = stop_words, min_df = 0.1, max_df = 0.25)
vec_dictionary = cvzer.fit_transform(df["X"])
#Getting words set
my_dict = cvzer.vocabulary_

#Writing vocabulary to csv
#You will need 'wb' mode in Python 2.x
with open('mycsvfile.csv', 'w', encoding='latin-1', newline='') as f: 
    w = csv.DictWriter(f,my_dict.keys())
    w.writeheader()
    w.writerow(my_dict)
