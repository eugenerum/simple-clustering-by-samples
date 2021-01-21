from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english") 

#Stemming data to improve model quality
def stemming_tokenizer(text):
    text = re.split('\W+', text)
    text = [stemmer.stem(word) for word in text]
    return text
