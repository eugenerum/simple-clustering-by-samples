from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

#Vectorization (transformation) data for the model
vectorizer = TfidfVectorizer(use_idf=True, min_df = 0.1, max_df = 0.25, ngram_range = (1,1))
X_sparse_mtrix = vectorizer.fit_transform(df["X"])
words = vectorizer.get_feature_names()

#K-Means Algorithm for Determining a Cluster of Words
kmeans = KMeans(n_clusters = 5, random_state = 567, max_iter=1000, n_init=20)
kmeans.fit(X_sparse_mtrix)
kmeans_fit = kmeans.fit_predict(X_sparse_mtrix)

#Outputting grouped words
common_words = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))
    
#Naming clusters
df['clusters'] = kmeans_fit

d = {0:"rare", 1:"medium rare",
     2:"medium", 3:"medium well", 
     4:"well done"}

df["roast"] = df["clusters"].replace(d)