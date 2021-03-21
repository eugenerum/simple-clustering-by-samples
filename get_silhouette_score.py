from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state = 0).fit(X)
data2D = pca.transform(X)

from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=7, random_state = 0)
kmeans_fit = kmeans_model.fit_predict(data2D)
print("silhouette_score", " ", silhouette_score(data2D, kmeans_fit))