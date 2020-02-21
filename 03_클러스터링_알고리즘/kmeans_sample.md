#


### 1. 정수 벡터 클러스터링
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

```
from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],
             [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
kmeans.labels_
#array([1, 1, 1, 0, 0, 0], dtype=int32)
kmeans.predict([[0, 0], [12, 3]])
#array([1, 0], dtype=int32)
kmeans.cluster_centers_
#array([[10.,  2.],
#       [ 1.,  2.]])
```

### 2. 문장 클러스터링
```
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

docs = ['document format', 'list of str like']

# vectorizing
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

# training k-means
k = 2
kmeans = KMeans(n_clusters=k).fit(X)

# trained labels and cluster centers
labels = kmeans.labels_
print('labels:', labels)
centers = kmeans.cluster_centers_
print('centers:', centers)

#labels: [1 0]
#centers: [[0. 0. 1. 1. 1. 1.] [1. 1. 0. 0. 0. 0.]]
```
