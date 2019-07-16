from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
text = ["London Paris Paris", "Paris London London"]
cv = CountVectorizer()

count_matrix = cv.fit_transform(text)
similarity_scores = cosine_similarity(count_matrix)
print (similarity_scores)
print(cv.get_feature_names())
print (count_matrix.toarray())