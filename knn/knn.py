import numpy as np
from collections import Counter

def euclidean_distance(a, b):
    dis=np.sqrt(np.sum((a-b)**2))
    return dis
    

class knn:

    def __init__(self,k=3):
        self.k = k
    def fit(self,X,Y):
        self.X_train = X
        self.Y_train = Y
    def predict(self,X):
            predictions=[self.prediction(X) for x in X]
            return predictions


    def prediction(self,X):
        distance=[euclidean_distance(X,x_train) for x_train in self.X_train]

        k_indice=np.argsort(distance)[:self.k]
        labels=[self.Y_train[i] for i in k_indice]
        mostcomman=Counter(labels).most_common()
        return mostcomman


