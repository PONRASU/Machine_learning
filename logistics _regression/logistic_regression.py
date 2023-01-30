import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
    

class logistic_regression:
    def __init__(self,learning_rate=0.001,n_iter=1000):
            self.learning_rate = learning_rate
            self.n_iter = n_iter
            self.w = None
            self.bias = None
    def fit(self,X,y):
        samples,features = X.shape
        self.w = np.zeros((features))
        self.bias = 0
        for _ in range(self.n_iter):
            pred=np.dot(X,self.w)+self.bias
            preddiction=sigmoid(pred)
            
            dw=(1/samples)*np.dot(X.T,(y-preddiction))
            grade=(1/samples)*np.sum(preddiction-y)
            self.w=self.w-self.learning_rate*dw
            self.bias=self.bias-self.learning_rate*grade
    def predict(self,X):
        pred=np.dot(X,self.w)+self.bias
        y_pred=sigmoid(pred)
        class_pred=[0 if y<=0.5 else 1 for y in y_pred]
        return class_pred
    
        

