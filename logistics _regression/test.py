import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from logistic_regression import logistic_regression



def accuracy(y_test,y_pred):
    return np.mean(y_test==y_pred)/len(y_test)

data=datasets.load_breast_cancer()
X,Y=data.data,data.target
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=1234)
logistic_regrsssor=logistic_regression()
logistic_regrsssor.fit(X_train,y_train)
y_pr=logistic_regrsssor.predict(X_test)

acc=accuracy(y_test,y_pr)
print("Accuracy",acc)

# plt.plot(X_train,y_train)
# plt.scatter(X_test,y_pr)
# plt.show()
