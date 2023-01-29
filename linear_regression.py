import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


class linear_regression:
    def fit(x, y):
        prod=0
        n=len(x)-1
        x_sq=0
        y_sq=0
        y_sum=0
        x_sum=0

        for i in range(n):
            prod+=x[i]*y[i]
            x_sq+=x[i]**2
            y_sq+=y[i]**2
            y_sum+=y[i]
            x_sq+=x[i]
        prod=prod/n
        x_sq=x_sq/n
        y_sq=y_sq/n
        y_sum=y_sum/n
        x_sum=x_sum/n
        #------------------------------------------------------------------------
        
        m=((n*prod)-(x_sum-y_sum))/((n*x_sq)-(x_sq))    
        c=((y_sum*x_sq)-(prod*x_sum))/((n*x_sq)-x_sq)



        max_x= np.max(x)
        min_x=np.min(x)

        X=np.linspace(min_x,max_x,5) 
        Y=m*X+c
        print(X,Y)
    
        plt.plot(X,Y)
        plt.scatter(x,y)
        plt.show()      

data=pd.read_csv('data.csv')
x=data['Head Size(cm^3)'].values
y=data['Brain Weight(grams)'].values
lr=linear_regression
lr.fit(x,y)


    



