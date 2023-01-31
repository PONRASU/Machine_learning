import numpy as np
from collections import Counter
class node:
    def __init__(self,feature=None,threshold=None,right=None,left=None,*,value=None):
            self.feature = feature
            self.threshold = threshold
            self.right = right
            self.left = left
            self.value = value
        
    def isleaf(self):
        return self.value is not None

class decision_tree:
    def __init__(self,max_depth=None,min_samples_split=None,features=None):

        self.max_depth =max_depth
        self.min_samples_split =min_samples_split
        self.features = features
        self.root = None
    def fit(self,X,y):
        self.features = X.shape[1] if not self.features else min(X.shape[1], self.features)
        self.root = self.grow_tree(X,y)
    def grow_tree(self,X,y,depth=0):
        n_samples ,n_feats= X.shape
        n_labels=len(np.unique(y))

        #----------------------------------
        if (depth>=self.max_depth) or (n_samples<self.min_samples_split)or(n_labels==1):
            leaf_value= self.mo_co_val(y)
            return node(value=leaf_value)
        feat_index = np.random.choice(n_feats, self.features, replace=False)
        
        bestf,best_t=self.best_threshold(X,y,feat_index)
    def best_threshold(self,X,y,feat_index):
        best_gain=-1
        split_idx,split_threshold=None,None
        for feat_idx in feat_index:
            X_colum=X[:,feat_idx]
            threshold=np.unique(X_colum)
            for thr in threshold:
                gain=self.information_gain(X_colum,y,thr)

                if gain>best_gain:
                    best_gain=gain
                    split_idx=feat_idx
                    split_threshold=thr
        return split_idx,split_threshold
    
    def information_gain(self,X_colum,y,thr):
        parent_entropy=self.entropy(y)
        left_idx,right_idx= self.split(X_colum,y,thr)

        if



    def split(self,X_colum,thr):
        left_idx=np.argwhere(X_colum<=thr)
        right_inx=np.argwhere(X_colum>thr)
        return left_idx,right_inx



    def entropy(self,y):
        hist=np.bincount(y)
        p=hist/len(y)
        return -np.sum([p*np.log2(p) for pin in p if pin>0])









         


    def mo_co_val(self,y):
        counter=Counter()
        value=counter.most_common(1)[0][0]
        return value
                