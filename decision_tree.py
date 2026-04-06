import numpy as np
from collections import Counter

class CARTClassifier:
    def __init__(self):
        self.root = None

    def gini(self,y):
        counter = Counter(y)
        g = 1.0
        for count  in counter.values():
            p = count / len(y)
            g -= p ** 2
        return g
    def split(self,X,y,feat_idx,threshold):
        left = X[:,feat_idx] <= threshold
        right = X[:,feat_idx] > threshold

        return X[left],y[left],X[right],y[right]

    def best_split(self,X,y):
        min_gini = float('inf')
        best_feat = 0
        best_thresh = 0
        n_feat = X.shape[1]

        for f in range(n_feat):
            unique_vals = np.unique(X[:,f])
            for val in unique_vals:
                X1,y1,X2,y2 = self.split(X,y,f,val)
                if len(y1) == 0 or len(y2) == 0:
                    continue
                current_gini = len(y1)/len(y)*self.gini(y1) +len(y2)/len(y)*self.gini(y2)

                if current_gini < min_gini:
                    min_gini = current_gini
                    best_feat = f
                    best_thresh = val
        return best_feat,best_thresh

    def build_tree(self,X,y,depth=0,max_depth=5):
        if len(y) == 0:
            return None
        if len(set(y)) == 1:
            return Counter(y).most_common(1)[0][0]
        if depth >= max_depth:
            return Counter(y).most_common(1)[0][0]
        if len(y) <= 2:
            return Counter(y).most_common(1)[0][0]

        feat,thresh = self.best_split(X,y)
        X1,y1,X2,y2 = self.split(X,y,feat,thresh)

        return{
            'feat':feat,
            'thresh':thresh,
            'left':self.build_tree(X1,y1,depth=depth+1,max_depth=max_depth),
            'right':self.build_tree(X2,y2,depth=depth+1,max_depth=max_depth)
        }

    def fit(self,X,y):
        self.root = self.build_tree(np.array(X),np.array(y))

    def _predict(self,node,x):
        if node is None or not isinstance(node,dict):
            return 0
        if x[node['feat']] <= node['thresh']:
            return self._predict(node['left'],x)
        else:
            return self._predict(node['right'],x)

    def predict(self,X):
        return [self._predict(self.root,x) for x in np.array(X)]
        
if __name__ == '__main__':
    X_clf = [[0,0],[0,0],[1,0],[2,1],[2,2],[2,2],[1,2]]
    y_clf = [0,0,1,1,1,0,1]
    clf = CARTClassifier()
    clf.fit(X_clf,y_clf)
    print(clf.root)
    print(clf.predict([[0,0]]))