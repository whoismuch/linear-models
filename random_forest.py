import numpy as np
from decision_tree import DecisionTree


class RandomForest:
    def __init__(self, n_trees=50, max_depth=float('Inf')):
   
        self.max_depth = max_depth
        self.n_trees = n_trees
        self.trees = []        
    
    def fit(self, X, y):
        for i in range(self.n_trees):
            sample_idx = np.random.choice(X.shape[0], X.shape[0], replace=True)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X[sample_idx], y[sample_idx])
            self.trees.append(tree)        
            
    def predict(self, X):
        y = []
        for row in X:
            predictions = [t.predict([row])[0] for t in self.trees]
            y.append(max(set(predictions), key=predictions.count))
        return np.array(y) 
    
def accuracy(model, X_test, y_test):
    predictions = model.predict(X_test)
    return (np.array(predictions) == np.array(y_test)).mean()
    
