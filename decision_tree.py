import numpy as np
from decision_node import DecisionNode


class DecisionTree:
    
    def __init__(self, max_depth=float('Inf')):
        self.max_depth = max_depth
        self.root = None   
    
    def fit(self, X, y):
        self.root = self.add_child(np.c_[X, y], 0)
    
    
    def add_child(self, data, depth):
        if data.shape[0]==0:
            return None
        if depth >= self.max_depth:
            return self.make_leaf(data)

        col, val, data_t, data_f = best_split_rf(data)
        child_t = self.add_child(data_t, depth+1)
        child_f = self.add_child(data_f, depth+1)
        
        if (child_t == None) and (child_f != None):
            return self.make_leaf(data_f)
        if (child_f == None) and (child_t != None):
            return self.make_leaf(data_t)
        if (child_t == None) and (child_f == None):
            return self.make_leaf(data)

        node = DecisionNode()
        if child_t.is_leaf() and child_f.is_leaf() and child_t.label==child_f.label:
            node.label = child_t.label
        else:
            node.col, node.val, node.child_t, node.child_f = col, val, child_t, child_f
        return node 
    
    #Метод, который создает листовой узел с прогнозируемым классом.
    def make_leaf(self, data):
        labels = data[:,-1].tolist()
        node = DecisionNode()
        node.label = max(set(labels), key=labels.count)
        return node
    

    def predict(self, X):
        y = np.array([self.node_search(self.root, row) for row in X])
        return y
    
    def node_search(self, node, sample):
        if node.is_leaf():
            return node.label
            
        if sample[node.col] < node.val:
            return self.node_search(node.child_t, sample)
        else:
            return self.node_search(node.child_f, sample)                   


def best_split_rf(data):
    b_loss = float('Inf')
    b_col = b_val = None
    b_data_t = b_data_f = np.array([])
   
    cols = np.random.choice(np.arange(data.shape[1]-1), 4, replace=False)

    for col in cols:
        feature_vals = np.sort(np.unique(data[:,col]))
        #ищем среднее у соседних точек
        midpoints = (feature_vals[1:] + feature_vals[:-1]) / 2.

        for val in midpoints:
            data_t = data[data[:,col] < val]
            data_f = data[data[:,col] >= val]
            loss = gini(data_t[:,-1], data_f[:,-1])
            if loss < b_loss:
                b_loss, b_col, b_val, b_data_t, b_data_f = loss, col, val, data_t, data_f

    return (b_col, b_val, b_data_t, b_data_f)


def gini(d1, d2):

    n1, n2 = d1.shape[0], d2.shape[0]
    g1 = 1 - np.sum((np.unique(d1, return_counts=True)[1] / n1)**2)
    g2 = 1 - np.sum((np.unique(d2, return_counts=True)[1] / n2)**2)
    return (g1*n1 + g2*n2) / (n1 + n2)

