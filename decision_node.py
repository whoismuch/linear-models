import numpy as np
class DecisionNode:
    def __init__(self):
        self.col = None
        self.val = None
        self.child_t = None
        self.child_f = None
        self.label = None
    
    def is_leaf(self):

        if self.label == None:
            return False
        return True
