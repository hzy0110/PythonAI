import numpy as np
from sklearn import datasets

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    pass  # TODO: Compute and return softmax(x)
    ex = np.exp(x)/sum(np.exp(x))
    return ex

scores = np.array([3.0, 1.0, 0.2])

print(softmax(scores*10))
print(softmax(scores/10))
