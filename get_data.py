import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

# If there's no dataset in the project directory, create a reasonably large one.
# If it exists, append some new observations.
if os.path.isfile("data.csv.gzip"):
    n = 1
else:
    n = 50

for i in range(0, n):
    X, y = make_regression(10000, n_features=10)
    df = pd.DataFrame(X)
    df.to_csv("data.csv.gzip", mode="a", index=None, compression="gzip")
