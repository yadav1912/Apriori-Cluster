# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 17:57:51 2020

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, 
                min_support = 0.003,
                min_confidence = 0.2,
                min_lift = 3,
                min_length = 2)

# Visualising the results
MB = list(rules)
Result = [list(MB[i][0]) for i in range(0,len(MB))]