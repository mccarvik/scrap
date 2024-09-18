"""
Amanda Script
"""

# read the file BPA submissions into pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# read the file BPA submissions into pandas
df = pd.read_csv('BPA submissions.csv')
print(df.head())    