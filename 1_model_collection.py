import pandas as pd
import numpy as np

# Importing the Dataset
dataset_columns = ['target', 'ids', 'date',' flag', 'user', 'text']
dataset_encoding = "ISO-8859-1"
df = pd.read_csv("project_dataset.csv", encoding = dataset_encoding, names = dataset_columns)
# print(df.head())


# Checking some info in the dataset
# print(df['target'].unique().tolist())

# print(df.shape)

# print(df.info())

# print(np.sum(df.isnull().any(axis=1)))

