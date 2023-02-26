import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Importing the Dataset
dataset_columns = ['target', 'ids', 'date',' flag', 'user', 'text']
dataset_encoding = "ISO-8859-1"
df = pd.read_csv("project_dataset.csv", encoding = dataset_encoding, names = dataset_columns)


# Plotting the distribution for dataset
ax = df.groupby('target').count().plot(kind='bar', title='distribution of the DataSet', legend = False)
ax.set_xticklabels(['Negative', 'Positive'], rotation = 0)

# Storing the Data in Lists
text, sentiment = list(df['text']), list(df['target'])
# plt.show()

# Plotting the same thing with seaborn library
sns.countplot(x = 'target', data = df)
plt.show()