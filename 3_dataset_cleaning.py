import nltk
import re  # to implement lambda functionality - read regular expressions
import string
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Importing the Dataset
dataset_columns = ['target', 'ids', 'date', ' flag', 'user', 'text']
dataset_encoding = "ISO-8859-1"
df = pd.read_csv("project_dataset.csv",
                 encoding=dataset_encoding, names=dataset_columns)

# data cleaning start from here
data = df[['target', 'text']]
data['target'] = data['target'].replace(4, 1)
# print(data['target'].unique())

# Seperating the Positive and Negative Tweet Data
data_pos = data[data['target'] == 1]
data_neg = data[data['target'] == 0]
# print(data_neg,data_pos)

# Taking Less values for calculation purpose
data_pos = data_pos.iloc[:int(20000)]
data_neg = data_neg.iloc[:int(20000)]

# Making text in lower case for consistency

dataset = pd.concat([data_pos, data_neg])
dataset['text'] = dataset['text'].str.lower()
# print(dataset['text'].tail())

# Tryig to remove stopwords - common words with no conclusiveness
stopwordlist = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
                'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
                'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
                'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
                'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
# print(stopwordlist)

# removing stop words using lambda

STOPWORD = set(stopwordlist)


def cleaning_stopword(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORD])


dataset['text'] = dataset['text'].apply(lambda text: cleaning_stopword(text))
# print(dataset['text'].tail())

# cleaning and removing punctuation
english_punctuations = string.punctuation
punctuation_list = english_punctuations


def cleaning_punctuation(text):
    translator = str.maketrans('', '', punctuation_list)
    return text.translate(translator)


dataset['text'] = dataset['text'].apply(lambda x: cleaning_punctuation(x))
# print(dataset['text'].tail())

# Removing repeating characters


def cleaning_repeating_char(text):
    return re.sub(r'(.)1+', r'1', text)


dataset['text'] = dataset['text'].apply(lambda x: cleaning_repeating_char(x))
# print(dataset['text'].tail())


# Removing the urls

def cleaning_urls(data):
    return re.sub('((www.[^s]+)|(https://[^s]+))', ' ', data)


dataset['text'] = dataset['text'].apply(lambda x: cleaning_urls(x))
# print(dataset['text'].tail())


# Removing numbers
def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)


dataset['text'] = dataset['text'].apply(lambda x: cleaning_numbers(x))
# print(dataset['text'].tail())

# Tokenization


def tokenization(text):
    text = re.split('\W', text)
    return text


dataset['text'] = dataset['text'].apply(lambda x: tokenization(x.lower()))
# print(dataset['text'].tail())

# Stemming --> it'll combine the similar type of word in a tree
st = nltk.PorterStemmer()


def stemming_on_text(data):
    text = [st.stem(word) for word in data]
    return data


dataset['text'] = dataset['text'].apply(lambda x: stemming_on_text(x))
# print(dataset['text'].head())


# ------ Visualization -----

X = data.text
y = data.target


# Word cloud for negative words
# data_neg = data['text'][:800000]
# plt.figure(figsize = (20,20))
# wc = WordCloud(max_words = 1000, width = 1600, height = 800, collocations = False).generate(" ".join(data_neg))
# plt.imshow(wc)
# plt.show()

# Word cloud for positive words
# data_pos = data['text'][800000:]
# plt.figure(figsize = (20,20))
# wc = WordCloud(max_words = 1000, width = 1600, height = 800, collocations = False).generate(" ".join(data_pos))
# plt.imshow(wc)
# plt.show()

#  ---- building testing and training data ------

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

# for training data and 5% for testing data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.05, random_state=26105111)

# Training the model
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)
# print("No. of Feature_words: ", len(vectoriser.get_feature_names()))

X_train = vectoriser.transform(X_train)
X_test = vectoriser.transform(X_test)


# ----Model Evaluation----
# Accuracy Score
# Confusion Matrix with Plot
# ROC-AUC Curve

def model_evaluation(model):
    # Preddict values for test dataset
    y_pred = model.predict(X_test)
    # print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))

    # Compute and plot the confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Neagtive', 'Positive']
    group_name = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentage = ['{0:.2%}'.format(value) for value in cf_matrix.feature_extraction]