import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GuassianNB
from sklearn import svm
from sklearn.model_selection import train_test_split

"""
- Reading and writing CSV files.
- Data Understanding and Data Exploration:
- Generating word cloud using Python wordcloud library
- Data cleaning: Removing punctuation, stop words
- Applying Count Vector Word Embedding Technique
- Applying Naive Bayes and Support Vector Machine (SVM)
- Evaluation: Accuracy and F1 Score, Confusion Matrix, ROC, AUC Curve
"""
#Step1:Reading the dataset
spam = pd.read_csv('webspam.csv')
print(spam.head())
#Split into traning and test data
z = spam['Emailtext']
y = spam['Label']
z_train, z_test,y_train, y_test = train_test_split(z,y,test_size = 0.2)
#
cv = CountVectorizer() 

features = cv.fit_transform(z_train)

