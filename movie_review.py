import pandas as pd
import numpy as np
from numpy import array
from scipy.sparse import csr_matrix

dataset = pd.read_csv("IMDB Dataset.csv")
X = dataset.iloc[: , :-1]
y = dataset.iloc[:, 1:]

#clearing data

import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from nltk.corpus import stopwords
corpus = []
for i in range(0,len(X)):
    review = re.sub('[^A-Za-z]', ' ', X['review'][i])
    review = review.lower()
    review = nltk.word_tokenize(review)
    lm = WordNetLemmatizer()
    #ps = PorterStemmer()
    stop_words = stopwords.words('english')
    review = [lm.lemmatize(words) for words in review if not words in set(stop_words)]
    #review = [ps.stem(words) for words in review if not words not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X= cv.fit_transform(corpus)

X_dense = X.todense()

from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X_pca = pca.fit_transform(X_dense)
explained_variance = pca.explained_variance_ratio_

from sklearn.decomposition import TruncatedSVD
tsvd = TruncatedSVD(n_components = 500)
X_tsvd = tsvd.fit_transform(X)
explained_variance = tsvd.explained_variance_ratio_.sum()

#train_test_split
from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X_tsvd, y, test_size = 0.1)

from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_tsvd, y)

y_pred = dt_classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.model_selection import cross_val_score
accuracy_dt = cross_val_score(estimator = dt_classifier, X= X_tsvd, y= y, cv= 10).mean()

from xgboost import XGBClassifier
xg_classifier = XGBClassifier()
xg_classifier.fit(X_train, y_train)

y_pred = xg_classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

y_pred = xg_classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
