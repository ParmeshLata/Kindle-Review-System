### LOADING THE DATASET
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import re
import streamlit as st
import pandas as pd

data=pd.read_csv("all_kindle_review.csv")
data.head()
df=data[["reviewText", "rating"]]
df.head()
df.loc[:, "rating"]=df["rating"].apply(lambda x:0 if x<3 else 1)
df.loc[:, "reviewText"] = df["reviewText"].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s-]", "", str(x)))
df.loc[:, 'reviewText']=df['reviewText'].apply(lambda x:re.sub(r"[^a-zA-Z0-9\s-]", '',x))
df.loc[:, 'reviewText']=df['reviewText'].apply(lambda x:" ".join([y for y in x.split() if y not in stopwords.words('english')]))
df.loc[:, 'reviewText']=df['reviewText'].apply(lambda x: re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , str(x)))
df.loc[:, 'reviewText']=df['reviewText'].apply(lambda x: " ".join(x.split()))
x_train, x_test, y_train, y_test=train_test_split(df["reviewText"], df["rating"], test_size=0.20)
cv=CountVectorizer()
x_train_bow=cv.fit_transform(x_train).toarray()
x_test_bow=cv.transform(x_test).toarray()
model=GaussianNB()
model_bow=model.fit(x_train_bow, y_train)
y_test_pred_bow=model_bow.predict(x_test_bow)
cm_bow=confusion_matrix(y_test_pred_bow, y_test)
acc_bow=accuracy_score(y_test_pred_bow, y_test)
report_bow=classification_report(y_test_pred_bow, y_test)
st.title("KINDLE REVIEW SYSTEM")
review=st.text_input("Enter your Review Below", height=150)
review=str(review).lower()
review=re.sub("[^a-z A-Z 0-9-]+", "",review)
review=" ".join([y for y in review.split() if y not in stopwords.words("english")])
review=re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , str(review))
review=" ".join(review.split())
vector_review=cv.transform([review]).toarray()
user_review=model_bow.predict(vector_review)
if user_review[0]==1:
    st.write("Positive Review")
else:
    st.write("Negetive Review")