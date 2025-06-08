import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("student-mat.csv", sep=";")
df['label'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
X = df[['G1', 'G2', 'absences', 'studytime']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
joblib.dump(lr, "lr_model.pkl")

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
joblib.dump(dt, "dt_model.pkl")