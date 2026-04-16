import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

data = pd.read_csv("data.csv")

X = data.drop("focus", axis=1)
y = data["focus"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model trained and saved successfully! Accuracy: {accuracy:.2f}")

joblib.dump(model, "model.pkl")
