import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Step 1: Load data
data = pd.read_csv("data.csv")

# Step 2: Split input and output
X = data.drop("focus", axis=1)
y = data["focus"]

# Step 3: Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 4: Create model
model = DecisionTreeClassifier()

# Step 5: Train model
model.fit(X_train, y_train)

# Step 6: Check accuracy
accuracy = model.score(X_test, y_test)
print(f"Model trained and saved successfully! Accuracy: {accuracy:.2f}")

# Step 7: Save model
joblib.dump(model, "model.pkl")