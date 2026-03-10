import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

# Load dataset
df = pd.read_csv("enjoysport.csv")

# Encode categorical values
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

# Split features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train model
model = GaussianNB()
model.fit(X, y)

print("Model trained successfully")

test = pd.DataFrame([[1,0,1,2,0,1]], columns=X.columns)
prediction = model.predict(test)

print("Prediction:", prediction)
