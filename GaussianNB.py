import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

# Step 1: Load the dataset--multinomial and bernouli naive bayes
data = pd.read_csv("enjoysport.csv")

# Step 2: Convert categorical values to numbers
le = LabelEncoder()
for column in data.columns:
    data[column] = le.fit_transform(data[column])

# Step 3: Separate features and target
X = data.iloc[:, :-1]   # all columns except last
y = data.iloc[:, -1]    # last column (target)

# Step 4: Create Naive Bayes model
model = GaussianNB()

# Step 5: Train the model
model.fit(X, y)

# Step 6: Predict using a sample input
test = X.iloc[[0]]   # using first row as example input
prediction = model.predict(test)

# Step 7: Display results
print("Dataset:\n", data)
print("\nPrediction", prediction)

/*Dataset:
    sky  air_temp  humidity  wind  water  forecast  enjoy_sport
0    1         1         1     0      1         1            1
1    1         1         0     0      1         1            1
2    0         0         0     0      1         0            0
3    1         1         0     0      0         0            1

Prediction [1]
*/
