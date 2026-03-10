import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import BernoulliNB

# Step 1: Load the dataset--multinomial and bernouli naive bayes
data = pd.read_csv("Iris.csv")

# Step 2: Convert categorical values to numbers
le = LabelEncoder()
for column in data.columns:
    data[column] = le.fit_transform(data[column])

# Step 3: Separate features and target
X = data.iloc[:, :-1]   # all columns except last
y = data.iloc[:, -1]    # last column (target)

# Step 4: Create Naive Bayes model
model = BernoulliNB()

# Step 5: Train the model
model.fit(X, y)

# Step 6: Predict using a sample input
test = X.iloc[[0]]   # using first row as example input
prediction = model.predict(test)

# Step 7: Display results
print("Dataset:\n", data)
print("\nPrediction", prediction)

/*
Dataset:
       Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm  Species
0      0              8            14              4             1        0
1      1              6             9              4             1        0
2      2              4            11              3             1        0
3      3              3            10              5             1        0
4      4              7            15              4             1        0
..   ...            ...           ...            ...           ...      ...
145  145             24             9             28            19        2
146  146             20             4             26            15        2
147  147             22             9             28            16        2
148  148             19            13             30            19        2
149  149             16             9             27            14        2

[150 rows x 6 columns]

Prediction [0]
*/
