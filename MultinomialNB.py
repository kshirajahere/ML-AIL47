import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

# Step 1: Load the dataset--multinomial and bernouli naive bayes
data = pd.read_csv("weather.csv")

# Step 2: Convert categorical values to numbers
le = LabelEncoder()
for column in data.columns:
    data[column] = le.fit_transform(data[column])

# Step 3: Separate features and target
X = data.iloc[:, :-1]   # all columns except last
y = data.iloc[:, -1]    # last column (target)

# Step 4: Create Naive Bayes model
model = MultinomialNB()

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
       Temperature  Humidity  Wind_Speed  Cloud_Cover  Pressure  Rain
0            1367      2152         923         1275      1897     1
1            1773       595         768          138       465     0
2            1500      1924         182          403      1009     0
3            1360      1630         888         1683        98     1
4            1068      2397         610         1186        23     0
...           ...       ...         ...          ...       ...   ...
2495         1180       549        1478         1398      1386     0
2496         1733       593        1366          971      1058     0
2497         1797       495         378         1891       720     0
2498          480      1024         308           73      2391     0
2499         1632      2406        2339         2031       822     0

[2500 rows x 6 columns]

Prediction [0]
*/
