import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'Outlook':['Rainy','Rainy','Overcast','Sunny','Sunny','Sunny','Overcast','Rainy','Rainy','Sunny','Rainy','Overcast','Overcast','Sunny'],
    'Temperature':['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild'],
    'Humidity':['High','High','High','High','Normal','Normal','Normal','High','Normal','Normal','Normal','High','Normal','High'],
    'Windy':[False,True,False,False,False,True,True,False,False,False,True,True,False,True],
    'Play Golf':['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Encode categorical data
label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Separate features and target
X = df[['Outlook','Temperature','Humidity','Windy']]
y = df['Play Golf']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=64)

# Train Naive Bayes Classifier
model = CategoricalNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the Naive Bayes Classifier:", accuracy * 100)

# Decode predictions and actual values for better readability
y_test_decoded = label_encoders['Play Golf'].inverse_transform(y_test)
y_pred_decoded = label_encoders['Play Golf'].inverse_transform(y_pred)

# Display predictions vs actual values
result_df = pd.DataFrame({
    'Actual': y_test_decoded,
    'Predicted': y_pred_decoded
})

print("\nPredictions vs Actual values:")
print(result_df)

/*write a program to implement the nave bayesian classifier for a sample training dataset stored as a .csv file 
compute the accuracy of classifier considering few test data sets*/
/*
Accuracy of the Naive Bayes Classifier: 80.0

Predictions vs Actual values:
  Actual Predicted
0    Yes       Yes
1    Yes       Yes
2    Yes        No
3    Yes       Yes
4     No        No
*/
