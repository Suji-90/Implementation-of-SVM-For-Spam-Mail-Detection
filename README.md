# Implementation-of-SVM-For-Spam-Mail-Detection
### NAME: SUJITHRA K
### REG NO: 212223040212
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import libraries.

2.Read the CSV file and display data using head().

3.Split the dataset using train_test_split().

4.Calculate predictions and accuracy.

5.Print the outputs.

6.End the program. 

## Program:
```
Program to implement the SVM For Spam Mail Detection..
```
```
import chardet, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn import metrics

# Detect encoding
with open('spam.csv', 'rb') as f:
    print(chardet.detect(f.read(100000)))
```
## Output:
![image](https://github.com/user-attachments/assets/2d196618-556e-451c-9aa8-e8e970c80abe)

```
# Load data
data = pd.read_csv('spam.csv', encoding='windows-1252')
print(data.head())
print(data.info())
print(data.isnull().sum())
```
## Output:
![image](https://github.com/user-attachments/assets/55665a0b-689b-44ae-8561-483bf09d0244)

```
# Split data
x = data['v1'].values   # Labels
y = data['v2'].values   # Messages
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Text vectorization
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

# Train & predict
model = SVC()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Predictions:", y_pred)
```
## Output:
![image](https://github.com/user-attachments/assets/7975b42d-4809-4dd6-8d42-2ee31c397c87)

```
# Accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```
## Output:
![image](https://github.com/user-attachments/assets/06373ef4-f452-4fd7-a0bc-c32427fb59b3)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
