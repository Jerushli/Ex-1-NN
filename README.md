### ENTER YOUR NAME : JERUSHLIN JOSE JB
### ENTER YOUR REGISTER NO : 212222240039
### EX. NO.1
### DATE : 25/2/2024
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
``` PYTHON
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
data
data.head()

X=data.iloc[:,:-1].values
X

y=data.iloc[:,-1].values
y

data.isnull().sum()

data.duplicated()

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

X_train

X_test

print("Lenght of X_test ",len(X_test))


```


## OUTPUT:
### DATASET
![image](https://github.com/Jerushli/Ex-1-NN/assets/120041243/2f22317a-6f14-4cc2-b2c0-901f7b029b98)

### X VALUES
![image](https://github.com/Jerushli/Ex-1-NN/assets/120041243/c256d0fa-4d0f-4ea1-8fe7-68d34bd598bb)

### Y VALUES
![image](https://github.com/Jerushli/Ex-1-NN/assets/120041243/7175520b-2ca7-44b6-9a3c-d03192971364)

### NULL
![image](https://github.com/Jerushli/Ex-1-NN/assets/120041243/b27c4942-a597-4438-acf8-b1b936e60843)

### DUPLICATE
![image](https://github.com/Jerushli/Ex-1-NN/assets/120041243/6c81e437-f28e-49cc-b8f0-176ae7350822)

### DESCRIBE
![image](https://github.com/Jerushli/Ex-1-NN/assets/120041243/f81782d1-b10e-46d3-8e7d-b0c369deaf3d)

### DATASET AFTER DROPPING
![image](https://github.com/Jerushli/Ex-1-NN/assets/120041243/20cd3959-4439-4070-94e8-c4cdf079473d)

### NORMALIZE DATASET
![image](https://github.com/Jerushli/Ex-1-NN/assets/120041243/0ab4684b-fec6-4801-b009-e3370db004a4)

### X TRAIN
![image](https://github.com/Jerushli/Ex-1-NN/assets/120041243/4bcac946-8da5-4c58-945e-bfa994ed4b6a)

### X TEST
![image](https://github.com/Jerushli/Ex-1-NN/assets/120041243/f6cc3623-1444-4298-974c-e368339dfea1)

### LENGTH
![image](https://github.com/Jerushli/Ex-1-NN/assets/120041243/98c45985-a202-4608-ab70-403bd030a7f1)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


