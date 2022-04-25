from sklearn.preprocessing import LabelEncoder
from pandas import read_csv, DataFrame, Series
import sklearn.linear_model as lm
import numpy as np
import math
import itertools


data = read_csv('training_set123.csv', sep=';', encoding='latin1')
data.columns.str.lower()

MaxHypertension = data.groupby('Hypertension').count()['N']
data.loc[data.Hypertension.isnull(), 'Hypertension'] = MaxHypertension[MaxHypertension == MaxHypertension.max()].index[0]

data.loc[data.MDRD.isnull(), 'MDRD'] = data.MDRD.mean()

data["Glucose_Fasting"] = [float(str(i).replace(",", ".")) for i in data["Glucose_Fasting"]]
data.loc[data.Glucose_Fasting.isnull(), 'Glucose_Fasting'] = data.Glucose_Fasting.mean()

data["Total_Cholesterol"] = [float(str(i).replace(",", ".")) for i in data["Total_Cholesterol"]]

data.loc[data.Systolic_Blood_Pressure.isnull(), 'Systolic_Blood_Pressure'] = data.Systolic_Blood_Pressure.mean()
data.loc[data.Diastolic_Blood_Pressure.isnull(), 'Diastolic_Blood_Pressure'] = data.Diastolic_Blood_Pressure.mean()
data.loc[(data.Diastolic_Blood_Pressure == 0), 'Diastolic_Blood_Pressure'] = data.Diastolic_Blood_Pressure.mean()
data.loc[data.BMI.isnull(), 'BMI'] = data.BMI.mean()

MaxSmokingStatus = data.groupby('Smoking_Status').count()['N']
data.loc[data.Smoking_Status.isnull(), 'Smoking_Status'] = MaxSmokingStatus[MaxSmokingStatus == MaxSmokingStatus.max()].index[0]

label = LabelEncoder()
dicts = {}

label.fit(data.Hypertension.drop_duplicates())
dicts['Hypertension'] = list(label.classes_)
data.Hypertension = label.transform(data.Hypertension)
data.loc[(data.Hypertension == 0), 'Hypertension'] = data.Hypertension.max() + 1

label.fit(data.Patient_Gender.drop_duplicates())
dicts['Patient_Gender'] = list(label.classes_)
data.Patient_Gender = label.transform(data.Patient_Gender)
data.loc[(data.Patient_Gender == 0), 'Patient_Gender'] = data.Patient_Gender.max() + 1

label.fit(data.Smoking_Status.drop_duplicates())
dicts['Smoking_Status'] = list(label.classes_)
data.Smoking_Status = label.transform(data.Smoking_Status)
data.loc[(data.Smoking_Status == 0), 'Smoking_Status'] = data.Smoking_Status.max() + 1

test = data.drop(['N'], axis=1)
y = np.array([test.Risk_Score_CVRM])
x = np.array([[test.MDRD], [test.Glucose_Fasting], [test.Total_Cholesterol], [test.Systolic_Blood_Pressure], [test.Diastolic_Blood_Pressure], [test.BMI], [test.Smoking_Status], [test.Patient_Gender], [test.Age], [test.Hypertension]])
#x1,x2,x3,x4,x5,x6,x7,x8,x9,x10

x = x.reshape(x.shape[0], -1)
y = y.reshape(y.shape[1])
x = x.transpose()
skm = lm.LinearRegression()
sizeStr = len(x)
sizeCol = len(x[0])
N = 9


def combinations_liner_model(count):
    l1 = itertools.combinations([0, 1, 2, 3, 4, 5, 6, 7, 8], count)
    listComb = list(l1)
    for i in range(len(listComb)):
        for j in range(count):
            x1 = x[:, listComb[i][j]]
            x1 = x1.reshape(len(x1), -1)
            if j == 0:
                x2 = x1
            else:
                x2 = np.hstack((x2, x1))
        skm.fit(x2, y)
        print("column number = " + str(listComb[i]))
        print("coefficient of determination = ", skm.score(x2, y))


#model: y = b0 + b1x1 + b2x2 + ... + e
def first():
    print("model: y = b0 + b1x1 + b2x2 + ... + e")
    for i in range(1, N + 1):
        combinations_liner_model(i)


first()


#model: y = b0 + b1xi + b2xi^2 + e
def second_model():
    for i in range(sizeCol):
        c = i + 1
        print("model: y = b0 + b1x" + str(c) + " + b2x" + str(c) + "^2 + e")
        x2_first = x[:, i]
        b = np.power(x2_first, 2)
        x2_first = x2_first.reshape(len(x2_first), -1)
        b = b.reshape(len(b), -1)
        x2_first = np.hstack((x2_first, b))
        skm.fit(x2_first, y)
        print("coefficient of determination = ", skm.score(x2_first, y))


second_model()

def combinations_ln_model(count):
    l1 = itertools.combinations([0, 1, 2, 3, 4, 5, 6, 7, 8], count)
    listComb = list(l1)
    for i in range(len(listComb)):
        for j in range(count):
            x1 = x[:, listComb[i][j]]
            x1 = x1.reshape(len(x1), -1)
            if j == 0:
                x2 = x1
            else:
                x2 = np.hstack((x2, x1))
        x_ln = np.array([np.array([math.log(x2[k][m]) for k in range(sizeStr)]) for m in range(count)])
        x_ln = x_ln.transpose()
        skm.fit(x_ln, y)
        print("column number = " + str(listComb[i]))
        print("coefficient of determination = ", skm.score(x_ln, y))


#model: y = b0 + b1lnx1 + ... + e
def second():
    print("model: y = b0 + b1*lnx1 + b2*lnx2 + ... + e")
    for i in range(1, N + 1):
        combinations_ln_model(i)


second()


def combinations_quadratic_model(count):
    l1 = itertools.combinations([0, 1, 2, 3, 4, 5, 6, 7, 8], count)
    listComb = list(l1)
    for i in range(len(listComb)):
        for j in range(count):
            x1 = x[:, listComb[i][j]]
            x1 = x1.reshape(len(x1), -1)
            if j == 0:
                x2 = x1
            else:
                x2 = np.hstack((x2, x1))
        x_q = np.power(x2, 2)
        skm.fit(x_q, y)
        print("column number = " + str(listComb[i]))
        print("coefficient of determination = ", skm.score(x_q, y))


#model: y = b0 + b1x1^2 + b2x2^2 + ... + e
def third():
    print("model: y = b0 + b1x1^2 + b2x2^2 + ... + e")
    for i in range(1, N + 1):
        combinations_quadratic_model(i)


third()
