from sklearn.preprocessing import LabelEncoder
from pandas import read_csv, DataFrame, Series
import sklearn.linear_model as lm
import numpy as np
import math
import itertools
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

data = read_csv('training_set111.csv', sep=';', encoding='latin1')
data.columns.str.lower()

MaxHypertension = data.groupby('Hypertension').count()['N']

data["Glucose_Fasting"] = [float(str(i).replace(",", ".")) for i in data["Glucose_Fasting"]]
data.loc[data.Glucose_Fasting.isnull(), 'Glucose_Fasting'] = data.Glucose_Fasting.mean()

data["Total_Cholesterol"] = [float(str(i).replace(",", ".")) for i in data["Total_Cholesterol"]]

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
x = np.array([[test.Glucose_Fasting], [test.Total_Cholesterol], [test.Systolic_Blood_Pressure], [test.Diastolic_Blood_Pressure], [test.BMI], [test.Smoking_Status], [test.Patient_Gender], [test.Age], [test.Hypertension]])

x = x.reshape(x.shape[0], -1)
y = y.reshape(y.shape[1])
x = x.transpose()
sizeStr = len(x)
sizeCol = len(x[0])
N = 8
y_last = []
models = []
skm = lm.LinearRegression()


def combinations_liner_model(count):
    global models
    l1 = itertools.combinations([0, 1, 2, 3, 4, 5, 6, 7], count)
    listComb = list(l1)
    y_score_model = []
    for i in range(len(listComb)):
        for j in range(count):
            x1 = x[:, listComb[i][j]]
            x1 = x1.reshape(len(x1), -1)
            if j == 0:
                x2 = x1
            else:
                x2 = np.hstack((x2, x1))
        skm.fit(x2, y)
        r = skm.score(x2, y)
        if r >= 0.6:
            y_score_model = y_score_model + [skm.predict(x2)]
            #print("model: y = b0 + b1x1 + b2x2 + ... + e")
            models = models + [listComb[i]]
            #print("column number = " + str(listComb[i]))
            #print("coefficient of determination = ", r)
    if y_score_model:
        return y_score_model
    else:
        return []


#model: y = b0 + b1x1 + b2x2 + ... + e
def first():
    global models
    models = models + ["y = b0 + b1x1 + b2x2 + ... + e"]
    for i in range(1, N + 1):
        y_new_mass = combinations_liner_model(i)
        if y_new_mass:
            global y_last
            y_last = y_last + y_new_mass


first()

#model: y = b0 + b1xi + b2xi^2 + e
def second_model():
    global models
    y_score_model = []
    for i in range(sizeCol):
        c = i + 1
        x2_first = x[:, i]
        b = np.power(x2_first, 2)
        x2_first = x2_first.reshape(len(x2_first), -1)
        b = b.reshape(len(b), -1)
        x2_first = np.hstack((x2_first, b))
        skm.fit(x2_first, y)
        r = skm.score(x2_first, y)
        if r >= 0.6:
            y_score_model = y_score_model + [skm.predict(x2_first)]
            #print("model: y = b0 + b1x" + str(c) + " + b2x" + str(c) + "^2 + e")
            #print("coefficient of determination = ", r)
    if y_score_model:
        return y_score_model
    else:
        return []


def quadratic():
    y_new_mass = second_model()
    if y_new_mass:
        global y_last
        y_last = y_last + y_new_mass


quadratic()


def combinations_ln_model(count):
    l1 = itertools.combinations([0, 1, 2, 3, 4, 5, 6, 7], count)
    listComb = list(l1)
    global models
    y_score_model = []
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
        r = skm.score(x_ln, y)
        if r >= 0.6:
            y_score_model = y_score_model + [skm.predict(x_ln)]
            models = models + [listComb[i]]
            #print("model: y = b0 + b1*lnx1 + b2*lnx2 + ... + e")
            #print("column number = " + str(listComb[i]))
            #print("coefficient of determination = ", r)
    if y_score_model:
        return y_score_model
    else:
        return []


#model: y = b0 + b1lnx1 + ... + e
def second():
    global models
    models = models + ["y = b0 + b1lnx1 + ... + e"]
    for i in range(1, N + 1):
        y_new_mass = combinations_ln_model(i)
        if y_new_mass:
            global y_last
            y_last = y_last + y_new_mass


second()


def combinations_quadratic_model(count):
    l1 = itertools.combinations([0, 1, 2, 3, 4, 5, 6, 7], count)
    global models
    listComb = list(l1)
    y_score_model = []
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
        r = skm.score(x_q, y)
        if r >= 0.6:
            y_score_model = y_score_model + [skm.predict(x2)]
            models = models + [listComb[i]]
            #print("model: y = b0 + b1x1^2 + b2x2^2 + ... + e")
            #print("column number = " + str(listComb[i]))
            #print("coefficient of determination = ", r)
    if y_score_model:
        return y_score_model
    else:
        return []


#model: y = b0 + b1x1^2 + b2x2^2 + ... + e
def third():
    global models
    models = models + ["y = b0 + b1x1^2 + b2x2^2 + ... + e"]
    for i in range(1, N + 1):
        y_new_mass = combinations_quadratic_model(i)
        if y_new_mass:
            global y_last
            y_last = y_last + y_new_mass


third()

y_score = np.empty([0, len(y_last[0])])

for i in range(len(y_last)):
    y_score = np.vstack((y_score, np.array(y_last[i])))

y_score = y_score.transpose()

print("model: y = b0 + b1x1 + b2x2 + ... + e")
skm.fit(x, y)
print("last coefficient of determination = ", skm.score(x, y))

skm.fit(y_score, y)
print("new coefficient of determination = ", skm.score(y_score, y))

model1 = sm.OLS(y, y_score)
res = model1.fit()
d = durbin_watson(res.resid)
print("durbin watson = ", d)