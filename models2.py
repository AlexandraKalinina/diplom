from sklearn.preprocessing import LabelEncoder
from pandas import read_csv, DataFrame, Series
import sklearn.linear_model as lm
import numpy as np
import math
import itertools
import statsmodels.api as sm
import statsmodels.stats.api as sms
import re
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
countModel = 0
models_step2 = []
number_models = []
skm_mass = []
skm_mass_2 = []
all_mod = []


def combinations_liner_model(count, x_mass, y_mass, column_mass, step):
    l1 = itertools.combinations(column_mass, count)
    listComb = list(l1)
    global number_models
    global skm_mass
    global all_mod
    global skm_mass_2
    y_score_model = []
    global countModel
    models2 = []
    c = 0
    for i in range(len(listComb)):
        for j in range(count):
            x1 = x_mass[:, listComb[i][j]]
            x1 = x1.reshape(len(x1), -1)
            if j == 0:
                x2 = x1
            else:
                x2 = np.hstack((x2, x1))
        skm = lm.LinearRegression()
        skm.fit(x2, y_mass)
        r = skm.score(x2, y_mass)
        if step == 1:
            if r >= 0.66:
                skm_mass = skm_mass + [skm]
                all_mod = all_mod + ["y = b0 + b1x1 + b2x2 + ... + e"]
                number_models = number_models + [str(listComb[i])]
                y_score_model = y_score_model + [skm.predict(x2)]
        elif step == 2:
            if r > 0.77:
                if c < countModel:
                    model1 = sm.OLS(y_mass, x2)
                    res = model1.fit()
                    d = durbin_watson(res.resid)
                    models2 = models2 + ["y = b0 + b1x1 + b2x2 + ... + e"]
                    models2 = models2 + [str(listComb[i])]
                    models2 = models2 + ["determination"]
                    models2 = models2 + [str(r)]
                    models2 = models2 + ["dw"]
                    models2 = models2 + [str(d)]
                    models2 = models2 + ["mult"]
                    mult = np.linalg.cond(res.model.exog)
                    models2 = models2 + [str(mult)]
                    test_br_pg = sms.het_breuschpagan(res.resid, res.model.exog)
                    models2 = models2 + ["heteroscedasticity"]
                    models2 = models2 + [str(test_br_pg)]
                    skm_mass_2 = skm_mass_2 + [res]
                    c = c + 1
                else:
                    return models2
    if step == 1:
        if y_score_model:
            return y_score_model
        else:
            return []
    elif step == 2:
        if models:
            return models2
        else:
            return []


#model: y = b0 + b1xi + b2xi^2 + e
def second_model(x_mass, y_mass, step):
    y_score_model = []
    global countModel
    global skm_mass
    global skm_mass_2
    global all_mod
    global number_models
    models2 = []
    c = 0
    for i in range(sizeCol):
        x2_first = x_mass[:, i]
        b = np.power(x2_first, 2)
        x2_first = x2_first.reshape(len(x2_first), -1)
        b = b.reshape(len(b), -1)
        x2_first = np.hstack((x2_first, b))
        skm = lm.LinearRegression()
        skm.fit(x2_first, y_mass)
        r = skm.score(x2_first, y_mass)
        if step == 1:
            if r >= 0.66:
                skm_mass = skm_mass + [skm]
                all_mod = all_mod + ["y = b0 + b1xi + b2xi^2 + e"]
                number_models = number_models + [str(i)]
                y_score_model = y_score_model + [skm.predict(x2_first)]
        elif step == 2:
            if r > 0.77:
                if c < countModel:
                    model1 = sm.OLS(y_mass, x2_first)
                    res = model1.fit()
                    d = durbin_watson(res.resid)
                    models2 = models2 + ["y = b0 + b1xi + b2xi^2 + e"]
                    models2 = models2 + [str(i)]
                    models2 = models2 + ["determination"]
                    models2 = models2 + [str(r)]
                    models2 = models2 + ["dw"]
                    models2 = models2 + [str(d)]
                    models2 = models2 + ["mult"]
                    mult = np.linalg.cond(res.model.exog)
                    models2 = models2 + [str(mult)]
                    test_br_pg = sms.het_breuschpagan(res.resid, res.model.exog)
                    models2 = models2 + ["heteroscedasticity"]
                    models2 = models2 + [str(test_br_pg)]
                    skm_mass_2 = skm_mass_2 + [res]
                    c = c + 1
    if step == 1:
        if y_score_model:
            return y_score_model
        else:
            return []
    elif step == 2:
        if models:
            return models2
        else:
            return []


def combinations_ln_model(count, x_mass, y_mass, column_mass, step):
    l1 = itertools.combinations(column_mass, count)
    listComb = list(l1)
    global countModel
    global skm_mass
    global skm_mass_2
    global all_mod
    models2 = []
    global number_models
    c = 0
    y_score_model = []
    for i in range(len(listComb)):
        for j in range(count):
            x1 = x_mass[:, listComb[i][j]]
            x1 = x1.reshape(len(x1), -1)
            if j == 0:
                x2 = x1
            else:
                x2 = np.hstack((x2, x1))
        x_ln = np.array([np.array([math.log(x2[l][m]) for l in range(sizeStr)]) for m in range(count)])
        x_ln = x_ln.transpose()
        skm = lm.LinearRegression()
        skm.fit(x_ln, y_mass)
        r = skm.score(x_ln, y_mass)
        if step == 1:
            if r >= 0.66:
                skm_mass = skm_mass + [skm]
                all_mod = all_mod + ["y = b0 + b1lnx1 + ... + e"]
                number_models = number_models + [str(listComb[i])]
                y_score_model = y_score_model + [skm.predict(x_ln)]
        elif step == 2:
            if r > 0.77:
                if c < countModel:
                    model1 = sm.OLS(y_mass, x_ln)
                    res = model1.fit()
                    d = durbin_watson(res.resid)
                    models2 = models2 + ["y = b0 + b1lnx1 + ... + e"]
                    models2 = models2 + [str(listComb[i])]
                    models2 = models2 + ["determination"]
                    models2 = models2 + [str(r)]
                    models2 = models2 + ["dw"]
                    models2 = models2 + [str(d)]
                    models2 = models2 + ["mult"]
                    mult = np.linalg.cond(res.model.exog)
                    models2 = models2 + [str(mult)]
                    test_br_pg = sms.het_breuschpagan(res.resid, res.model.exog)
                    models2 = models2 + ["heteroscedasticity"]
                    models2 = models2 + [str(test_br_pg)]
                    skm_mass_2 = skm_mass_2 + [res]
                    c = c + 1
                else:
                    return models2
    if step == 1:
        if y_score_model:
            return y_score_model
        else:
            return []
    elif step == 2:
        if models:
            return models2
        else:
            return []


def combinations_quadratic_model(count, x_mass, y_mass, column_mass, step):
    l1 = itertools.combinations(column_mass, count)
    global countModel
    c = 0
    global skm_mass
    global skm_mass_2
    global number_models
    global all_mod
    listComb = list(l1)
    y_score_model = []
    models2 = []
    for i in range(len(listComb)):
        for j in range(count):
            x1 = x_mass[:, listComb[i][j]]
            x1 = x1.reshape(len(x1), -1)
            if j == 0:
                x2 = x1
            else:
                x2 = np.hstack((x2, x1))
        x_q = np.power(x2, 2)
        skm = lm.LinearRegression()
        skm.fit(x_q, y_mass)
        r = skm.score(x_q, y_mass)
        if step == 1:
            if r >= 0.66:
                skm_mass = skm_mass + [skm]
                all_mod = all_mod + ["y = b0 + b1x1^2 + b2x2^2 + ... + e"]
                number_models = number_models + [str(listComb[i])]
                y_score_model = y_score_model + [skm.predict(x_q)]
        elif step == 2:
            if r > 0.77:
                if c < countModel:
                    model1 = sm.OLS(y_mass, x_q)
                    res = model1.fit()
                    d = durbin_watson(res.resid)
                    models2 = models2 + ["y = b0 + b1x1^2 + b2x2^2 + ... + e"]
                    models2 = models2 + [str(listComb[i])]
                    models2 = models2 + ["determination"]
                    models2 = models2 + [str(r)]
                    models2 = models2 + ["dw"]
                    models2 = models2 + [str(d)]
                    models2 = models2 + ["mult"]
                    mult = np.linalg.cond(res.model.exog)
                    models2 = models2 + [str(mult)]
                    test_br_pg = sms.het_breuschpagan(res.resid, res.model.exog)
                    models2 = models2 + ["heteroscedasticity"]
                    models2 = models2 + [str(test_br_pg)]
                    skm_mass_2 = skm_mass_2 + [res]
                    c = c + 1
                else:
                    return models2
    if step == 1:
        if y_score_model:
            return y_score_model
        else:
            return []
    elif step == 2:
        if models:
            return models2
        else:
            return []


def all_models():
    global y_last
    global models
    column_mass = np.array([i for i in range(N)])
    for i in range(1, N + 1):
        y_new_mass1 = combinations_liner_model(i, x, y, column_mass, 1)
        y_new_mass2 = combinations_ln_model(i, x, y, column_mass, 1)
        y_new_mass3 = combinations_quadratic_model(i, x, y, column_mass, 1)
        if y_new_mass1:
            models = models + ["y = b0 + b1x1 + b2x2 + ... + e"]
            models = models + [i]
            y_last = y_last + y_new_mass1
        if y_new_mass2:
            models = models + ["y = b0 + b1lnx1 + ... + e"]
            models = models + [i]
            y_last = y_last + y_new_mass2
        if y_new_mass3:
            models = models + ["y = b0 + b1x1^2 + b2x2^2 + ... + e"]
            models = models + [i]
            y_last = y_last + y_new_mass3
    y_new_mass = second_model(x, y, 1)
    if y_new_mass:
        y_last = y_last + y_new_mass


all_models()

y_score = np.empty([0, len(y_last[0])])

for i in range(len(y_last)):
    y_score = np.vstack((y_score, np.array(y_last[i])))

y_score = y_score.transpose()

for j in range(len(y_score)):
    if min(y_score[j]) <= 0:
        for k in range(len(y_score[j])):
            if y_score[j][k] <= 0.0:
                y_score[j][k] = 0.00000001


countModel = 150 / (len(models) / 2)


def step2():
    global models
    global models_step2
    column_mass = np.array([i for i in range(len(y_score[0]))])
    column_mass1 = np.array([i for i in range(10)])
    column_mass2 = np.array([i for i in range(11, 21)])
    for i in range(0, len(models), 2):
        if models[i] == "y = b0 + b1x1 + b2x2 + ... + e":
            if (models[i + 1] == 4) | (models[i + 1] == 5) | (models[i + 1] == 6):
                y_new_mass1 = combinations_liner_model(models[i + 1], y_score, y, column_mass, 2)
                if y_new_mass1:
                    models_step2 = models_step2 + y_new_mass1
            else:
                y_new_mass4 = combinations_liner_model(models[i + 1], y_score, y, column_mass1, 2)
                models_step2 = models_step2 + y_new_mass4
                y_new_mass5 = combinations_liner_model(models[i + 1], y_score, y, column_mass2, 2)
                models_step2 = models_step2 + y_new_mass5
        if models[i] == "y = b0 + b1x1^2 + b2x2^2 + ... + e":
            if (models[i + 1] == 4) | (models[i + 1] == 5) | (models[i + 1] == 6):
                y_new_mass2 = combinations_quadratic_model(models[i + 1], y_score, y, column_mass, 2)
                if y_new_mass2:
                    models_step2 = models_step2 + y_new_mass2
            else:
                y_new_mass = combinations_quadratic_model(models[i + 1], y_score, y, column_mass1, 2)
                models_step2 = models_step2 + y_new_mass
                y_new_mass6 = combinations_quadratic_model(models[i + 1], y_score, y, column_mass2, 2)
                models_step2 = models_step2 + y_new_mass6


step2()

dw = []
for i in range(0, len(models_step2), 10):
    dw = dw + [models_step2[i + 5]]
max_dw = max(dw)

t = 0
for i in range(0, len(models_step2), 10):
    if models_step2[i + 5] == max_dw:
        result_model = np.array([models_step2[j] for j in range(i, i + 10)])
        skm_last = skm_mass_2[t]
        print(result_model)
    t = t + 1

print("Риск развития сердечно-сосудистого заболевания")
print("Введите уровень глюкозы: ")
glucose = input()
print("Введите уровень холестерина: ")
cholesterol = input()
print("Введите уровень систолического давления: ")
sistolic = input()
print("Введите уровень диастолического давления: ")
diastolic = input()
print("BMI: ")
bmi = input()
print("Статус курения (введите: Yes/No/Formerly): ")
smoking = input()
print("Пол (введите: M/F): ")
gender = input()
print("Возраст: ")
age = input()
print("Имеет ли пациент сердечно-сосудистое заболевание? (введите: Yes/No): ")
hypertension = input()
smoking_num = 0
gender_num = 0
hypertension_num = 0
if smoking == "Yes":
    smoking_num = 2
elif smoking == "No":
    smoking_num = 1
elif smoking == "Formerly":
    smoking_num = 3
if gender == "M":
    gender_num = 1
elif gender == "F":
    gender_num = 2
if hypertension == "Yes":
    hypertension_num = 1
elif hypertension == "No":
    hypertension_num = 2
test_data = np.array([float(glucose), float(cholesterol), float(sistolic), float(diastolic), float(bmi), float(smoking_num), float(gender_num), float(age), float(hypertension_num)])

def clear_list(mass):
    xx = [re.sub('\D+', '', i) for i in mass]
    new_list_m = [val for val in xx if val]
    return new_list_m


list_column = list(result_model[1])
new_list = clear_list(list_column)
y_score_test_1 = []

for i in range(len(new_list)):
    for j in range(len(skm_mass)):
        if j == int(new_list[i]):
            new_mass_c = clear_list(number_models[j])
            last_m = np.empty((0, len(new_mass_c)), float)
            for k in range(len(new_mass_c)):
                last_m = np.append(last_m, test_data[int(new_mass_c[k])])
            if all_mod[j] == "y = b0 + b1x1 + b2x2 + ... + e":
                y_score_test_1 = y_score_test_1 + [skm_mass[j].predict([last_m])]
            elif all_mod[j] == "y = b0 + b1x1^2 + b2x2^2 + ... + e":
                x_l = np.power(last_m, 2)
                y_score_test_1 = y_score_test_1 + [skm_mass[j].predict([x_l])]
            elif all_mod[j] == "y = b0 + b1lnx1 + ... + e":
                x_ln = np.array([math.log(last_m[l]) for l in range(len(last_m))])
                y_score_test_1 = y_score_test_1 + [skm_mass[j].predict([x_ln])]
            elif all_mod[j] == "y = b0 + b1xi + b2xi^2 + e":
                x_m = np.power(last_m, 2)
                x_m_1 = np.append(last_m, x_m)
                y_score_test_1 = y_score_test_1 + [skm_mass[j].predict([x_m_1])]
            break

y_score_2 = np.empty([0, len(y_score_test_1[0])])

for i in range(len(y_score_test_1)):
    y_score_2 = np.vstack((y_score_2, np.array(y_score_test_1[i])))

y_score_2 = y_score_2.transpose()

y_score_2 = np.power(y_score_2, 2)

res = skm_last.predict(y_score_2)
print("риск развития сердечно-сосудистого заболевния = ", res)

res2 = skm_last.get_prediction(y_score_2)
interval = res2.summary_frame(alpha=0.05)
print("доверительный интервал = ", "[", np.float64(interval.obs_ci_lower), ";", np.float64(interval.obs_ci_upper), "]")
