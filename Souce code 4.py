#Arun Patwa
#B20184
#6266752643


import pandas as pd
from statsmodels.tsa.ar_model import AutoReg as AR
import math


data = pd.read_csv('D://College//3rd Sem//IC272(DS3)//Assignments//B20184_Assn6//daily_covid_cases.csv')

print("-------------------------------Question 4(a)-----------------------------\n")
test_size = 0.35  
tst_sze = math.ceil(len(data)*test_size)
train,test = data[:len(data)-tst_sze],data[len(data)-tst_sze:]
train_list = list(train['new_cases'])
test_list = list(test['new_cases'])
p = 1

#Computing auto correlation
while p>=1 :
    train[f'{p}-days lag'] =''
    for j in range(p,len(train)):
        train[f'{p}-days lag'][j] = train['new_cases'][j-p]

    s_1 = 0
    s_2 = 0
    s_1_2 = 0
    s_2_2 = 0
    for j in range(p, len(train)):
        s_1 += train['new_cases'][j]
        s_2 += train[f'{p}-days lag'][j]
        s_1_2 += (train['new_cases'][j]) ** 2
        s_2_2 += (train[f'{p}-days lag'][j]) ** 2
    e_1 = s_1 / (len(train) - p)
    e_2 = s_2 / (len(train) - p)
    e_1_2 = s_1_2 / (len(train) - p)
    e_2_2 = s_2_2 / (len(train) - p)
    std_1 = (e_1_2 - (e_1 ** 2)) ** 0.5
    std_2 = (e_2_2 - (e_2 ** 2)) ** 0.5
    s = 0
    for j in range(p, len(train)):
        s += (train['new_cases'][j] - e_1) * (train[f'{p}-days lag'][j] - e_2)
    corr = s / ((len(train) - p) * (std_1 * std_2))
    if corr <= 2/(len(train)**0.5):           
        optimal_lag = p-1
        print(f'Optimal lag is {optimal_lag}')
        break
    else:
        p = p+1

#AR model for optimal lag
model = AR(train_list,lags=optimal_lag)
model_fit = model.fit()   #fit train model
coef = model_fit.params   

history = train_list[len(train_list)-optimal_lag:]
history = [history[i] for i in range(len(history))]
predictions =[]

for i in range(len(test_list)):
    length = len(history)
    lag = [history[j] for j in range(length-optimal_lag,length)]
    y_hat= coef[0]

    for d in range(optimal_lag):
        y_hat += coef[d+1]*lag[optimal_lag-d-1]    
    predictions.append(y_hat)
    obs = test_list[i]
    history.append(obs)

print("-------------------------------RMSE-----------------------------\n")
n = len(test_list)
e = 0
for k in range(len(test_list)):
    e += (predictions[k] - test_list[k]) ** 2  
RMSE = math.sqrt(e / n)
RMSE_per = (RMSE / (sum(test_list)/len(test_list))) * 100

print("-------------------------------MAPE-----------------------------\n")
s = 0
for l in range(len(test_list)):
    s += float(abs(test_list[l] - predictions[l]) / test_list[l])
MAPE = (s / n) * 100  

print(f'RMSE % is {RMSE_per}')
print(f'MAPE between actual and predicted data is {MAPE}')









