import pandas as pd
from statsmodels.tsa.ar_model import AutoReg as AR
import math
import matplotlib.pyplot as plt


#q3
data = pd.read_csv('D://College//3rd Sem//IC272(DS3)//Assignments//B20184_Assn6//daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')

#a
test_size = 0.35  
X = data.values
tst_sze = math.ceil(len(X)*test_size)
train,test = X[:len(X)-tst_sze],X[len(X)-tst_sze:]

lagged_values = [1,5,10,15,25]
RMSE_err_per = []
MAPE_err =[]
#AR model
for p in lagged_values:
    model = AR(train,lags=p)
    model_fit = model.fit()   #fit train model
    coef = model_fit.params   

    history = train[len(train)-p:]
    history = [history[i] for i in range(len(history))]
    predictions =[]

    for i in range(len(test)):
        length = len(history)
        lag = [history[j] for j in range(length-p,length)]
        y_hat= coef[0]

        for d in range(p):
            y_hat += coef[d+1]*lag[p-d-1]    
        predictions.append(y_hat)
        obs = test[i]
        history.append(obs)

    
    n = len(test)
    e = 0
    for k in range(len(test)):
        e += (predictions[k] - test[k]) ** 2  
    RMSE = math.sqrt(e / n)
    RMSE_per = (RMSE / test.mean()) * 100
    RMSE_err_per.append(RMSE_per)

    # MAPE
    s = 0
    for l in range(len(test)):
        s += float(abs(test[l] - predictions[l]) / test[l])
    MAPE = (s / n) * 100  
    MAPE_err.append(MAPE)

values = pd.DataFrame()
values['lag values'],values['RMSE'],values['MAPE'] = lagged_values,RMSE_err_per,MAPE_err
print(values)
plt.bar(lagged_values,RMSE_err_per)
plt.xlabel('lagged values')
plt.ylabel('RMSE %')
plt.show()

plt.bar(lagged_values,MAPE_err)
plt.xlabel('lagged values')
plt.ylabel('MAPE')
plt.show()



