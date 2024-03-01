#Arun Patwa
#B20184
#6266752643


import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error

print("-------------------------------Question 1(a)-----------------------------\n")
#importing the required csv file
df = pd.read_csv('D://College//3rd Sem//IC272(DS3)//Assignments//B20184_Assn6//daily_covid_cases.csv')
n_rows = df.shape[0] 
print(n_rows)
#  only the dates and the new covid cases data included in this not the column no.

plt.plot(range(1,n_rows+1),df['new_cases'])         #line plot between index of day and power consumed
# plt.scatter(range(1,n_rows+1),df['new cases'])


plt.xlabel('Index of the day')
plt.ylabel('the number of Covid-19 case')
plt.title('LinePlot-1a')
plt.savefig("1a.png")
plt.show()

print("-------------------------------Question 1(b)-----------------------------\n")
newTime1 = df['new_cases'][1:612,]                 #X(t-1) Lag time data

newTime2 = df['new_cases'][0:611,]                 #x(t) given data

pear,_ = pearsonr(newTime1,newTime2)        #correlation between X(t-1) and X(t)
print("The value of Pearson's correlation coefficient (with one day lag) : ",pear)

print("-------------------------------Question 1(c)-----------------------------\n")
plt.scatter(newTime1,newTime2)              #scatter plot between X(t-1) and X(t)
plt.xlabel("Lag time")
plt.ylabel("Given time")
plt.title("Given time sequence and one day lagged")
plt.savefig('1c')
plt.show()



print("-------------------------------Question 1(d)-----------------------------\n")
correlationList = []        #list of Correlation between the given time series and the Lag value time series
for i in range(1,7):
    givenTime = df['new_cases'][0:612-i,]          #given time series
    lagTime = df['new_cases'][i:612,]              #lag value time series
    corrCoefficient,_ = pearsonr(lagTime,givenTime)   #finding Correlation coefficient
    print('\nCorrelation Coefficient for the Lag value is ',i,' days is : ',corrCoefficient)
    correlationList.append(corrCoefficient)

plt.plot(range(1,7),correlationList)
plt.xlabel("Lagged values (in days)")
plt.ylabel("Correlation coefficient")
plt.title("Correlation coefficient with different lag values")
plt.savefig('1d')
plt.show()  

print("-------------------------------Question 1(e)-----------------------------\n")
plot_acf(df['new_cases'], lags=25)            #plotting the Correlation vs lag value 
plt.xlabel("Lagged values (in days)")
plt.ylabel("Correlation coefficient")
plt.title("Autocorrelation")
plt.savefig('1e')
plt.show()
print("the graph is plotted for 1e")
