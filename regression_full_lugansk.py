from pydataset import data
import matplotlib as matplotlib
import pandas as pd
import numpy as np
from sklearn.linear_model import  LinearRegression
from sklearn.model_selection import  train_test_split
from matplotlib import pyplot as plt
import seaborn as sns


excel_data_luh_2014 = pd.read_excel('luh_2014_excel.xlsx')
excel_data_luh_2016 = pd.read_excel('luh_2016_excel.xlsx')
excel_data_luh_2018 = pd.read_excel('luh_2018_excel.xlsx')

excel_to_csv_luh_2014 = excel_data_luh_2014.to_csv('luh_2014.csv',index=False)
excel_to_csv_luh_2016 = excel_data_luh_2016.to_csv('luh_2016.csv',index=False)
excel_to_csv_luh_2018 = excel_data_luh_2018.to_csv('luh_2018.csv',index=False)

df_luh_2014 = pd.read_csv('luh_2014.csv')
df_luh_2016 = pd.read_csv('luh_2016.csv')
df_luh_2018 = pd.read_csv('luh_2018.csv')

fights_2014 = np.array(df_luh_2014['fights']).reshape((-1,1))
fights_2016 = np.array(df_luh_2016['fights']).reshape((-1,1))
fights_2018 = np.array(df_luh_2018['fights']).reshape((-1,1))

fires_2014 = np.array(df_luh_2014['type']).reshape((-1,1))
fires_2016 = np.array(df_luh_2016['type']).reshape((-1,1))
fires_2018 = np.array(df_luh_2018['type']).reshape((-1,1))

X_2014 = fires_2014
X_2016 = fires_2016
X_2018 = fires_2018

y_2014 = fights_2014
y_2016 = fights_2016
y_2018 = fights_2018

plt.subplot(2,2,1)
plt.scatter(X_2014,y_2014)
plt.xlabel('fires')
plt.ylabel ('fights')
plt.title('Lugansk region 2014', color = 'black', fontname = 'Times New Roman', size = 16, fontweight = 'bold')

plt.subplot(2,2,2)
plt.scatter(X_2016,y_2016)
plt.xlabel('fires')
plt.ylabel ('fights')
plt.title('Lugansk region 2016', color = 'black', fontname = 'Times New Roman', size = 16, fontweight = 'bold')

plt.subplot(2,1,2)
plt.scatter(X_2018,y_2018)
plt.xlabel('fires')
plt.ylabel ('fights')

plt.title('Lugansk region 2018', color = 'black', fontname = 'Times New Roman', size = 16, fontweight = 'bold')


plt.subplots_adjust(wspace= 0.3, hspace=0.3)
plt.show()

from sklearn.model_selection import train_test_split

X_train_2014,X_test_2014,y_train_2014,y_test_2014 = train_test_split(X_2014,y_2014, test_size= 0.4, random_state=23)
X_train_2016,X_test_2016,y_train_2016,y_test_2016 = train_test_split(X_2016,y_2016, test_size= 0.4, random_state=23)
X_train_2018,X_test_2018,y_train_2018,y_test_2018 = train_test_split(X_2018,y_2018, test_size= 0.4, random_state=23)

plt.subplot(2,2,1)
plt.scatter(X_train_2014,y_train_2014, color = 'g', label = 'Train')
plt.scatter(X_test_2014,y_test_2014, color = 'r', label = 'Test')
plt.xlabel('fires')
plt.ylabel ('fights')
plt.legend()
plt.title('Lugansk region 2014', color = 'black', fontname = 'Times New Roman', size = 16, fontweight = 'bold')

plt.subplot(2,2,2)
plt.scatter(X_train_2016,y_train_2016, color = 'g', label = 'Train')
plt.scatter(X_test_2016,y_test_2016, color = 'r', label = 'Test')
plt.xlabel('fires')
plt.ylabel ('fights')
plt.legend()
plt.title('Lugansk region 2016', color = 'black', fontname = 'Times New Roman', size = 16, fontweight = 'bold')

plt.subplot(2,1,2)
plt.scatter(X_train_2018,y_train_2018, color = 'g', label = 'Train')
plt.scatter(X_test_2018,y_test_2018, color = 'r', label = 'Test')
plt.xlabel('fires')
plt.ylabel ('fights')
plt.legend()
plt.title('Lugansk region 2018', color = 'black', fontname = 'Times New Roman', size = 16, fontweight = 'bold')

plt.subplots_adjust(wspace= 0.3, hspace=0.3)
plt.show()

X_train_2014 = np.array(X_train_2014).reshape(-1,1)
X_train_2016 = np.array(X_train_2016).reshape(-1,1)
X_train_2018 = np.array(X_train_2018).reshape(-1,1)

X_test_2014 = np.array(X_test_2014).reshape(-1,1)
X_test_2016 = np.array(X_test_2016).reshape(-1,1)
X_test_2018 = np.array(X_test_2018).reshape(-1,1)

from sklearn.linear_model import LinearRegression

lr_2014 = LinearRegression()
lr_2016 = LinearRegression()
lr_2018 = LinearRegression()

lr_2014.fit(X_train_2014,y_train_2014)
lr_2016.fit(X_train_2016,y_train_2016)
lr_2018.fit(X_train_2018,y_train_2018)

c_2014 = lr_2014.intercept_
c_2016 = lr_2016.intercept_
c_2018 = lr_2018.intercept_

m_2014= lr_2014.coef_
m_2016= lr_2016.coef_
m_2018= lr_2018.coef_

# print(c_2014)
# print(c_2016)
# print(c_2018)

# print(m_2014)
# print(m_2016)
# print(m_2018)

Y_pred_train_2014 = lr_2014.predict(X_train_2014)
Y_pred_train_2016 = lr_2016.predict(X_train_2016)
Y_pred_train_2018 = lr_2018.predict(X_train_2018)

Y_pred_test_2014 = lr_2014.predict(X_test_2014)
Y_pred_test_2016 = lr_2016.predict(X_test_2016)
Y_pred_test_2018 = lr_2018.predict(X_test_2018)

plt.subplot(3,2,1)   # train lug 2014
plt.scatter(X_train_2014,y_train_2014)
plt.plot(X_train_2014,Y_pred_train_2014, color = 'r', label = 'Train')
plt.xlabel('fires')
plt.ylabel ('fights')
plt.legend()
plt.title('Lugansk region 2014', color = 'black', fontname = 'Times New Roman', size = 14, fontweight = 'bold')

plt.subplot(3,2,3)   # train lug 2016
plt.scatter(X_train_2016,y_train_2016)
plt.plot(X_train_2016,Y_pred_train_2016, color = 'r', label = 'Train')
plt.xlabel('fires')
plt.ylabel ('fights')
plt.legend()
plt.title('Lugansk region 2016', color = 'black', fontname = 'Times New Roman', size = 14, fontweight = 'bold')

plt.subplot(3,2,5)   # train lug 2018
plt.scatter(X_train_2018,y_train_2018)
plt.plot(X_train_2018,Y_pred_train_2018, color = 'r', label = 'Train')
plt.xlabel('fires')
plt.ylabel ('fights')
plt.legend()
plt.title('Lugansk region 2018', color = 'black', fontname = 'Times New Roman', size = 14, fontweight = 'bold')


plt.subplot(3,2,2) # test lug 2014
plt.scatter(X_test_2014,y_test_2014)
plt.plot(X_test_2014,Y_pred_test_2014, color = 'g', label = 'Test')
plt.xlabel('fires')
plt.ylabel ('fights')
plt.legend()
plt.title('Lugansk region 2014', color = 'black', fontname = 'Times New Roman', size = 14, fontweight = 'bold')

plt.subplot(3,2,4) # test lug 2016
plt.scatter(X_test_2016,y_test_2016)
plt.plot(X_test_2016,Y_pred_test_2016, color = 'g', label = 'Test')
plt.xlabel('fires')
plt.ylabel ('fights')
plt.legend()
plt.title('Lugansk region 2016', color = 'black', fontname = 'Times New Roman', size = 14, fontweight = 'bold')

plt.subplot(3,2,6) # test lug 2018
plt.scatter(X_test_2018,y_test_2018)
plt.plot(X_test_2018,Y_pred_test_2018, color = 'g', label = 'Test')
plt.xlabel('fires')
plt.ylabel ('fights')
plt.legend()
plt.title('Lugansk region 2018', color = 'black', fontname = 'Times New Roman', size = 14, fontweight = 'bold')

plt.subplots_adjust(wspace= 0.3, hspace=1)
plt.show()

# score for models

score_2014_train = lr_2014.score(X_train_2014,y_train_2014)
score_2016_train = lr_2016.score(X_test_2016,y_test_2016)
score_2018_train = lr_2018.score(X_test_2018,y_test_2018)

score_2014_test = lr_2014.score(X_test_2014,y_test_2014)
score_2016_test = lr_2016.score(X_test_2016,y_test_2016)
score_2018_test = lr_2018.score(X_test_2018,y_test_2018)

# print(score_2014_test)
# print(score_2016_test)
# print(score_2018_test)


print(score_2014_train)
print(score_2016_train)
print(score_2018_train)
