from pandas import Series
from matplotlib import pyplot as plt
from numpy import polyfit
import pandas as pd
import numpy as np

def get_user_data(data, userId):
    try:
        return data.loc[data.index.get_level_values(0) == userId].copy()
    except KeyError:
        print('El usuario ', userId, ' no existe.')

df = pd.read_pickle('./pkl/dataset.pkl')

for i in df.index.get_level_values(0).drop_duplicates():
    userdata = get_user_data(df, i).iloc[0:500]
    print(i)
    series = userdata.slevel
    # fit polynomial: x^2*b1 + x*b2 + ... + bn
    X = userdata.index.get_level_values(1).hour
    y = series.values

    degree = 5
    coef = polyfit(X, y, degree)
    print('Coefficients: %s' % coef)
    # create curve
    curve = list()
    for i in range(len(X)):
        value = coef[-1]
        for d in range(degree):
            value += X[i]**(degree-d) * coef[d]
        curve.append(value)
    # plot curve over original data
    plt.close()
    plt.plot(series.values,color='blue',linewidth=1)
    plt.plot(curve, color='red', linewidth=1)
    plt.show()
'''
for i in df.index.get_level_values(0).drop_duplicates():
    userdata = get_user_data(df, i)
    print(i, ' ',userdata.shape)
'''



#datos random
a = np.asarray((list(range(20))+list(reversed(range(20))))*20)
a = a + 20 * (np.random.rand(a.shape[0]) * 0.2 - 0.1)

import matplotlib.pyplot as plt
plt.plot(a)
plt.show()

#Genero listas de obsevaciones con corrimientos hasta en 70 unidades de tiempo
b = [a[i:-(71-i)] for i in range(70)]
#Calculo la autocorrelación y me quedo con la autocorrelación de mis datos originales contra todos los corrimientos
c = np.corrcoef(b)[0, :]
#"nuetralizo" la autocorrelación consigo mismo ya que es 1 siempre
c[0] = 0

#Veo el maximo, supongo que me indica seasonality
print(np.argmax(c))
print(c[np.argmax(c)])

print(np.argmin(c))
print(c[np.argmin(c)])


from pandas.plotting import autocorrelation_plot

dfu = get_user_data(df, 32).droplevel(0).loc[:,'slevel']
idx = pd.date_range('2013-03-27 04:00:00', '2013-06-01 3:00:00', freq='h')
d = pd.DataFrame(index=idx)
d['slevel'] = dfu
d.isna().sum()
d.ffill(inplace=True)


# Autocorrelation Plot
plt.close()
plt.rcParams.update({'figure.figsize':(9,5), 'figure.dpi':120})
autocorrelation_plot(dfu.slevel)
plt.show()