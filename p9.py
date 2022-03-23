import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import statsmodels.api as sm

x = [i/5.0 for i in range(30)]
y = [1,2,1,2,1,1,3,4,5,4,5,6,5,6,7,8,9,10,11,11,12,11,11,10,12,11,11,10,9,13]

lowess = sm.nonparametric.lowess(y,x)
lowess_x = list(zip(*lowess))[0]
lowess_y = list(zip(*lowess))[1]
f = interp1d(lowess_x, lowess_y,bounds_error = False)

xnew = [i/10.0 for i in range(100)]
ynew = f(xnew)
plt.plot(x,y,'o')
plt.plot(lowess_x,lowess_y,'+')
plt.plot(xnew,ynew,'-')
plt.show()

