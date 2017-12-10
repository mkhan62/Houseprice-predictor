import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn import metrics

# predicting housing prices through Boston's housing prices dataset from 1970
dataset = load_boston()
df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target


X = df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = LinearRegression()
model.fit(X_train, y_train)
modelcf = pd.DataFrame(data=model.coef_, index=X.columns, columns=['Coefficients'])

predictions = model.predict(X_test)
plt.scatter(y_test, predictions)
plt.show()
sns.distplot((y_test-predictions), bins=50)
plt.show()

# cost function
abs_error = metrics.mean_absolute_error(y_test,predictions)
ms_error = metrics.mean_squared_error(y_test, predictions)
rms_error = np.sqrt(ms_error)
print(abs_error, ms_error, rms_error)
