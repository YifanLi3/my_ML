import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import  fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, Ridge
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error

def demo01_linearRegression():
    boston =  fetch_openml(name="boston",version=1,parser="auto")
    # print(boston)
    data = boston.data
    target = boston.target.values

    x_train,x_test,y_train,y_test = train_test_split(data,target,
                                                    test_size=0.2,random_state=10)


    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = LinearRegression()
    estimator.fit(x_train,y_train)

    print(estimator.coef_)
    print(estimator.intercept_)

    y_pred = estimator.predict(x_test)
    print(y_pred)

    print(mean_squared_error(y_test,y_pred))
    print(root_mean_squared_error(y_test,y_pred))
    print(mean_absolute_error(y_test,y_pred))

def demo02_linearRegression():
    boston =  fetch_openml(name="boston",version=1,parser="auto")
    # print(boston)
    data = boston.data
    target = boston.target.values

    x_train,x_test,y_train,y_test = train_test_split(data,target,
                                                    test_size=0.2,random_state=10)


    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = SGDRegressor(fit_intercept=True,learning_rate='constant',eta0=0.01)
    estimator.fit(x_train,y_train)
    print(estimator.coef_)
    print(estimator.intercept_)

    y_pred = estimator.predict(x_test)
    print(y_pred)

    print(mean_squared_error(y_test,y_pred))
    print(root_mean_squared_error(y_test,y_pred))
    print(mean_absolute_error(y_test,y_pred))

def dm01_under_fitting():
    np.random.seed(23)
    x = np.random.uniform(-3,3,100)
    y = 0.5 * x  ** 2 + x + 2 + np.random.normal(0,1,100)

    print(x[:5])
    print(y[:5])

    X = x.reshape(-1,1)
    print(X[:5])

    estimator = LinearRegression()
    estimator.fit(X,y)
    y_pred = estimator.predict(X)

    print(f'mse:{mean_squared_error(y,y_pred)}')
    plt.scatter(x,y)
    plt.plot(x,y_pred, color='red')
    plt.show()

def dm02_just_fitting():
    np.random.seed(23)
    x = np.random.uniform(-3, 3, 100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)

    print(x[:5])
    print(y[:5])

    X = x.reshape(-1, 1)
    print(X[:5])

    X2 = np.hstack([X,X ** 2])
    print(f'X2{X2[:5]}')

    estimator = LinearRegression()
    estimator.fit(X2, y)
    y_pred = estimator.predict(X2)
    print(f'mse:{mean_squared_error(y, y_pred)}')
    plt.scatter(x, y)
    plt.plot(np.sort(x), y_pred[np.argsort(x)], color='red')
    plt.show()

def dm03_over_fitting():
    np.random.seed(23)
    x = np.random.uniform(-3, 3, 100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)

    print(x[:5])
    print(y[:5])

    X = x.reshape(-1, 1)
    print(X[:5])

    X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])
    print(f'X3{X3[:5]}')

    estimator = LinearRegression()
    estimator.fit(X3, y)
    y_pred = estimator.predict(X3)
    print(f'mse:{mean_squared_error(y, y_pred)}')
    plt.scatter(x, y)
    plt.plot(np.sort(x), y_pred[np.argsort(x)], color='red')
    plt.show()

def dm04_L1():
    np.random.seed(23)
    x = np.random.uniform(-3, 3, 100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)

    print(x[:5])
    print(y[:5])

    X = x.reshape(-1, 1)
    print(X[:5])

    X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])
    print(f'X3{X3[:5]}')

    estimator = Lasso(alpha=0.1)
    estimator.fit(X3, y)
    y_pred = estimator.predict(X3)
    print(f'mse:{mean_squared_error(y, y_pred)}')
    plt.scatter(x, y)
    plt.plot(np.sort(x), y_pred[np.argsort(x)], color='red')
    plt.show()

def dm05_L2():
    np.random.seed(23)
    x = np.random.uniform(-3, 3, 100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)

    print(x[:5])
    print(y[:5])

    X = x.reshape(-1, 1)
    print(X[:5])

    X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])
    print(f'X3{X3[:5]}')

    estimator = Ridge(alpha=10)
    estimator.fit(X3, y)
    y_pred = estimator.predict(X3)
    print(f'mse:{mean_squared_error(y, y_pred)}')
    plt.scatter(x, y)
    plt.plot(np.sort(x), y_pred[np.argsort(x)], color='red')
    plt.show()

if __name__ == '__main__':
    #demo01_linearRegression()
    #demo02_linearRegression()
    #dm01_under_fitting()
    #dm02_just_fitting()
    #dm04_L1()
    dm05_L2()