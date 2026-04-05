from sklearn.datasets import load_iris
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV   #分割测试集和训练集
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score  #模型评估。计算模型的准确率

def demo01_load_iris():
    iris_data = load_iris()
    print(f'type(iris_data):{type(iris_data)}')
    print(f'keys:{iris_data.keys()}')
    print(dir(iris_data))
    print(iris_data.target)
    print(iris_data.feature_names)

def demo02_show_iris():
    iris_data = load_iris()

    iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    iris_df['label'] = iris_data.target
    print(iris_df.head())

    sns.lmplot(data=iris_df,
                x='sepal length (cm)',
                y='sepal width (cm)',
                hue='label',
                fit_reg=True
                )


    plt.title('Iris Dataset')
    plt.tight_layout()
    plt.show()

def demo03_split_train_test():
    iris_data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
                        iris_data.data, 
                        iris_data.target, 
                        test_size=0.2, 
                        random_state=2
                        )

    print(len(x_train))
    print(len(x_test))
    print(len(y_train))
    print(len(y_test))


def demo04_iris_test():
    iris_data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
                        iris_data.data, 
                        iris_data.target, 
                        test_size=0.2, 
                        random_state=2
                        )
    
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train, y_train)
    y_test_pred = estimator.predict(x_test)
    print(y_test_pred)

    print(estimator.score(x_test, y_test))
    print(accuracy_score(y_test, y_test_pred))


def demo05_gscv():
    iris_data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
                        iris_data.data, 
                        iris_data.target, 
                        test_size=0.2, 
                        random_state=2
                        )
    
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = KNeighborsClassifier()
    param_dict = {'n_neighbors': [i for i in range(1, 11)]}

    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=4)

    estimator.fit(x_train, y_train)

    print(estimator.best_params_)
    print(estimator.best_score_)
    print(estimator.best_estimator_)
    #print(estimator.cv_results_)

    best_estimator = estimator.best_estimator_
    y_pred = best_estimator.predict(x_test)
    print(accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    #demo01_load_iris()
    #demo02_show_iris()
    #demo03_split_train_test()
    #demo04_iris_test()
    demo05_gscv()