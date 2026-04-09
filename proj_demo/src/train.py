from curses import OK
from matplotlib.pyplot import grid
import pandas as pd
from pandas.core.indexing import NDFrameIndexerBase
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path

def train_and_select_best_model():
    mlflow.set_experiment('iris_model_selection')
    iris = load_iris()

    X = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    y = iris.target

    x_train, x_test, y_train, y_test = train_test_split(
                        X, 
                        y, 
                        test_size=0.2, 
                        random_state=42
                        )

    model_configs = {
        'KNN': {
            'model': KNeighborsClassifier(),
            'params':{
                "n_neighbors": [3, 5, 7, 9],
                "weights": ['uniform', 'distance']
            }
        },
        'Logistic_Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            #?: Logistic Regression（逻辑回归）：
            #?: C 是正则化强度的倒数：C 越大惩罚越小，模型越复杂
            'params':{
                'C': [0.1, 1.0, 10.0]
            }
        },
        'Random_Forest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators':[50,100,200],
                'max_depth': [None, 10, 20],
                'min_samples_split':[2,5],
            }
        }
    }

    best_model_name = None
    best_accuracy = 0.0
    best_model_object = None

    for name, config in model_configs.items():
        with mlflow.start_run(run_name=f"tune_{name}"):
            print(f"Tuning {name}...")

            grid_search = GridSearchCV(
                estimator=config['model'],
                param_grid=config['params'],
                cv = 5,
                scoring="accuracy",
                n_jobs=-1,
            )

            grid_search.fit(x_train, y_train)
            best_estimator = grid_search.best_estimator_
            best_params = grid_search.best_params_

            y_pred = best_estimator(x_test)

            acc = accuracy_score(y_test, y_pred)
            print(f"Best parameters: {best_params}")
            print(f"Best Accuracy: {acc}")

            mlflow.log_params(best_params)
            mlflow.log_metric('accuracy', acc)
            mlflow.sklearn.log_model(best_estimator, name)

            if acc > best_accuracy:
                best_accuracy = acc
                best_model_name = name
                best_model_object = best_estimator

    print(f"Best Model:{best_model_name} with accuracy: {best_accuracy:.4f}")

    model_dir = Path(__file__).parents[1] / "models"
    model_dir.mkdir(exist_ok=True)

    joblib.dump(best_model_object, model_dir / "best_model.pkl")

    metadata = {
        'model_name': best_model_name,
        'accuracy': best_accuracy,
        'target_names': iris.target_names.tolist()
    }

    joblib.dump(metadata, model_dir / "model_metadata.pkl")

    print(f"Best tuned model saved to {model_dir}")

if __name__ == "__main__":
    train_and_select_best_model()