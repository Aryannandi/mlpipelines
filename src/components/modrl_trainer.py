import os,sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustmeException
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object
from src.utils import evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV



@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts/model_trainer","model.pk")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def inititate_model_trainer(self, train_array, test_array):
        try:
            logging.info("splitting our data into dependent and indepent features")
            X_train, y_train,X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )


            model = {
                "random forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "logistic": LogisticRegression()
                
            }


            params = {
                "random forest":{
                    "class_weight":["balanced"],
                    'n_estimators':[20,50,30],
                    'max_depth':[10,8,5],
                    'min_samples_split':[2,5,10],

                },
                
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "splitter": ["best", "random"],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": [None, "auto", "sqrt", "log2"],
                    "class_weight": [None, "balanced"]
                },

                "logistic": {
                    "penalty": ["l1", "l2", "elasticnet", "none"],
                    "C": [0.01, 0.1, 1, 10, 100],
                    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                    "max_iter": [100, 200, 300],
                    "class_weight": [None, "balanced"]
                }
            }


            model_report:dict = evaluate_model(X_train= X_train, y_train= y_train, X_test=X_test, y_test=y_test,
                                               models=model,params=params)
            

            #to get best model from our report Dock
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model.keys())[
                list(model_report.values()).index(best_model_score)

            ]

            best_model = model[best_model_name]

            print(f"Best Model Found, Model Name is: {best_model_name},Accuracy_Score: {best_model_score}")
            print("\n***************************************************************************************\n")
            logging.info(f"best model found, Model Name is {best_model_name}, accuracy Score: {best_model_score}")
            

            save_object(file_path=self.model_trainer_config.train_model_file_path,
                        obj=best_model
                        )
            
        
            # logging.info("done the program")
        except Exception as e:
            raise CustmeException(e, sys)