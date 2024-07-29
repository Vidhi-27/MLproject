import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split train and test input data")
            X_train,y_train,X_test,y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "DecisionTreeRegressor" :DecisionTreeRegressor(),
                "RandomForestRegressor" :RandomForestRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "LinearRegression" : LinearRegression(),
                "K-NeighborsRegressor" :KNeighborsRegressor(),
                "XGBRegressor" :XGBRegressor(),
                "CatBoostRegressor" :CatBoostRegressor(verbose=False),
                "AdaBoostRegressor" :AdaBoostRegressor(),
            }
            param = {
                "DecisionTreeRegressor" : {
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    # 'splitter':['random','best'],
                    # "max_features":['sqrt','log2']
                },
                "RandomForestRegressor" : {
                    # 'criterion':['square_error','friedman_mse','absolute_error','poisson'],
                    'n_estimators': [8,16,32,64,128,256],
                    # "max_features":['sqrt','log2',None]
                },
                "Gradient Boosting" : {
                    # 'criterion':['squared_error','friedman_mse'],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # "max_features":['auto','sqrt','log2'],
                    "learning_rate":[.1,.01,.05,.001],
                    # "loss" : ['square_error','absolute_error','huber','quantile'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "LinearRegression" : {},
                "K-NeighborsRegressor":{
                    "n_neighbors":[5,7,9,11],
                    # "weights" :['uniform','distaance'],
                    # "algorithm":['ball_tree',"kd_tree",'brute']
                },
                "XGBRegressor":{
                    "learning_rate":[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoostRegressor" :{
                    'depth':[6,8,10],
                    "learning_rate":[.1,.01,.05],
                    'iterations':[30,50,100]
                },
                "AdaBoostRegressor" :{
                    "learning_rate":[.1,.01,.05,.001],
                    # "loss" : ['square','linear','square'],
                    'n_estimators': [8,16,32,64,128,256]
                    
                }
            }

            model_report : dict = evaluate_model(X_train = X_train,y_train=y_train,X_test=X_test,y_test=y_test,models = models,param = param)
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report)[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise Exception("No  best model found")
            
            logging.info("Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )
            predicted = best_model.predict(X_test)
            pred_score = r2_score(predicted,y_test)

            return pred_score                      
            
        except Exception as e:
            raise CustomException(e,sys)
        