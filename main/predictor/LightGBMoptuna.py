import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from functools import partial


class LightGBM():
    def __init__(self,depth=3,datadir='') -> None:
        self.datadir=datadir
        self.depth = depth
       
    def fit(self,x_train,y_train):
        
        x_lgbtrain,x_lgbeval,y_lgbtrain,y_lgbeval = train_test_split(x_train, y_train, test_size=0.3, shuffle=True, random_state=314)
        lgb_train = lgb.Dataset(x_lgbtrain, y_lgbtrain, free_raw_data=False)
        lgb_eval = lgb.Dataset(x_lgbeval, y_lgbeval, reference=lgb_train, free_raw_data=False)
        model = lgb.LGBMRegressor()
        
        def rmse_score(y_true, y_pred):
            mse = mean_squared_error(y_true, y_pred)
            rmse = math.sqrt(mse)
            return rmse
                
        def bayes_objective(trial,dep):
            params = {
                'max_depth': dep,
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 50,),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0001, 0.1,),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0001, 0.1,),
                'num_leaves': trial.suggest_int('num_leaves', 2, 6),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                'subsample': trial.suggest_float('subsample', 0.4, 1.0),
                'subsample_freq': trial.suggest_int('subsample_freq', 0, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 0, 10)
            }
            # モデルにパラメータ適用
            model.set_params(**params)
            # cross_val_scoreでクロスバリデーション
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            #誤差関数適応
            score_funcs = {
                'rmse': make_scorer(rmse_score),
            }
            scores = cross_validate(model,
                                    pd.concat([x_lgbtrain,x_lgbeval],axis=0).values,
                                    pd.concat([y_lgbtrain,y_lgbeval],axis=0).values,
                                    cv=kf,
                                    scoring=score_funcs)
            
            return scores['test_rmse'].mean()
            
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(partial(bayes_objective,dep=self.depth), n_trials=400)
        
        self.model = lgb.train(study.best_trial.params, # 上記で設定したパラメータ
                          lgb_train,                                # 使用するデータセット
                          num_boost_round=1000,                     # 学習の回数
                          valid_names=['train', 'valid'],           # 学習経過で表示する名称
                          valid_sets=[lgb_train, lgb_eval],         # モデル検証のデータセット
                          verbose_eval=-1)                          # 学習の経過の表示(10回毎)
    
    def predict(self,x_test):
        y_predict = pd.Series(self.model.predict(x_test), index=x_test.index)
        return pd.DataFrame(y_predict,index=x_test.index,columns=['prediction'])
