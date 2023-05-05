import pandas as pd
import os
import argparse
from sklearn.metrics import mean_squared_error as mse
import numpy as np
#本プログラムはモデルの汎化性能を評価することを目的としている
#ハイパラ調整として使うのはクロスバリデーションの目的から外れる
from predictor import ScoringService
from sklearn.model_selection import KFold
import joblib


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='original_train')
    parser.add_argument('--model-dir', default = 'model')
    parser.add_argument('--submit-dir', default = 'submits')
    args = parser.parse_args()
    return args

def main():
    args=parse_args()
    datadir=args.data_dir
    modeldir=args.model_dir
    submitdir=args.submit_dir
    traindata=pd.read_csv(os.path.join(datadir,'train_data.csv'),index_col=0)
    testdata=pd.read_csv(os.path.join(datadir,'test_data.csv'),index_col=0)

    model=ScoringService(datadir=datadir)#トレーニングデータの入っているディレクトリを指定
    
    x_train=traindata.drop('cover',axis=1)
    y_train=traindata['cover']
    x_test=testdata
        
    model.fit(x_train,y_train)
    joblib.dump(model,os.path.join(modeldir,'submit_model.joblib'))
        
    y_predict = model.predict(x_test)
    y_predict.to_csv(os.path.join(submitdir,'submit.csv'),header=False)


if __name__ == "__main__":
    main()
