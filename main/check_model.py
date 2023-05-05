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
    parser.add_argument('--result-dir', default = 'result')
    parser.add_argument('--nrows', default = None)
    args = parser.parse_args()
    return args

def main():
    args=parse_args()
    datadir=args.data_dir
    modeldir=args.model_dir
    resultdir=args.result_dir
    nrows=args.nrows
    if nrows != None:
        nrows=int(nrows)
    #print('loading train data')
    traindata=pd.read_csv(os.path.join(datadir,'train_data.csv'),index_col=0,nrows=nrows)
    #traindata = traindata[traindata.year!=1999].reset_index().drop("index",axis=1)

    
    model=ScoringService(datadir=datadir)#トレーニングデータの入っているディレクトリを指定
    score=list()
    i=0
    for train_index,test_index in  list(KFold(random_state=0,shuffle=True).split(traindata)):
        trainx=traindata.loc[train_index,:].drop('cover',axis=1)
        trainy=traindata.loc[train_index,'cover']
        testx=traindata.loc[test_index,:].drop('cover',axis=1)
        testy=traindata.loc[test_index,'cover']
        
        model.fit(trainx,trainy)
        joblib.dump(model,os.path.join(modeldir,f'model_{i}.joblib'))
        
        result=pd.concat([model.predict(testx),testy],axis=1)
        #print(result)
        score.append(np.sqrt(mse(result['prediction'],result['cover'])))
        
        result.to_csv(os.path.join(resultdir,f'error_data_{i}.csv'))
        i+=1
    print('valid score ',score)


if __name__ == "__main__":
    main()


