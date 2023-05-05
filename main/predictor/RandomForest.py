from sklearn.ensemble import RandomForestRegressor
import pandas as pd


class RandomForest():
    def __init__(self, state=314, datadir='') -> None:
        self.datadir=datadir
        self.model = RandomForestRegressor(n_estimators=100, criterion='squared_error', max_depth=None,random_state=state)
       
    def fit(self, x_train, y_train):
        self.model.fit(x_train,y_train)
    
    def predict(self, x_test):
        y_predict = pd.DataFrame(self.model.predict(x_test),index=x_test.index,columns=['prediction'])
        return y_predict
