from sklearn.neighbors import KNeighborsRegressor
import pandas as pd


class kNearestNeighbor():
    def __init__(self,datadir='') -> None:
        self.datadir=datadir
        k = 2
        self.model = KNeighborsRegressor(n_neighbors=k)
        self.usecolumn = ["lat","lon","year"]
       
    def fit(self,x_train,y_train):
        x_train = x_train[self.usecolumn]
        self.model.fit(x_train,y_train)
    
    def predict(self,x_test):
        x_test = x_test[self.usecolumn]
        y_predict = pd.DataFrame(self.model.predict(x_test),index=x_test.index,columns=['prediction'])
        return y_predict
