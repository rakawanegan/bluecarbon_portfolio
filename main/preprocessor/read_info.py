import pandas as pd
import numpy as np
from functools import partial
import os
class Categorize_Training_Data:
	def __init__(self,) -> None:
		self.categorydf=pd.read_excel(os.path.join("../official_data",'feature_description.xlsx'))
		self.category_dict={i:name for i,name in enumerate(self.categorydf.iloc[:,1].unique())}
	
	def _set_catergory_num(self,category_num:list)->None:
		self._category_num=category_num
		self.select_calname=self.categorydf[self.categorydf.iloc[:,1].isin([self.category_dict[i] for i in category_num])].iloc[:,0].array
	
	def fit(self,category_num:list) -> None:
		self._set_catergory_num(category_num=category_num)
	
	def transform(self,traindf:pd.DataFrame) -> pd.DataFrame:
		return traindf.filter(items=self.select_calname,axis=1).copy()

	def fit_transform(self,category_num:list,traindf:pd.DataFrame) -> pd.DataFrame:
		self._set_catergory_num(category_num=category_num)
		return traindf.filter(items=self.select_calname,axis=1).copy()


class Search_Randsat_Data():
	def __init__(self,) -> None:
		self.unique_feartedf=pd.read_csv(os.path.join("preprocessor",'minimal_feature_description_for_Landsat_data.csv') ,index_col=0)

	def specify_data(self,year:tuple,column_num:slice or list) -> None:
		self.year=year
		self.column_num=column_num
		self.picup_column=self.unique_feartedf.iloc[self.column_num,0]
		
	def transform(self,traindf:pd.DataFrame) -> pd.DataFrame:
		result_=pd.DataFrame()
		if self.column_num=='all':
			result_=traindf.copy()
		else:
			for s in self.picup_column.to_list():
				result_=pd.concat([result_,traindf.filter(like= s,axis=1)],axis=1)
		
		result=pd.DataFrame()
		if self.year=='all':
			return result_
		for year_ in range(self.year[0],self.year[1]+1):
			result=pd.concat([result,result_.filter(like= str(year_),axis=1)],axis=1)
		del result_
		return result

class Fill_Nan_Randsat():
	def __init__(self,) -> None:
		self.randsat_col=pd.read_csv(os.path.join("preprocessor",'minimal_feature_description_for_Landsat_data.csv') ,index_col=0)
		pass
	
	def _fillna(self,valdf,x):
		x.copy()
		x.loc[x.isna()]=valdf.loc[x.isna()]
		return x

	def transform(self,data_df:pd.DataFrame,)-> pd.DataFrame:
		for i in range(len(self.randsat_col)):
			pic_col=data_df.filter(like= self.randsat_col.iloc[i,0],axis=1).columns
			pic_df=data_df.loc[:,pic_col].copy()
			#線形補間
			pic_df=pic_df.interpolate(method='linear',axis=1)
			#spline
			pic_df=pic_df.interpolate(method='ffill',axis=1)
			f=partial(self._fillna,pic_df.mean(axis=1))
			pic_df=pic_df.transform(f)
			data_df.loc[:,pic_col]=pic_df.fillna(pic_df.mean())
		return data_df.copy()


class Drop_Outler_Randsat():
	def __init__(self,) -> None:
		self.randsat_col=pd.read_csv(os.path.join("preprocessor",'minimal_feature_description_for_Landsat_data.csv') ,index_col=0)
	
	def _trainsforn_func(self,quantiledf,x):
		x=x.copy()
		x.loc[x>=quantiledf]=np.nan
		return x.copy()

	def get(self):
		self.quantiledict= pickle.load(open(os.path.join("preprocessor",'quantidict.pickle'),'rb'))
	
	def fit(self,data_df:pd.DataFrame,q=0.85) -> None:
		self.quantiledict=dict()
		data_df=data_df.copy()
		for i in range(len(self.randsat_col)):
			pic_col=data_df.filter(like= self.randsat_col.iloc[i,0],axis=1).columns
			pic_df=data_df.loc[:,pic_col].copy()
			self.quantiledict[self.randsat_col.iloc[i,0]] = pd.Series(pic_df.to_numpy().flatten()).quantile(q=q)
		pickle.dump(self.quantiledict,open(os.path.join("preprocessor",'quantidict.pickle'),'wb'))
	
	def transform(self,data_df:pd.DataFrame) ->pd.DataFrame:
		data_df=data_df.copy()
		for i in range(len(self.randsat_col)):
			pic_col=data_df.filter(like= self.randsat_col.iloc[i,0],axis=1).columns
			pic_df=data_df.loc[:,pic_col].copy()
			f=partial(self._trainsforn_func,self.quantiledict[self.randsat_col.iloc[i,0]])
			data_df.loc[:,pic_col]=pic_df.transform(f,axis=1)
		return data_df

	def fit_transform(self,data_df:pd.DataFrame,q=0.85):
		self.fit(data_df,q)
		return self.transform(data_df)


class Pre_Process_Randsat():
	def __init__(self,) -> None:
		self._drop_outler=Drop_Outler_Randsat()
		self._fill_nan=Fill_Nan_Randsat()
	
	def get(self):
		self._drop_outler.get()

	def fit_transform(self,input:pd.DataFrame)->pd.DataFrame:
		result=self._drop_outler.fit_transform(input)
		result=self._fill_nan.transform(result)
		return result

	def transform(self,input:pd.DataFrame)->pd.DataFrame:
		result=self._drop_outler.transform(input)
		result=self._fill_nan.transform(result)
		return result

from sklearn import preprocessing as ps
import pickle
class Make_Feature_2000_2020():
	def __init__(self,) -> None:
		self.categorize=Categorize_Training_Data()
		self.min_randsat_peryear=pd.read_csv(os.path.join("preprocessor",'minimal_feature_description_for_Landsat_data.csv') ,index_col=0)
		self._dic={'中央値':'MED','最大値':'MAX','最小値':'MIN'}
	
	def _pic_randsat_data_from_peryear(self,datadf:pd.DataFrame,jp_str='中央値'):
		pic_col=self.min_randsat_peryear.loc[self.min_randsat_peryear.iloc[:,1].str.contains(jp_str),'年を除いた列名']
		_datadf=pd.concat([pd.DataFrame(columns=pic_col),datadf.year],axis=1)
		__datadf=datadf.filter(like='{}'.format(self._dic[jp_str]))
		for year in datadf.year.unique():
			_datadf.loc[_datadf.year==year,pic_col]=__datadf.loc[datadf.year==year,:].filter(like=str(int(year)),axis=1).values
		_datadf=_datadf.drop(labels=['year'],axis=1)
		return _datadf.astype('float64')

	def fit_transform(self,datadf:pd.DataFrame) -> pd.DataFrame:
		datadf=datadf.copy()
		result=self._pic_randsat_data_from_peryear(datadf.loc[datadf.year!=1999,:])
		return result

class Make_Feature_1999():
	def __init__(self,) -> None:
		self.categorize=Categorize_Training_Data()

	def fit_transform(self,datadf:pd.DataFrame) -> pd.DataFrame:
		datadf=datadf.loc[datadf.year==1999,:].copy()
		result=self.categorize.fit_transform(category_num=[0,1],traindf=datadf)
		droplist=['YMD','year','month','depth_original']
		result=result.drop(labels=droplist,axis=1)
		return result

import re
class Fillnan_Randsat_Series():
	def __init__(self,) -> None:
		self.min_randsat_peryear=pd.read_csv(os.path.join("preprocessor",'minimal_feature_description_for_Landsat_data.csv') ,index_col=0)
		self.feature_description_df=pd.read_excel(os.path.join("../official_data",'feature_description.xlsx'))
		self._dic={'中央値':'MED','最大値':'MAX','最小値':'MIN'}
	
	def _pic_randsat_data_from_peryear(self,datadf:pd.DataFrame,jp_str='中央値'):
			pic_col=self.min_randsat_peryear.loc[self.min_randsat_peryear.iloc[:,1].str.contains(jp_str),'年を除いた列名']
			_datadf=pd.concat([pd.DataFrame(columns=pic_col),datadf.year],axis=1)
			__datadf=datadf.filter(like='{}'.format(self._dic[jp_str]))
			for year in datadf.year.unique():
				if year == 1999:
					continue
				_datadf.loc[_datadf.year==year,pic_col]=__datadf.loc[datadf.year==year,:].filter(like=str(int(year)),axis=1).values
			_datadf=_datadf.drop(labels=['year'],axis=1)
			return _datadf.astype('float64')

	def fit(self,datadf:pd.DataFrame,jp_str='中央値') ->None:
		self._correspondence_dict=dict()
		self._complementary_dict=dict()
		for _info in self.feature_description_df.loc[self.feature_description_df.iloc[:,1]=='時系列「ランドサット」衛星画像データ',:].values:
			info=_info[2]
			col=_info[0]
			if col=='SAVImir' or (col in datadf.select_dtypes(include=object).columns):
				continue
			self._complementary_dict[col]=datadf.loc[:,col].mean()
			pic_df=self.min_randsat_peryear.loc[self.min_randsat_peryear.iloc[:,1].str.contains(re.escape(info)) & self.min_randsat_peryear.iloc[:,1].str.contains(jp_str),'年を除いた列名']#条件に合う年ごとのランドサットデータのカラム名を取得
			if len(pic_df)==0:
				continue
			pic_col=pic_df.values[0]
			self._correspondence_dict[col]=pic_col
			

	def _filnan_func(self,valdf,x):
		x=x.copy()
		x.loc[x.isna()]=valdf.loc[x.isna()]
		return x.copy()
	
	def transform(self,datadf:pd.DataFrame):
		datadf=datadf.copy()
		randsat_peryear=self._pic_randsat_data_from_peryear(datadf=datadf)
		for serise_col,peryer_col in self._correspondence_dict.items():
			f=partial(self._filnan_func,randsat_peryear.loc[:,peryer_col])
			datadf.loc[:,serise_col]=datadf.loc[:,serise_col].transform(f).copy()
		datadf=datadf.fillna(self._complementary_dict)
		return datadf
	
	def fit_transform(self,datadf,jp_str='中央値'):
		self.fit(datadf=datadf,jp_str=jp_str)
		return self.transform(datadf=datadf)
