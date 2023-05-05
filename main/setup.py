import re
from preprocessor.read_info import Categorize_Training_Data
import pandas as pd
import glob




def naiyou(info):
        return re.sub('[0-9]{4}年の','',info)
def ignore_year_col_name(x):
    return re.sub('_[0-9]{4}','',x)


def split_dataframe(df:pd.DataFrame, name:str)->None:
    index_num,column_num = df.shape
    for i in range(index_num//100+1):
        df.iloc[100*i:100*i+100,:].to_csv(f"../official_data/{name}s/{name}_{i}.csv",index=False)

def concat_dataframe(name:str):
    csv_files = glob.glob(f'../official_data/{name}s/*.csv')
    df = pd.DataFrame()
    for csv_file in csv_files:
        df = pd.concat([df, pd.read_csv(csv_file)], axis=0)
    df.to_csv(f'../official_data/{name}_data.csv')
    print(f'ok output {name}_data.csv')

def main():
    concat_dataframe("train")
    concat_dataframe("test")
    ctd=Categorize_Training_Data()
    ctd._set_catergory_num(category_num=[4])
    randsatdf=ctd.categorydf[ctd.categorydf.iloc[:,0].isin(ctd.select_calname)].copy()
    randsatdf=ctd.categorydf[ctd.categorydf.iloc[:,0].isin(ctd.select_calname)].copy()
    randsatdf['年を除いた列名']=randsatdf['列名'].map(ignore_year_col_name)
    randsatdf['年を除いた内容']=randsatdf.loc[:,'内容'].map(naiyou)
    randsatdf=randsatdf.loc[~randsatdf.iloc[:,3:].duplicated(),['年を除いた列名','年を除いた内容']].reset_index(drop=True)
    randsatdf.to_csv('preprocessor/minimal_feature_description_for_Landsat_data.csv')
    print('ok output minimal_feature_description_for_Landsat_data.csv')

if __name__ == "__main__":
    main()
