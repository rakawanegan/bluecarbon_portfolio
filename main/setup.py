import re
from preprocessor.read_info import Categorize_Training_Data

def naiyou(info):
        return re.sub('[0-9]{4}年の','',info)
def ignore_year_col_name(x):
    return re.sub('_[0-9]{4}','',x)

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../official_data')
    parser.add_argument('--out-dir', default = 'preprocessor')
    args = parser.parse_args()
    return args
import os

def main():
    args=parse_args()
    datadir=os.path.abspath(args.data_dir)
    outdir=os.path.abspath(args.out_dir)
    ctd=Categorize_Training_Data(datadir)
    ctd._set_catergory_num(category_num=[4])
    randsatdf=ctd.categorydf[ctd.categorydf.iloc[:,0].isin(ctd.select_calname)].copy()
    randsatdf=ctd.categorydf[ctd.categorydf.iloc[:,0].isin(ctd.select_calname)].copy()
    randsatdf['年を除いた列名']=randsatdf['列名'].map(ignore_year_col_name)
    randsatdf['年を除いた内容']=randsatdf.loc[:,'内容'].map(naiyou)
    randsatdf=randsatdf.loc[~randsatdf.iloc[:,3:].duplicated(),['年を除いた列名','年を除いた内容']].reset_index(drop=True)
    randsatdf.to_csv(os.path.join(outdir,'minimal_feature_description_for_Landsat_data.csv'))
    print('ok output minimal_feature_description_for_Landsat_data.csv')

if __name__ == "__main__":
    main()
