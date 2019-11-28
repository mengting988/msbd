import numpy as np
import pandas as pd
import datetime 


def extract_date(df, column):
    df[column+"_year"] = df[column].apply(lambda x: x.year)
    df[column+"_month"] = df[column].apply(lambda x: x.month)

def feature_engineering():

    parser = lambda date: pd.datetime.strptime(date, '%b %d, %Y')
    parser2 = lambda date: pd.datetime.strptime(date, '%d-%b-%Y')
    df1 = pd.read_csv('./data/train.csv', index_col = 'id', parse_dates=[7], date_parser=parser)
    df2 = pd.read_csv('./data/test.csv', index_col = 'id', parse_dates=[6], date_parser=parser)
    df = pd.concat([df1, df2], axis=0, sort=False)

    df['release_date'] = pd.to_datetime(df['release_date'], format='%d-%b-%y')
    df['time_diff'] = pd.DataFrame(pd.to_datetime(df['purchase_date']) - pd.to_datetime(df['release_date']))
    df['time_diff'] = df['time_diff'].astype('timedelta64[D]')

    extract_date(df, 'purchase_date')
    extract_date(df, 'release_date')
    
    df_genres = df["genres"].str.get_dummies(",") 
    df_categories = df["categories"].str.get_dummies(",") 
    df_all = pd.concat([df, df_genres, df_categories], axis=1)
    df_final = df_all.drop(['is_free', 'genres', 'categories', 'tags', 'purchase_date', 'release_date'], axis=1)
    
    return df_final