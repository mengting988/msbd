import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from feature_engineering import feature_engineering

def main():   
    df = feature_engineering()
    df_train = df.iloc[0:357, :]
    df_test = df.iloc[357:447, :]
    
    X = df_train.iloc[:, 1:58]
    y = df_train.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=0)

    # following is RandomForestRegressor, or choose other model in model.py
    RF = RandomForestRegressor(max_depth=2, random_state=0,
                             max_features='sqrt', n_estimators=100)
    RF.fit(X_train, y_train)  

    y_pred = RF.predict(X_test)

    print("Mean squared error: %.2f"
        % mean_squared_error(y_test, y_pred))
    
    RF.fit(X, y)
    test = df_test.iloc[:, 1:58]
    y_pred = RF.predict(test)
    
    df = pd.DataFrame(y_pred, columns=['playtime_forever'])
    df['id']=df.index
    df = df[['id','playtime_forever']]
    df.to_csv(r'./result/submission.csv',index=False)
    
    
    
if __name__ == '__main__':
    main()