import pandas as pd 
#import numpy as np

# reading the file
df = pd.read_csv("D:/git/Classification-of-BlockChain-transactions-via-Supervised-ML-Models/Machine_Learning/dataset/all_three_binery.csv") 

# Seagreagting the features and labels 
Y = df[['label']]
X = df[['balance' ,'rec/sent','amount','size','weight','version','lock_time','is_coinbase','has_witness','input_count','output_count','input_total_usd','output_total_usd','fee_usd','fee_per_kb_usd','fee_per_kwu_usd','cdd_total']]

# preprocessing 
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
x = scaler.fit_transform(X)
print('scaler')

from sklearn.ensemble import RandomForestClassifier 
model = RandomForestClassifier(n_jobs=-1) 
model.fit(x, Y) 
print('random')

import pickle
with open("model.pkl", "wb") as model_pkl : 
    pickle.dump(model, model_pkl) 

with open("scaler.pkl", 'wb') as scaler_pkl: 
    pickle.dump(scaler, scaler_pkl)  
print('Done')