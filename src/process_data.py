import pandas as pd
import torch
import lightning as pl
from sklearn.model_selection import KFold 
from torch.utils.data import Dataset,DataLoader

df=pd.read_csv(r"data\news_summary (1).csv",encoding="latin-1")

df=df.iloc[:,-2:]

df.columns = ['summary', 'text']

df = df.dropna()

df=df.reset_index(drop=True)

kfold=KFold(n_splits=5)
df["fold"]=-1
for fold,(train_idx,val_idx) in enumerate(kfold.split(X=df)):
    df.loc[val_idx,"fold"]=fold

