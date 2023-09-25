import process_data
import pandas as pd
from dataset import *
from model import *
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

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



folds=5
N_EPOCHS = 1
BATCH_SIZE=8
MODEL_NAME = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model_1 = SummaryModel()


if __name__=="__main__":

    for fold in range(folds):
        train_df=df[df["fold"]!=fold]
        valid_df=df[df["fold"]==fold]

        data_module = NewsDataModule(train_df, valid_df,tokenizer,batch_size=BATCH_SIZE)

        callbacks = ModelCheckpoint(dirpath="/content/checkpoints",
                                    filename="base-checkpoint",
                                    save_top_k=1,
                                    verbose=True,
                                    monitor="val_loss",
                                    mode='min'
                                )

        logger = TensorBoardLogger("lightning_logs", name="news_summary")

        trainer= Trainer(logger=logger,
                        callbacks=callbacks,
                        max_epochs=N_EPOCHS,)

        trainer.fit(model_1, data_module)