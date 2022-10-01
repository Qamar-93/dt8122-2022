
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import os 

class LoadData(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        return self.df[item]
    
    
def data_load_split(dataset_name, path="../data/", batch_size=32, training_frac = 0.8):
    df = pd.read_csv(os.path.join(path, f"{dataset_name}.csv"))
    
    return split_data(df, batch_size, training_frac)
    
    
def split_data(df, batch_size, training_frac=0.8):
    
    train_df = df.sample(frac = training_frac)
    test_df = df.drop(train_df.index)

    train_df_tensor = torch.tensor(list(train_df.values))
    test_df_tensor = torch.tensor(list(test_df.values))

    train_loader= DataLoader(LoadData(train_df_tensor), batch_size=batch_size, shuffle=True)

    test_loader= DataLoader(LoadData(test_df_tensor), batch_size=batch_size, shuffle=True)
    return train_loader, test_loader