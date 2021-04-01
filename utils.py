import torch

def get_count(df, cat):
    """
    print(df['Path'])
    print(df['Path'].str)
    print(df['Path'].str.contains(cat))
    """
    return df[df['Path'].str.contains(cat)]['Count'].sum()

def np_tensor(x):
    return torch.FloatTensor([x])