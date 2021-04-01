import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torchvision.datasets.folder import pil_loader

import pandas as pd
from tqdm import tqdm
# import tqdm
import os


phase_cat = ['train', 'valid']


def get_study_data(study_type, DIR):
    case_data = {}
    case_label = {'positive': 1, 'negative': 0}
    for phase in phase_cat:
        BASE_DIR = DIR.format(phase, study_type)
        # test = list(os.walk(BASE_DIR))
        patients = list(os.walk(BASE_DIR))[0][1]
        case_data[phase] = pd.DataFrame(columns=['Path', 'Count', 'Label'])
        i=0
        for patient in tqdm(patients):
            for case in os.listdir(BASE_DIR + patient):
                label = case_label[case.split('_')[1]]
                path = BASE_DIR + patient + '/' + case + '/'
                case_data[phase].loc[i] = [path, len(os.listdir(path)), label]
                i+=1
    return case_data

class CustomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        # self.x = torch.from_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        case_path = self.df.iloc[idx, 0]
        case_num = self.df.iloc[idx, 1]
        images = []
        for i in range(case_num):
            image = pil_loader(case_path + 'images{}.png'.format(i+1))
            images.append(self.transform(image))
        images = torch.stack(images)
        label = self.df.iloc[idx, 2]
        item = {'images' :images, 'label':label}
        return item


def get_dataloaders(data, batch_size=8):
    data_transforms = {
        'train' : transforms.Compose([
            transforms.Resize((244, 244)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid' : transforms.Compose([
            transforms.Resize((244, 244)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    image_datasets = {x: CustomDataset(data[x], transform=data_transforms[x]) for x in phase_cat}
    data_loaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in phase_cat}
    return data_loaders