from torch.utils.data import Dataset, DataLoader
from PIL import Image

class SlideDataset(Dataset):
    def __init__(self, metadata, transform=None):
        self.metadata = metadata
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image = Image.open(row['filepath']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, row['label']

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image = Image.open(row['filepath']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, row['label']

def create_data_loader(metadata, batch_size, shuffle=True, num_workers=2, transform=None):
    dataset = SlideDataset(metadata, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)