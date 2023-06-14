from os import cpu_count, path
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as T
from lightning import LightningDataModule
from utils import visualize


class NyuDataset(Dataset):
    def __init__(self, root, csv, apply_augs=True):
        self.apply_augs = apply_augs
        if apply_augs:
            self.augs = T.Compose(
                [
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(),
                    T.Normalize(0, 1)
                ]
            )
        with open(path.join(root, csv)) as f:
            self.samples = list(
                map(
                    lambda pair: list(
                        map(
                            lambda file_name: path.join(root, file_name),
                            pair.strip().split(','),
                        )
                    ),
                    f.readlines(),
                )
            )

    def __getitem__(self, index):
        input_path, target_path = self.samples[index]
        input_image = Image.open(input_path)
        target_image = Image.open(target_path)
        x_tensor = T.functional.to_tensor(input_image)
        y_tensor = T.functional.to_tensor(target_image).float()
        y_tensor = y_tensor / y_tensor.max()
        # reduce the size to make the transforms less expensive
        x_tensor = T.functional.resize(
            x_tensor, size=[224, 224], antialias=True)
        if self.apply_augs:
            # reduce the targets to stack them later
            y_tensor = T.functional.resize(
                y_tensor, size=[224, 224], antialias=True)
            input_image = T.ColorJitter()(input_image)
            # stack and apply the same transformations to both
            tensors = torch.cat([x_tensor, y_tensor])
            tensors = self.augs(tensors)
            x_tensor = tensors[0:3]
            y_tensor = tensors[3:]
        y_tensor = T.functional.resize(y_tensor, size=[56, 56], antialias=True)
        return x_tensor, y_tensor

    def __len__(self):
        return len(self.samples)


class NyuDataModule(LightningDataModule):
    def __init__(self, root, csv_train, csv_test, batch_size=32, apply_augs=True):
        super().__init__()
        dataset = NyuDataset(root, csv_train, apply_augs)
        self.test = NyuDataset(root, csv_test, apply_augs)
        proportions = [.7, .3]
        lengths = [int(p * len(dataset)) for p in proportions]
        lengths[-1] = len(dataset) - sum(lengths[:-1])
        self.train, self.val = random_split(dataset, lengths)
        self.batch_size = batch_size
        self.workers = cpu_count()

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.workers,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.workers,
        )


if __name__ == '__main__':
    dataset = NyuDataset('', 'data/nyu2_train.csv')
    x, y = dataset[0]
    dm = NyuDataModule('', 'data/nyu2_train.csv', 'data/nyu2_test.csv')
    vis_batch = next(iter(dm.test_dataloader()))
    visualize('Data examples', vis_batch)
