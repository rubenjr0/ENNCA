from os import listdir
from PIL import Image
from torch import cat
from random import choice
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, to_pil_image, invert


class NyuDataset(Dataset):
    def __init__(self, root):
        with open(root) as f:
            self.samples = list(
                map(lambda pair: pair.strip().split(','), f.readlines()))

    def __getitem__(self, index):
        input_path, target_path = self.samples[index]
        input_image = Image.open(input_path).convert('RGB')
        input_image = T.ColorJitter()(input_image)
        target_image = Image.open(target_path).convert('L')
        t = T.Compose([
            # T.RandomCrop(256),
            T.RandomHorizontalFlip(),
            T.Normalize(0, 1)
        ])
        input_tensor = to_tensor(input_image)
        target_tensor = to_tensor(target_image)
        tensors = cat([
            input_tensor,
            target_tensor])
        tensors = t(tensors)
        return tensors[0:3], invert(tensors[3:])

    def __len__(self):
        return len(self.samples)


train_dataset_instance = NyuDataset(
    root='data/nyu2_train.csv')


def main():
    input_tensor, target_tensor = train_dataset_instance[0]
    print('Input shape:', input_tensor.shape)
    print('Target shape:', target_tensor.shape)
    to_pil_image(input_tensor, 'RGB').save('demos/dataset_image.png')
    to_pil_image(target_tensor, 'L').save('demos/target_mage.png')


if __name__ == '__main__':
    main()
