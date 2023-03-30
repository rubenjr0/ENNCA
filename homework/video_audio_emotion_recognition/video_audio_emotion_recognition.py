import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.io import read_video
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
from copy import deepcopy
import numpy as np
import cv2
import torch
from torch.nn.functional import interpolate
from PIL import Image


def get_size(img):
    if isinstance(img, (np.ndarray, torch.Tensor)):
        return img.shape[1::-1]
    else:
        return img.size


def extract_face(img, box, image_size=160, margin=[0, 0], save_path=None):
    """Extract face + margin from PIL Image given bounding box.

    Arguments:
        img {PIL.Image} -- A PIL Image.
        box {numpy.ndarray} -- Four-element bounding box.
        image_size {int} -- Output image size in pixels. The image will be square.
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image.
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size.
        save_path {str} -- Save path for extracted face image. (default: {None})

    Returns:
        torch.tensor -- tensor representing the extracted face.
    """
    box[1] = box[1] - margin[0]
    box[3] = box[3] + margin[0]
    box[0] = box[0] - margin[1]
    box[2] = box[2] + margin[1]
    face = (
        img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)
    )
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) + '/', exist_ok=True)
        save_img(face, save_path)

    return face, box


def imresample(img, sz):
    im_data = interpolate(img, size=sz, mode='area')
    return im_data


def crop_resize(img, box, image_size):
    if isinstance(img, np.ndarray):
        img = img[box[1] : box[3], box[0] : box[2]]
        out = cv2.resize(
            img, (image_size, image_size), interpolation=cv2.INTER_AREA
        ).copy()
    elif isinstance(img, torch.Tensor):
        img = img[box[1] : box[3], box[0] : box[2]]
        out = (
            imresample(
                img.permute(2, 0, 1).unsqueeze(0).float(),
                (image_size, image_size),
            )
            .byte()
            .squeeze(0)
            .permute(1, 2, 0)
        )
    else:
        out = (
            img.crop(box)
            .copy()
            .resize((image_size, image_size), Image.BILINEAR)
        )
    return out


class Frames_prediction_dataset(Dataset):
    def __init__(self, frames, transform=None):
        self.frames = frames
        self.detector = MTCNN()
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        image = self.frames[idx]
        box1 = self.detector.detect(image)
        crop_image, box2 = extract_face(
            Image.fromarray(image.numpy()),
            box1[0][0],
            image_size=224,
            margin=(20, 84),
        )
        if self.transform:
            crop_image = self.transform(crop_image)
        return crop_image, box1

    def draw_prediction(self, idx, box):
        image = self.frames[idx]
        box = self.detector.detect(image)
        img = Image.fromarray(image.numpy())
        cv2.rectangle(img, (left, top), (right, bottom), colors[prediction], 2)
        return crop_image, box


video_name = '01-01-01-01-01-01-01.mp4'

transformations = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

frames = read_video(video_name)[0]
data = Frames_prediction_dataset(frames, transformations)
dataloader = DataLoader(data, batch_size=1, num_workers=1)

image_emotion_classifier = models.resnet18()
image_emotion_classifier.fc = nn.Linear(512, 7)
image_emotion_classifier.load_state_dict(
    torch.load('emotion.pth', map_location=torch.device('cpu'))
)

emotion_representitive = deepcopy(image_emotion_classifier)
# removes the fully connected layer, preserving the 512 features
emotion_representitive.fc = nn.Sequential() 


class VideoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = emotion_representitive
        self.lstm = nn.LSTM(input_size=512, hidden_size=512)
        self.fc = nn.Linear(512, 8)

    def forward(self, x):
        x = self.cnn(x)
        _, (_, x) = self.lstm(x)
        x = self.fc(x)
        # we apply softmax so that the sum of probabilities
        # for each class is 1.0
        return torch.softmax(x, 1)


vn = VideoNet()

# We take a face from the data loader
face, _ = next(iter(dataloader))
print('input shape:', face.shape)
logits = vn(face)
print('output shape:', logits.shape)
print('output:', logits)
