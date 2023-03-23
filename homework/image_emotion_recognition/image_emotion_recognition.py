from PIL import Image
import numpy as np
import torchvision.io
from facenet_pytorch import MTCNN
import torch
import numpy as np
import cv2
import torch
from torch.nn.functional import interpolate
from PIL import Image
import os

video_name='01-01-01-01-01-01-01.mp4'
mp4 = open(video_name,'rb').read()
frames=torchvision.io.read_video(video_name)[0]
mtcnn = MTCNN(image_size=224)

def get_size(img):
    if isinstance(img, (np.ndarray, torch.Tensor)):
        return img.shape[1::-1]
    else:
        return img.size

def save_img(face, save_path):
    face.save(save_path)

def extract_face(img, box, image_size=160, margin=[0,0], save_path=None):
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
    box[1]=box[1]-margin[0]
    box[3]=box[3]+margin[0]
    box[0]=box[0]-margin[1]
    box[2]=box[2]+margin[1]
    face = img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) + "/", exist_ok=True)
        save_img(face, save_path)

    return face,box
def imresample(img, sz):
    im_data = interpolate(img, size=sz, mode="area")
    return im_data


if not os.path.exists('faces'):
    os.makedirs('faces')
for (i, frame) in enumerate(frames):
  img_tensor = frame.numpy()
  img = Image.fromarray(img_tensor)
  box = mtcnn.detect(img)
  extract_face(img,box[0][0], save_path=f'faces/face_{i}.png')
