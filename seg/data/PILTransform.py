import numpy as np
import torch
import random
import math
import numbers
from torchvision.transforms import Pad
from torchvision.transforms import functional as F
from PIL import Image
from .CVTransforms import ImageEnhance
import cv2

class data_aug_color(object):

    def __call__(self, image, label):
        if random.random() < 0.5:
            return image, label
        # image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        random_factor = np.random.randint(4, 17) / 10.
        color_image = ImageEnhance.Color(image).enhance(random_factor)
        random_factor = np.random.randint(4, 17) / 10.
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
        random_factor = np.random.randint(6, 15) / 10.
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
        random_factor = np.random.randint(8, 13) / 10.
        image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)
        return image, label


class Normalize(object):
    '''
        Normalize the tensors
    '''

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], scaleIn=1):
        '''
        :param mean: global mean computed from dataset
        :param std: global std computed from dataset
        '''

        self.mean = mean
        self.std = std
        self.scale = scaleIn


    def __call__(self, rgb_img, label_img=None):

        if self.scale != 1:
            w, h = label_img.size
            label_img = label_img.resize((w//self.scale, h//self.scale), Image.NEAREST)


        rgb_img = F.to_tensor(rgb_img) # convert to tensor (values between 0 and 1)
        rgb_img = F.normalize(rgb_img, self.mean, self.std) # normalize the tensor
        label_img = torch.LongTensor(np.array(label_img).astype(np.int64))


        return rgb_img, label_img

class RandomFlip(object):
    '''
        Random Flipping
    '''
    def __call__(self, rgb_img, label_img):
        if random.random() < 0.5:
            rgb_img = rgb_img.transpose(Image.FLIP_LEFT_RIGHT)
            label_img = label_img.transpose(Image.FLIP_LEFT_RIGHT)
        return rgb_img, label_img

class Resize(object):
    '''
        Resize the images
    '''
    def __init__(self, size=(512, 512)):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

    def __call__(self, rgb_img, label_img):
        rgb_img = rgb_img.resize(self.size, Image.BILINEAR)
        label_img = label_img.resize(self.size, Image.NEAREST)
        return rgb_img, label_img


class Compose(object):
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
