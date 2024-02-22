import argparse
import logging
import sys
sys.path.append('HR/mask/seg')
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import data.CVTransforms as cvTransforms
from SINet import Dnc_SINet
# from tools.kalman import KalmanFilter


class Predict():
    def __init__(self, model_path, model_arch, model_config, threshold, input=None, output=None, save=False):
        self.model_path = model_path
        self.model_arch = model_arch
        self.num_classes = model_config[0]
        self.p = model_config[1]
        self.q = model_config[2]
        self.chnn = model_config[3]
        self.threshold = threshold
        self.input = input
        self.output = output
        self.save = save

    def init_model(self):
        if self.model_arch == 'SiNet':
            net =  Dnc_SINet(p=self.p, q=self.q, classes=self.num_classes, chnn=self.chnn)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.to(device=device)
        state_dict = torch.load(self.model_path, map_location=device)['state_dict']
        self.mask_values = state_dict.pop('mask_values', [0, 1])
        net.load_state_dict(state_dict)

        net.eval()
        self.net = net
        self.device = device
    
    def get_output_filenames(self):
        def _generate_name(fn):
            return f'{os.path.splitext(fn)[0]}_OUT.png'

        return self.output or list(map(_generate_name, self.input))


    def mask_to_image(self, mask: np.ndarray):
        if self.mask_values == [0, 1]:
            out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

        for i, v in enumerate(self.mask_values):
            out[mask == i] = v

        return out
        
    def predict_img(self, full_img):
        img = cv2.resize(full_img, (224,224))
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy((img - 128.)/128.)
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            output = self.net(img).cpu()
            output = F.interpolate(output, (full_img.shape[0], full_img.shape[1]), mode='bilinear')
            if self.num_classes > 1:
                mask = output.argmax(dim=1)
            else:
                mask = torch.sigmoid(output) > self.threshold

        return mask[0].long().squeeze().numpy()


    def result(self, img):
        if not isinstance(img, str):
            mask = self.predict_img(img)
            return self.mask_to_image(mask)
        else:
            out_files = self.get_output_filenames()
            res = []
            for i, filename in enumerate(self.input):
                img = cv2.imread(filename, 1)
                mask = self.predict_img(full_img=img)
                guide = img.astype(np.uint8)
                input = mask.astype(np.uint8)
                mask = cv2.ximgproc.guidedFilter(guide, input, radius=7, eps=1e-3)
                res.append(mask)
                if self.save:
                    out_filename = out_files[i]
                    result = self.mask_to_image(mask)
                    result[ np.where(result == 1)] = 255
                    cv2.imwrite(out_filename, result)
            return res

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='basic_model/segmentation/ExtC3_SINet/result/Dnc_SINet-01-19_2331/checkpoint.pth.tar', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', default=['Data/seg/Nukki/baidu_V1/input/1.png','Data/seg/Nukki/baidu_V1/input/27.png'],help='Filenames of input images')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', default=['result11.jpg', 'result22.jpg'], help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    
    return parser.parse_args()

    
if __name__ == '__main__':

    args = get_args()
    pred = Predict(model_arch='SiNet', model_path=args.model, model_config=[1, 2, 8, 1], threshold=0.5, input=args.input, output=args.output, save=True)
    pred.init_model()
    masks = pred.result('img')
    for i in masks:
        print(type(i))



