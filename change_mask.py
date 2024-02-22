import sys
sys.path.append('HR/mask/seg')
from predict import Predict
import cv2
import numpy as np

def trans_by_mask(image, pred, background):
    if not pred:
        pred = Predict(model_arch='SiNet', model_path='basic_model/segmentation/ExtC3_SINet/result/Dnc_SINet-01-19_2331/checkpoint.pth.tar', model_config=[1, 2, 8, 1], threshold=0.5)
        pred.init_model()
    mask = np.array(pred.result(img=image), dtype=np.uint8)
    image[np.where(mask==0)] = background[np.where(mask==0)]
    return image, pred

    



