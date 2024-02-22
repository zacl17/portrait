import cv2
import time
import numpy as np
import sys
from change_mask import trans_by_mask
import warnings

cv2.setNumThreads(8)
# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=FutureWarning)

# def resample_laebl(input_signal, target_length):
#         """Samples a PPG sequence into specific length."""
#         return np.interp(
#             np.linspace(
#                 1, input_signal.shape[0], target_length), np.linspace(
#                 1, input_signal.shape[0], input_signal.shape[0]), input_signal)  

def main():
    label_id = '7'
    label_path = 'git/s'+label_id+'/bvp_s'+label_id+'_T2.csv'
    label_bvps = []
    with open(label_path, 'r') as f_obj:
        data = f_obj.readlines()
        for d in data:
            label_bvps.append(float(d.strip().split()[-1]))

    result_path = 'mask'+'/_T2_result.avi'
    cap = cv2.VideoCapture('git/s'+label_id+'/vid_s'+label_id+'_T2.avi')  
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(result_path, fourcc, fps, (width, height), True)
    # label_bvps = resample_laebl(np.asarray(label_bvps), frames)
    background = cv2.imread('HR/background.jpg', 1)
    background = cv2.resize(background, (width, height))
    pred3 = None
    while cap.isOpened():
        ret, frame = cap.read()  #返回是否读取成功，以及该帧的图像
        if not ret:break
        # frame_in, messages, box, pred1, pred2 = check_roi_multi(frame_in, box, pred1, pred2, show)
        frame_result, pred3 = trans_by_mask(np.copy(frame), pred3, background)
        writer.write(frame_result)
    writer.release()


if __name__ == '__main__':
    main()