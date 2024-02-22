import cv2
import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
 
 
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
 
 
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
 
 
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


image = cv2.imread('/seg/Nukki/baidu_V1/input/27.png') 
h, w, c = image.shape
x, y = w//2, h//2
print("[%s]正在转换图片格式......" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
print("[%s]正在初始化模型参数......" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

sam_checkpoint = "LLMs/segment-anything-main/sam_vit_h_4b8939.pth"  # 定义模型路径
model_type = "default"  # 定义模型类型
device = "cuda"  # "cpu"  or  "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)  # 定义模型参数

# entire image
# mask_generator = SamAutomaticMaskGenerator(sam)
# masks_list = mask_generator.generate(image)
# input_label = np.array([1])
# for i, mask in enumerate(masks_list):
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image)
#     show_mask(mask['segmentation'], plt.gca())
#     show_points(np.array(mask['point_coords'], dtype=int), input_label, plt.gca())
#     plt.title(f"Mask {i + 1}, Score: {mask['stability_score']:.3f}", fontsize=18)
#     plt.axis('off')
#     plt.show()


predictor = SamPredictor(sam)  
predictor.set_image(image)
print("[%s]正在分割图片......" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
input_point = np.array([[x, y]])
input_label = np.array([1])  
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()
masks, scores, logit = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,  
)
print(masks.shape)  
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()

