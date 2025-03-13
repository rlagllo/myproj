
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import torch.nn.functional as F
from PIL import Image
import os
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.cluster import KMeans
from cityscapesscripts.helpers import labels as helper_labels
import glob

def split(image):
    image = np.array(image)
    view, label = image[:,:256,:], image[:,256:,:]
    return view, label

trainx_path = r'C:\Users\김해창\Desktop\unet\cityscapes_data\val'
before = glob.glob(os.path.join(trainx_path,'*'))
for file in before:
    
    basename,_ = os.path.splitext(os.path.basename(file))
    #print(basename)
    # sample_image_fp = os.path.join(trainx_path, '3.jpg')
    # sample_image_fp2 = os.path.join(trainx_path, '10.jpg')

    sample_image = Image.open(file).convert("RGB")
    #smaple_image2 = Image.open(sample_image_fp2).convert("RGB")
    gt, labelimg = split(sample_image)
    # gtt, labelimgg = split(smaple_image2)
    # fig, ax = plt.subplots(2, 2, figsize=(10, 5))


    # #img = original_image.permute(1, 2, 0) # [C,H,W]인 이미지 텐서형식을 [H,W,C]로 만들어서 matplotlib할거임
    # #img = (img - np.min(img)) / (np.max(img) - np.min(img)) #정규화, 원본 이미지이지만 matplotlib은 정규화해서 시각화하는게 일반적 + 정규화시킨 heatmap과 맞추기 위해
    # ax[0,0].imshow(gt)
    # ax[0,0].set_title("Original Image")
    # ax[0,0].axis('off')    
    # ax[0,1].imshow(labelimg)
    # ax[0,1].set_title("segmentation")
    # ax[0,1].axis('off')       
    # ax[1,0].imshow(gtt)
    # ax[1,0].axis('off') 
    # ax[1,1].imshow(labelimgg)
    # ax[1,1].axis('off') 
    # fig.tight_layout()
    # plt.show()     


    labels = helper_labels.labels

    # Print all the labels
    # print("List of cityscapes labels:")
    # print("")
    # print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12} | {:>1}"
    #     .format( 'name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval', 'color' ))
    # print("    " + ('-' * 110))
    color_to_class = {} # 키: RGB, 값: text
    text_to_color = {} # 키: text, 값: RGB
    #class_text = np.zeros((256,256),dtype='str')
    class_text = []
    for idx, label in enumerate(labels):
        color_to_class[label.color] = label.name
        #class_text.append(label.name)
    #     print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12} | {}".format( label.name, label.id, label.trainId, label.category, label.categoryId, label.hasInstances, label.ignoreInEval, label.color))
    # print("")
    #print("color_to_class:     ", color_to_class, "color_to_class size", len(color_to_class))  # 0,0,0 축소돼서 총 29개 클래스
    for coord in color_to_class:
        object = color_to_class[coord]
        text_to_color[object] = coord
    #label_rgb = np.array(Image.open(sample_image_fp))
    pixel_classnum = np.zeros((256,256), dtype= np.int64)
    #pixel_text = np.zeros((256,256), dtype= 'object')
    # for rgb, class_id in color_to_class.items():
    #     matches = np.all(label_rgb == rgb, axis=-1)
    #     label_class[matches] = class_id


    colors = np.array([
        [0, 0, 0], [111, 74, 0], [81, 0, 81], [128, 64, 128], [244, 35, 232],
        [250, 170, 160], [230, 150, 140], [70, 70, 70], [102, 102, 156], [190, 153, 153],
        [180, 165, 180], [150, 100, 100], [150, 120, 90], [153, 153, 153], [250, 170, 30],
        [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 0, 90],
        [0, 0, 110], [0, 80, 100], [0, 0, 230], [119, 11, 32]
    ])

    class_ids = np.array([
        0, 1, 2, 3, 4,
        5, 6, 7, 8, 9,
        10, 11, 12, 13, 14,
        15, 16, 17, 18, 19,
        20, 21, 22, 23, 24,
        25, 26, 27, 28
    ])
    
    
    class_to_color = {}
    for i in range(29):
        class_to_color[class_ids[i].item()] = tuple(map(int, colors[i]))
        
    print(class_to_color)
    
    #print("\n\n\n\n\n\n\n",color_to_class[(0,0,0)],"\n\n\n\n\n")
    for i in range(256):
        for j in range(256):
            pixel = labelimg[i, j]
            # RGB 거리 계산
            distances = np.sqrt(np.sum((colors - pixel)**2, axis=1))
            closest_class = np.argmin(distances) # argmin: 제일 작은 애가 몇 번째인지 반환
            pixel_classnum[i, j] = class_ids[closest_class]
    # print(pixel_classnum)
    # print(pixel_classnum.shape) # 256 256
    # processed = np.zeros((256,256,3),dtype=np.uint8)

    # for i in range(256):
    #     for j in range(256):
    #         text = pixel_classnum[i,j]
    #         processed[i,j] = color_to_class[text]
    # print(processed)
    # plt.imshow(processed)
    # plt.axis('off')
    # plt.show()

    #final = np.hstack((gt,pixel_classnum))

    # plt.imshow(final)
    # plt.show()
    processed_train_path = r'C:\Users\김해창\Desktop\unet\cityscapes_data\processed_val'
    precessed_label_path = "C:\\Users\\김해창\\Desktop\\unet\\cityscapes_data\\processed_val_label"
    file_name = f"{basename}.npy"

    np.save(os.path.join(processed_train_path,file_name),gt)
    np.save(os.path.join(precessed_label_path,file_name),pixel_classnum)
    print(f"saved {file_name}")
    #print("\n\ntext_to_color: ", class_to_color) # 어떤 클래스가 어떤 RGB값인지
    # print(class_text) 
    #print(pixel_text) # 레이블 이미지의 각 픽셀이 어느 클래스에 속하는 지

