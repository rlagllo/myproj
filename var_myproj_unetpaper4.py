
import torch
from torch.utils.data import DataLoader
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchsummary import summary
from PIL import Image
import os
import numpy as np
import glob
import gc
from var_myproj_UNetpaper2 import unet
from tqdm import tqdm # 반복문 진행 상황을 진행바 로 보여줌
import cv2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
        
    image_folder_path = "C:\\Users\\김해창\\Desktop\\unet\\cityscapes_data\\processed_val"
    label_foler_path = "C:\\Users\\김해창\\Desktop\\unet\\cityscapes_data\\processed_val_label"
    model_path = "C:\\Users\\김해창\\Desktop\\unet\\unet_epoch40.pth"
    input_video_path = "C:\\Users\\김해창\\Desktop\\unet\\road_driving.mp4"
    output_video_path = "C:\\Users\\김해창\\Desktop\\unet\\video\\unet_video.mp4"
    

    cap = cv2.VideoCapture(input_video_path) # opencv, 영상 받아옴               ==============   cv2는 RGB가 아니고 BGR을 써서 바꿔주는 작업이 필요함
    fps = int(cap.get(cv2.CAP_PROP_FPS)) # 비디오의 fps가 몇인지 반환
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 프레임의 가로크기 반환
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 너비 반환
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 비디오가 총 몇 프레임인지 반환
    resize = transforms.Resize((256, 256))
    frame_paths = [] 
    for i in tqdm(range(frame_count), desc="Extracting Frames"): # 넘파이 리사이즈: cv2 or skimage, 토치텐서 리사이즈: torchvision.tranforms
        ret, frame = cap.read() # 받아온 영상 cap에서 하나의 프레임을 읽음. ret:성공여부, frame: 읽은 프레임 넘파이로 반환
        if not ret:
            break
        #cv2.imshow("frame", frame) # (1142,1146,3)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.tensor(frame.transpose(2, 0, 1), dtype=torch.float32)
        frame = resize(frame)
        #print(frame.shape)
        frame = frame.unsqueeze(0)
        #print(frame.shape) # 1, 3, 256, 256
        frame_paths.append(frame)
    cap.release() # cap 객체 삭제
    
    output_frames = []
    
    #for frame_path in tqdm(frame_paths, desc="processing frames"):
        #frame = cv2.imread(frame_path)
        
       # frame_resized = 
        
    
    
    
    state_dict = torch.load(model_path, weights_only=True)

    class_to_color = {0: (0, 0, 0), 1: (111, 74, 0), 2: (81, 0, 81), 3: (128, 64, 128), 4: (244, 35, 232), 
                    5: (250, 170, 160), 6: (230, 150, 140), 7: (70, 70, 70), 8: (102, 102, 156), 9: (190, 153, 153), 
                    10: (180, 165, 180), 11: (150, 100, 100), 12: (150, 120, 90), 13: (153, 153, 153), 
                    14: (250, 170, 30), 15: (220, 220, 0), 16: (107, 142, 35), 17: (152, 251, 152), 18: (70, 130, 180),
                    19: (220, 20, 60), 20: (255, 0, 0), 21: (0, 0, 142), 22: (0, 0, 70), 23: (0, 60, 100), 24: (0, 0, 90), 
                    25: (0, 0, 110), 26: (0, 80, 100), 27: (0, 0, 230), 28: (119, 11, 32)}

    model = unet(3,29).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    with torch.no_grad():
        for frame in tqdm(frame_paths, desc='processing frames'):
            frame = frame.to(device)
            output_image = model(frame)
            #print(output_image.shape)
            segmented_image = torch.argmax(output_image,dim=1).squeeze(0)
            segmented_sh = segmented_image.cpu()
            segmented_sh = np.array([class_to_color[int(val)] for val in segmented_sh.flatten()])
            segmented_sh = segmented_sh.reshape((256,256,3)).astype(np.uint8)
            segmented_sh = cv2.cvtColor(segmented_sh, cv2.COLOR_RGB2BGR)
            output_frames.append(segmented_sh)

            # print(segmented_sh.shape)
            # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            # print(frame.shape)
            # original = frame.squeeze(0).permute(1,2,0).cpu().numpy() / 255.0
            
            # ax[0].imshow(original)
            # ax[1].imshow(segmented_sh)
            # plt.show()
            
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (256,256))
    
    for frame in output_frames:
        out.write(frame)
    out.release()
    
    print("done")