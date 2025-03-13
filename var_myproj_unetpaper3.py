
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from PIL import Image
import os
import numpy as np
import glob
import gc
from var_myproj_UNetpaper2 import unet, mydataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
        
    image_folder_path = "C:\\Users\\김해창\\Desktop\\unet\\cityscapes_data\\processed_val"
    label_foler_path = "C:\\Users\\김해창\\Desktop\\unet\\cityscapes_data\\processed_val_label"
    model_path = "C:\\Users\\김해창\\Desktop\\unet\\unet_epoch40.pth"
        
    dataset = mydataset(image_folder_path, label_foler_path)
    dataloader = DataLoader(dataset, batch_size=8, num_workers=4,shuffle=True)

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
    #summary(model, input_size=(3, 256, 256))
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            segmented = model(images)
            segmented_image = torch.argmax(segmented,dim=1)
            segmented_sh = segmented_image[0].cpu()
            segmented_sh = np.array([class_to_color[int(val)] for val in segmented_sh.flatten()])
            segmented_sh = segmented_sh.reshape(256,256,3)
            labelimg = labels[0].cpu()
            labelimg_sh = np.array([class_to_color[int(val)] for val in labelimg.flatten()])
            labelimg_sh = labelimg_sh.reshape(256,256,3)
            original_image = images[0].permute(1,2,0).cpu().numpy()
            original_image = original_image / 255.0
            
            fig, ax = plt.subplots(1,3, figsize=(15,5))
            ax[0].imshow(original_image)
            ax[0].axis('off')
            ax[1].imshow(labelimg_sh)
            ax[1].axis('off')
            ax[2].imshow(segmented_sh)
            ax[2].axis('off')
            plt.show()