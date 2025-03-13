
import os
import matplotlib.pyplot as plt
import csv
import numpy as np
import glob
from PIL import Image

base_path = "C:\\Users\\김해창\\Desktop\\VAR"
csv_files = glob.glob(os.path.join(base_path,"mnist_train*.csv"))

for index, csv_file in enumerate(csv_files):
    #subrow = 7
    #subcol = 7
    with open(csv_file,"r") as csv_file:
        reader = csv.reader(csv_file)
        rows = list(reader)
    #fig, axes = plt.subplots(subrow, subcol, figsize=(12, 12))
    for i,row in enumerate(rows):
        # if i>0 and i % (subrow*subcol) == 0:
        #     plt.tight_layout()
        #     plt.show()
        #     fig, axes = plt.subplots(subrow, subcol, figsize=(12, 12))
        title = row[0]
        row = list(map(int,row[1:]))
        data_2d = np.array(row).reshape(28,28)

            
        output_path_folder = os.path.join(base_path, "outputs","train")

        if not os.path.exists(output_path_folder):
            os.makedirs(output_path_folder)
        #image = Image.fromarray(data_2d.astype(np.uint8))
        output_path = os.path.join(output_path_folder, f"#{title}_#{i}_#{index}.csv")
        #image.save(output_path)
            
        with open(output_path, "w", newline='') as csv_file:
            writer = csv.writer(csv_file)
            for row in data_2d:
                writer.writerow(row)
        # row_idx = (i%(subrow*subcol)) // subrow  
        # col_idx = (i%(subrow*subcol)) % subcol   
        
        # axes[row_idx, col_idx].imshow(data_2d, cmap="gray")
        # axes[row_idx, col_idx].axis('off')
        # axes[row_idx, col_idx].set_title(title)
    # plt.tight_layout()
    # plt.show()