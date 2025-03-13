import os
import matplotlib.pyplot as plt
import csv
import numpy as np

os.chdir("C:\\Users\\김해창\\Desktop\\VAR")
test_path = "mnist_test.csv"

with open(test_path,"r") as csv_file:
    reader = csv.reader(csv_file)
    rows = list(reader)
firstrow = rows[0]
row = list(map(int,firstrow[1:]))

output_path = f"{firstrow[0]}-{firstrow[-1]}.csv"
data_2d = np.array(row).reshape(28, 28)
with open(output_path, "w", newline='') as csv_file:
    writer = csv.writer(csv_file)
    for row in data_2d:
        writer.writerow(row)