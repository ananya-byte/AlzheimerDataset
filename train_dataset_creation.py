from pathlib import Path
import os
import re
from PIL import Image

file_name = os.path.basename('\dataset\train\mild\mild_al1.jpg')# opening file name
x = re.search(r"[1-9]",file_name)# label name 
label = file_name[:x.span()[0]]
train_dataset = {}
train_dataset[label]='\dataset\train\mild\mild_al1.jpg'
print(train_dataset)# label name: path

our_file = Path(os.path.abspath('dataset\\train\mild\mild_al1.jpg'))# using relative to get absolute path
img = Image.open(our_file)
train_dataset[label]=img
print(f"Original size : {img.size}") # 5464x3640
img_resized = img.resize((32, 32))
train_dataset[label]=img_resized
img.close()
print(train_dataset)


