from pathlib import Path
import os
import re
from PIL import Image

train_dataset = []


mild_folder=os.path.abspath('dataset\\train\mild')
our_folder = Path(mild_folder)# using relative to get absolute path
for file in our_folder.iterdir():
    file_name = os.path.basename(file)
    label = file_name[:re.search(r"[1-9]",file_name).span()[0]]
    img = Image.open(file)
    img_resized = img.resize((32, 32))
    img_dict = {}
    img_dict['img']=img_resized
    img_dict['label']=label
    train_dataset.append(img_dict)
    img.close()
print(train_dataset[0]['label'])


