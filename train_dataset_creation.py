from pathlib import Path
import os
import re
from PIL import Image

file_name = os.path.basename('\dataset\train\mild\mild_al1.jpg')
x = re.search(r"[1-9]",file_name)
label = file_name[:x.span()[0]]
train_dataset = {}
train_dataset[label]=img
