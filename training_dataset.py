from pathlib import Path
import os
from datasets import load_dataset

DATASET_PATH = 'dataset//train' # the dataset file or root folder path.

# Image Parameters
n_classes = 4 # CHANGE HERE, total number of classes
img_height = 224# CHANGE HERE, the image height to be resized to
img_width = 224 # CHANGE HERE, the image width to be resized to
channels = 3 # The 3 color channels, change to 1 if grayscale
batch_size= 130

def read_image():
    dataset = load_dataset("imagefolder", data_dir=DATASET_PATH,drop_labels=False)             
    def transforms(examples):
        examples["pixel_values"] = [image.convert("RGB").resize((224,224)) for image in examples["image"]]
        return examples
    dataset = dataset.map(transforms, remove_columns=["image"], batched=True)
    labels = dataset["train"].features['label'].names
    num_classes = len(labels)
    return dataset,labels,num_classes
