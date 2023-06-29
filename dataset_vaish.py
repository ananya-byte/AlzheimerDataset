from pathlib import Path
import os
import re
from PIL import Image

test_dataset = []
labels=['mild_al','moderate_al','verymild_al','nondemented_al']
n_labels=len(labels)
n_samples = {}

def  create_testing_dataset():
    mild_folder=os.path.abspath('dataset\\test\mild')
    mil_folder = Path(mild_folder)# using relative to get absolute path
    mild_counter=0
    for file in mil_folder.iterdir():
        file_name = os.path.basename(file)
        label = file_name[:re.search(r"[1-9]",file_name).span()[0]]
        index=labels.index(label)
        img = Image.open(file)
        img_resized = img.resize((224, 224))
        img_dict = {}
        img_dict['img']=img_resized
        img_dict['label']=index
        test_dataset.append(img_dict)
        img.close()
        mild_counter=mild_counter+1

        

    moderate_folder=os.path.abspath('dataset\\test\moderate')
    mod_folder = Path(moderate_folder)# using relative to get absolute path
    moderate_counter=0
    for file in mod_folder.iterdir():
        file_name = os.path.basename(file)
        label = file_name[:re.search(r"[1-9]",file_name).span()[0]]
        index=labels.index(label)
        img = Image.open(file)
        img_resized = img.resize((224, 224))
        img_dict = {}
        img_dict['img']=img_resized
        img_dict['label']=index
        test_dataset.append(img_dict)
        img.close()
        moderate_counter=moderate_counter+1


    nondemented_folder=os.path.abspath('dataset\\test\\nondemented')
    non_folder = Path(nondemented_folder)# using relative to get absolute path
    nondemented_counter=0
    for file in non_folder.iterdir():
        file_name = os.path.basename(file)
        label = file_name[:re.search(r"[1-9]",file_name).span()[0]]
        index=labels.index(label)
        img = Image.open(file)
        img_resized = img.resize((224, 224))
        img_dict = {}
        img_dict['img']=img_resized
        img_dict['label']=index
        test_dataset.append(img_dict)
        img.close()
        nondemented_counter=nondemented_counter+1


    verymild_folder=os.path.abspath('dataset\\test\\verymild')
    vm_folder = Path(verymild_folder)# using relative to get absolute path
    verymild_counter=0
    for file in vm_folder.iterdir():
        file_name = os.path.basename(file)
        label = file_name[:re.search(r"[1-9]",file_name).span()[0]]
        index=labels.index(label)
        img = Image.open(file)
        img_resized = img.resize((224, 224))
        img_dict = {}
        img_dict['img']=img_resized
        img_dict['label']=index
        test_dataset.append(img_dict)
        img.close()
        verymild_counter=verymild_counter+1
        

    n_samples = {0:mild_counter,1:moderate_counter,2:nondemented_counter,3:verymild_counter}
    print(n_samples)
    return test_dataset,labels,n_labels,n_samples
create_testing_dataset()
