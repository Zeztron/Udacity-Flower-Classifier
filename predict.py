import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
from torchvision.transforms import functional as F
import torch.utils.data
import pandas as pd
from collections import OrderedDict
from PIL import Image
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument('image_dir', help='Provide path to image directory.', type=str)
parser.add_argument('load_dir', help = 'Provide path to checkpoint', type = str)
parser.add_argument('--top_k', help = 'Choose tp K', type=int)
parser.add_argument('--category_names', help='Mapping of categories to real names.', type=str)
parser.add_argument('--GPU', help = "Option to use GPU.", type=str)

def load_model(file_path):
    checkpoint = torch.load(file_path)
    if checkpoint ['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
    else: #vgg13 as only 2 options available
        model = models.vgg13(pretrained=True)
    model.classifier = checkpoint ['classifier']
    model.load_state_dict (checkpoint ['state_dict'])
    model.class_to_idx = checkpoint ['class_to_idx']

    for param in model.parameters():
        param.requires_grad = False #turning off tuning of the model

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = F.resize(image, 256)
    image = image.crop((int(image.width / 2) - 112,
                        int(image.height / 2) - 112,
                        int(image.width / 2) + 112,
                        int(image.height / 2) + 112))
    # Color channels of images are typically encoded as integers 0-255, 
    # but the model expected floats 0-1. You'll need to convert the values. 
    # It's easiest with a Numpy array, which you can get from a PIL image like so np_image = np.array(pil_image).
    
    np_image = np.array(image)
    np_image = np_image.transpose((2,0,1))
    np_image = np_image / 255.0
    np_image[0] = (np_image[0] - 0.485)/0.229
    np_image[1] = (np_image[1] - 0.456)/0.224
    np_image[2] = (np_image[2] - 0.406)/0.225
    
    return np_image

#defining prediction function
def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    img = Image.open(image_path)
    image = process_image(img)
    image = image.reshape(1,3,224,224)
    image_tensor = torch.from_numpy(image)
    
    logps = model.forward(image_tensor.to(device).float())

    # Calculate accuracy
    ps = torch.exp(logps)
    ps_topk = ps.topk(topk)

    topk_values = ps_topk[0][0].cpu().detach().numpy()
    topk_indexes = ps_topk[1][0].cpu().numpy()

    return topk_values, topk_indexes

#setting values data loading
args = parser.parse_args ()
image_path = args.image_dir

#defining device: either cuda or cpu
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'

#loading JSON file if provided, else load default file name
if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass

model = load_model(args.load_dir)

if args.top_k:
    number_of_classes = args.top_k
else:
    number_of_classes = 1

#calculating probabilities and classes
probs, classes = predict(image_path, model, number_of_classes)

#preparing class_names using mapping with cat_to_name
class_names = [cat_to_name[item] for item in classes]

for i in range (number_of_classes):
     print("Number: {}/{}.. ".format(l+1, number_of_classes),
            "Class name: {}.. ".format(class_names[i]),
            "Probability: {:.3f}..% ".format(probs[i]*100))