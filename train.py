import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torchvision.transforms import functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import argparse
import json

parser = argparse.ArgumentParser(description="Parser of training script")

parser.add_argument('data_dir', help='Data dir', type=str)
parser.add_argument('--save_dir', help='Provide saving directory. Optional argument', type=str)
parser.add_argument ('--lr', help='Learning rate, type=float)
parser.add_argument('--arch', help='CNN Model Architecture', type=str)
parser.add_argument('--hidden_units', help='Hidden units in Classifier', type=int)
parser.add_argument('--epochs', help='Number of epochs', type=int)
parser.add_argument('--gpu', help="Option to use GPU", type = str)

args = parser.parse_args()

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

if args.gpu == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'
    
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# Validation Transformation
validation_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])

# Testing Transformation
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def load_model(arch, hidden_units):
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad = False
            
        if hidden_units:
            classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(25088, 4096)),
                        ('relu1', nn.ReLU()),
                        ('dropout1', nn.Dropout(p=0.3)),
                        ('fc2', nn.Linear(4096, hidden_units)),
                        ('relu2', nn.ReLU()),
                        ('dropout2', nn.Dropout(p=0.3)),
                        ('fc3', nn.Linear(hidden_units, 102)),
                        ('output', nn.LogSoftmax(dim=1))]))
        else:
            classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(25088, 4096)),
                        ('relu1', nn.ReLU()),
                        ('dropout1', nn.Dropout(p=0.3)),
                        ('fc2', nn.Linear(4096, 2048)),
                        ('relu2', nn.ReLU()),
                        ('dropout2', nn.Dropout(p=0.3)),
                        ('fc3', nn.Linear (2048, 102)),
                        ('output', nn.LogSoftmax(dim=1))]))
    else:
        arch = 'alexnet'
        model = models.alexnet(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad = False
            
        if hidden_units: #in case hidden_units were given
            classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear (9216, 4096)),
                        ('relu1', nn.ReLU ()),
                        ('dropout1', nn.Dropout (p = 0.3)),
                        ('fc2', nn.Linear (4096, hidden_units)),
                        ('relu2', nn.ReLU ()),
                        ('dropout2', nn.Dropout (p = 0.3)),
                        ('fc3', nn.Linear (hidden_units, 102)),
                        ('output', nn.LogSoftmax (dim =1))]))
        else:
            classifier = nn.Sequential(OrderedDict ([
                        ('fc1', nn.Linear (9216, 4096)),
                        ('relu1', nn.ReLU ()),
                        ('dropout1', nn.Dropout (p = 0.3)),
                        ('fc2', nn.Linear (4096, 2048)),
                        ('relu2', nn.ReLU ()),
                        ('dropout2', nn.Dropout (p = 0.3)),
                        ('fc3', nn.Linear (2048, 102)),
                        ('output', nn.LogSoftmax (dim =1))]))
    model.classifier = classifier
    return model, arch

def validation(model, validloader, criterion):
    model.to(device)

    valid_loss = 0
    accuracy = 0
    for images, labels in validloader:

        images, labels = images.to(device), labels.to(device)
                
        logps = model(images)
        loss = criterion(logps, labels)
        valid_loss += loss.item()

        ps = torch.exp(logps)
        top_ps, top_class = ps.topk(1, dim=1)
        equality = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
        

    return valid_loss, accuracy

#loading model using above defined functiion
model, arch=load_model(args.arch, args.hidden_units)

#Actual training of the model
#initializing criterion and optimizer
criterion = nn.NLLLoss()

if args.lr:
    optimizer = optim.Adam(model.classifier.parameters(), lr = args.lr)
else:
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)

if args.epochs:
    epochs = args.epochs
else:
    epochs = 5

print_every = 5
steps = 0
running_loss = 0
model.to(device)

for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        steps += 1
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            model.eval () #switching to evaluation mode so that dropout is turned off
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                valid_loss, accuracy = validation(model, validloader, criterion)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Valid Accuracy: {:.3f}%".format(accuracy/len(validloader)*100))

                running_loss = 0
                # Make sure training is back on
                model.train()


model.to('cpu')

model.class_to_idx = train_data.class_to_idx

checkpoint = {
    'classifier': model.classifier,
    'state_dict': model.state_dict(),
    'arch': arch,
    'class_to_idx': model.class_to_idx
}

if args.save_dir:
    torch.save(checkpoint, args.save_dir + '/checkpoint.pth')
else:
    torch.save(checkpoint, 'checkpoint.pth')