import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch.autograd import Variable


# ====== Store trained data ======
path = 'output.txt'
f = open(path, 'w')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")
print()


# ====== Set Hyperparameters ======
DOWNLOAD_DATASET = True
LR = 0.001
BATCH_SIZE = 10
MODELS_PATH = './models'


# ======  Preprocessing: Transform datasets to tensors of nornmalized range [-1, 1] ======
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5),
                          (0.5, 0.5, 0.5))])


# ====== Load CIFAR10 Dataset ======
trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

trainloader = Data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

testloader = Data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

classes = ('plane', 'car', 'bird', 'cat',
           'deer',  'dog', 'frog', 'ship', 'truck')


# ====== Create Model (VGG19) ======
vgg19 = torchvision.models.vgg19_bn(num_classes=10)
vgg19.to(device)


# ====== Loss function & Optimimzer ======
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg19.parameters(), lr=LR, momentum=0.9)


# ====== Training part ======
# ------ Set all data to 0 & Reset time ------
print("Strat training...")
start = time.time()
training_acc = []
training_loss = []
training_acc = []
validation_acc = []
validation_loss = []


# ------ Start traing loop ------
for epoch in range(epochs):
    print(f'-------------------- epoch {epoch+1} --------------------')
    vgg19.train()

    # ------ Reset when a new epoch ------
    total = 0
    total_val = 0
    running_loss = 0.0
    train_loss = 0.0
    train_correct = 0.0
    val_correct = 0
    val_loss = 0.0

# ------ Calculate every batch ------
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        # Set gradient to 0
        optimizer.zero_grad()

        # Forward + backward + optimize
        ouputs = vgg19(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate batch loss
        running_loss += loss.item()
    
        # Calculate epoch loss
        train_loss += outputs.shape[0] * loss.item()

        # Calculate epoch accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        # Output training process & Reset batch loss every 50 batch
        if i % 50 == 49:
            print(
                f'[{epoch + 1}/{epochs}, {i + 1:3d}] loss: {running_loss / 50:.3f} time:{(time.time()-start)/60:.2f}')
            running_loss = 0.0

print("Finish training!")













