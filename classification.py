import os
import time
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
with open(path, 'w') as f:
    f.write("Training Log\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")
print()


# ====== Set Hyperparameters ======
DOWNLOAD_DATASET = True
LR = 0.001
BATCH_SIZE = 32
EPOCHS = 100


# ======  Preprocessing: Transform datasets to tensors of nornmalized range [-1,1] ======
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])


# ====== Load CIFAR10 Dataset ======
trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Testing
trainset = Data.Subset(trainset, range(100))
trainloader = Data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

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
print("Start training...")
start = time.time()


# ------ Start traing loop ------
for epoch in range(EPOCHS):
    print(f'-------------------- epoch {epoch + 1} --------------------')
    vgg19.train()

    # ------ Reset when a new epoch ------
    running_loss = 0.0
    total_train = 0
    train_correct = 0

# ------ Calculate every batch ------
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Set gradient to 0
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = vgg19(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate batch loss
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        # Output training process & Reset batch loss every 50 batch
        if i % 100 == 99:
            avg_loss = running_loss / 100
            elapsed_time = (time.time() - start) / 60
            print(f'[Epoch {epoch + 1}, Batch {i + 1:3d}] loss: {avg_loss:.3f} | time: {elapsed_time:.2f} min')
            running_loss = 0.0


# ------ Validation part (Verified every epoch) ------
    vgg19.eval()
    val_correct = 0
    total_val = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = vgg19(images)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / total_val
    train_acc = 100 * train_correct / total_train
    print(f"End of Epoch {epoch + 1} | Train Acc: {train_acc: .2f}% | Val Acc: {val_acc: .2f}%")

    with open(path, 'a') as f:
        f.write(f"Epoch {epoch + 1}: Train Acc {train_acc: .2f}%, Val Acc {val_acc: .2f}%\n")

print("Finish training!")


# ====== Model saving ======
MODELS_PATH = './models'
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

torch.save(vgg19.state_dict(), os.path.join(MODELS_PATH, 'vgg10_cifar10.pth'))
print(f"Model saved to {MODELS_PATH}/vgg10_cifar10.pth")
