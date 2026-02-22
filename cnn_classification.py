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


# ====== Load dataset =======
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = Data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = Data.DataLoader(testset, batch_size=32, shuffle=False)


# ====== Define a CNN model ======
class CNN_MODEL(nn.module):
    def __init__(self):
        super(CNN_MODEL, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


# ====== Device loading ======
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_MODEL().to(device)


# ====== Set loss function ======
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 20
history = {'loss': [], 'train_acc': [], 'test_acc': []}

for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        corrct += predicted.eq(labels).sum().item()
        
    # ------ Calculate testset accuracy ------
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    # ------ Store history ------
    history['loss'].append(running_loss / len(trainloader))
    history['train_acc'].append(100. * correct / total)
    history['test_acc'].append(100. * test_correct / test_total)
    
    print(f"Epoch {epoch+1}/{epochs} | Loss: {history['loss'][-1]:.4f} | "
          f"Train acc: {history['train_acc'][-1]:.2f}% | Test acc: {history['test_acc'][-1]:.2f}%")


# ====== Plotting learning curves ======
plt.figure(figsize=(12, 5))

# ------ Plot loss curve ------
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Training loss', color='blue')
plt.title("Training loss over Epochs")
plt.xlabel("Epochs")
plt.legend()

# ------ Plot accuracy curve ------
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train accuracy', color='green')
plt.plot(history['test_acc'], label='Test accuracy', color='orange')
plt.title("Acuracy over Epochs")
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()