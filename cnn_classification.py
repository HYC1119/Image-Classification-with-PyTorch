import os
import time
import numpy as np
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
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = Data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
testloader = Data.DataLoader(testset, batch_size=32, shuffle=False)


# ====== Define a CNN model ======
class CNN_MODEL(nn.Module):
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
        
        # ------ Output Fully connected layer independently ------
        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
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


# ====== Define class name ======
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def show_random_predictions(model, testloader, device):
    print("\nProducing predicted image...")
    # change to evaluate mode
    model.eval()
    
    # Get image and label in a batch
    detaiter = iter(testloader)
    images, labels = next(detaiter)
    
    # Put into GPU or CPU to predict
    images_device, labels_device = images.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(images_device)
        _, predicted = torch.max(outputs, 1)
        
    # Move predicted result to CPU
    predicted = predicted.cpu()
    
    # Prepare to plot 
    fig = plt.figure(figsize=(15, 6))
    
    # Only shows first 10 pictures 
    for i in range(10):
        ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
        
        img = images[i] / 2 + 0.5
        npimg = img.numpy()
        
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        
        pred_class = classes[predicted[i]]
        true_class = classes[labels[i]]
        color = 'green' if predicted[i] == labels[i] else 'red'
        
        ax.set_title(f"Pred: {pred_class}\nTrue: {true_class}", color=color)
    
    plt.tight_layout()
    plt.show()


# ====== Set loss function ======
epochs = 20
best_acc = 0.0
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
history = {'loss': [], 'train_acc': [], 'test_acc': []}
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

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
        correct += predicted.eq(labels).sum().item()
        
    # ------ Calculate testset accuracy ------
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            # Test overall accuracy
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            # Calculate accuracy of each class
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # ------ Store history ------
    history['loss'].append(running_loss / len(trainloader))
    history['train_acc'].append(100. * correct / total)
    history['test_acc'].append(100. * test_correct / test_total)
    
    current_acc = 100. * test_correct / test_total
    if current_acc > best_acc:
        best_acc = current_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Find better model! (Accuracy: {best_acc:.2f}%)")
    
    print(f"Epoch {epoch+1}/{epochs} | Loss: {history['loss'][-1]:.4f} | "
          f"Train acc: {history['train_acc'][-1]:.2f}% | Test acc: {history['test_acc'][-1]:.2f}%")
    
    scheduler.step()
    
    if (epoch + 1) % 5 == 0:
        show_random_predictions(model, testloader, device)

# ====== Finish training, print each calss' value ======
print("\n ====== Accuracy of each class ======")
for i in range(10):
    print(f"{classes[i]:>5s} : {100 * class_correct[i] / class_total[i]:.1f}%")    
    
show_random_predictions(model, testloader, device)


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

  