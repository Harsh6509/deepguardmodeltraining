import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from models.hybrid_model import HybridDetector
from tqdm import tqdm
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
#ye step kaafi imp h during model training , for faster training and to avoid memory issues , hame yaha gpu hi use karna h (if avaialabe)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
#yaha image ko transform karna h , jisse ki model ko input dene se pehle images ko resize aur tensor me convert kar sake, yaha pe maine 128x128 ka size set kiya hai to tum bhi 128x128 hi use karo,
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# yaha pe datasey ko load karna h (kyu ki hamne test_train_split use kiya h apne model me to hame yaha pe train aur test dono ko load karna h), ,
train_dataset = datasets.ImageFolder("dataset/train", transform=transform)
test_dataset = datasets.ImageFolder("dataset/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = HybridDetector().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

    # ye test image ki accuracy check karne ke liye h, har epoch ke baad ham test data pe model ko evaluate karenge, jisse hame pata chalega ki model kitna acha perform kar raha hai unseen data pe, aur agar accuracy me improvement nahi aa raha hai to ham training process ko adjust kar sakte hai (jaise ki learning rate ko change karna ya epochs badhana)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# model save karne ke liye , remember to use correct model name
torch.save(model.state_dict(), "deepguard_model.pth")
print("Model saved!")