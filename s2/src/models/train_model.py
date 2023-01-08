import torch
import click

from torchvision import transforms
from torch import nn, optim
from tqdm import tqdm

from data import mnist
from model import MyAwesomeModel
from torch.utils.data import TensorDataset, DataLoader

model = MyAwesomeModel()
train_set, _  = mnist()

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

trainloader = DataLoader(train_set, batch_size=64, shuffle=True)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

epochs = 25

for e in range(epochs):
    running_loss = 0
    model.train()
    for images, labels in tqdm(trainloader):
        # print(images.dtype)
        optimizer.zero_grad()
        
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
            
        running_loss += loss.item()
    print(f'Loss: {running_loss}')
torch.save(model.state_dict(), r'E:\DTU\MLOPs\dtu_mlops\MLOPs\s2\models\trained_model.pt')