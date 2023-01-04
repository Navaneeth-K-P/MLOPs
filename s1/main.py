import argparse
import sys

import torch
import click

from torchvision import transforms
from torch import nn, optim
from tqdm import tqdm

from data import mnist
from model import MyAwesomeModel
from torch.utils.data import TensorDataset, DataLoader


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
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
    torch.save(model.state_dict(), 'trained_model.pt')




@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)
    _, test_set = mnist()

    testloader = DataLoader(test_set, batch_size=64, shuffle=True)

    with torch.no_grad():
        flag = 0
        model.eval()
        epocs = 10
        for i in range(0,epocs):
            for images, labels in tqdm(testloader):
                ps = torch.exp(model(images))
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                if flag == 0:
                    equal_list = equals
                    flag = 1
                else:
                    equal_list = torch.cat((equal_list,equals))
    accuracy = torch.mean(equal_list.type(torch.FloatTensor))
    print(f'Accuracy: {accuracy.item()*100}%')



cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    