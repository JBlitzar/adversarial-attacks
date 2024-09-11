import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import os
from torch.optim import Adam
from torchvision.transforms import v2
import tqdm
class MNISTCNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.block = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(16,16,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16*28*28,10),
            nn.Sigmoid()

        )

    def forward(self,x):
        return self.block(x)
    

net = MNISTCNN()
transform = v2.Compose(
    v2.ToTensor(),
    v2.Normalize((0.5),(0.5))
)
trainset = MNIST(root=os.path.expanduser("~/torch_datasets/MNIST"),train=True,transform=transform)
dataloader = DataLoader(trainset,batch_size=64)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(net.parameters)

epochs = 10
for epoch in  tqdm.tqdm(epochs):
    for batch, target in dataloader:
        
        optimizer.zero_grad()
        logits = net(batch)

        loss = criterion(logits, target)

        loss.backward()
        optimizer.step()



def FGSM(image, label, net, epsilon=0.1):
    net.test()
    image.requires_grad = True

    logits = net(image)
    loss = criterion(logits, label)

    loss.backward()

    return image + epsilon * image.grad.sign()


    