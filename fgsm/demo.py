import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import os
from torch.optim import Adam
from torchvision.transforms import v2
import tqdm
from fgsm import demo_fgsm

device = "mps" if torch.backends.mps.is_available() else "cpu"

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
net.to(device)
transform = v2.Compose([
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    #v2.Normalize((0.5),(0.5))
]
)
trainset = MNIST(root=os.path.expanduser("~/torch_datasets/MNIST"),train=True,transform=transform)
dataloader = DataLoader(trainset,batch_size=64)
criterion = nn.CrossEntropyLoss()
try:
    net.load_state_dict(torch.load("MNIST_demo.pt", weights_only=True))
except:

   
    optimizer = Adam(net.parameters())

    epochs = 10
    for epoch in  tqdm.trange(epochs):
        for batch, target in tqdm.tqdm(dataloader, leave=False):
            batch = batch.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            logits = net(batch)

            loss = criterion(logits, target)

            loss.backward()
            optimizer.step()

    torch.save(net.state_dict(), "MNIST_demo.pt")


net.to("cpu")
demo_fgsm(dataloader, net, criterion)