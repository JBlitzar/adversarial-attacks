import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
import torchvision.models as models
import os
from torch.optim import Adam
from torchvision.transforms import v2
import tqdm
from fgsm import demo_fgsm
from PIL import Image
import glob

device = "mps" if torch.backends.mps.is_available() else "cpu"

        
class TinyImageNetDataset(Dataset):
    def __init__(self, root, transform=None, split="train"):
        root = os.path.expanduser(root)
        self.root_dir = root
        self.transform = transform
        
        
        self.subfolders = glob.glob(os.path.join(os.path.join(root, split), '*'))
        self.all_files = glob.glob(os.path.join(os.path.join(root, split), '*/images/*.JPEG'))

        with open(os.path.join(root, "wnids.txt"), "r") as f:
            wnids = f.read().split("\n")
            self.wnids = {}
            with open(os.path.join(root, "clsloc.txt"), "r") as clsloc:
                clsloc_dict = dict(
                    [
                        [
                            a.split(" ")[0],
                            a.split(" ")[1:3]
                        ] 
                        for a in clsloc.read().split('\n')
                    ]
                    )
                for idx, id in enumerate(wnids):
                    try:
                        self.wnids[id] = [torch.tensor(int(clsloc_dict[id][0])).type(torch.IntTensor), clsloc_dict[id][1]]
                    except (KeyError, IndexError):
                        self.wnids[id] = [torch.tensor(0).type(torch.IntTensor), "NOT_FOUND"]
        
        self.total = len(self.all_files)

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        img_path = self.all_files[idx]
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        label = img_path.split("/")[-1].split("_")[0]

        return image, self.wnids[label][0], self.wnids[label][1]

net = models.efficientnet_b7(pretrained=True)#models.inception_v3(pretrained=True)
net.to(device)
transform = v2.Compose([
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    v2.Resize((224,224))
    #v2.Normalize((0.5),(0.5))
]
)
trainset = TinyImageNetDataset(root=os.path.expanduser("~/torch_datasets/imagenet"),transform=transform)
dataloader = DataLoader(trainset,batch_size=64,shuffle=True)
criterion = nn.CrossEntropyLoss()


net.to("cpu")
demo_fgsm(dataloader, net, criterion)