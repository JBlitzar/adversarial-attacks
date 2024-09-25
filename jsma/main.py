import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import os
from torch.optim import Adam
from torchvision.transforms import v2
import tqdm
import matplotlib.pyplot as plt
from torch.autograd.functional import jacobian

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
            nn.Linear(16*28*28,10)

        )
        self.activation = nn.Sigmoid()

    def forward(self,x):
        x =  self.block(x)
        x = self.activation(x)
        return x
    
    def get_raw_logits(self,x):
        x =  self.block(x) 
        return x


net = MNISTCNN()
net.to(device)
transform = v2.Compose([
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    #v2.Normalize((0.5),(0.5))
]
)
trainset = MNIST(root=os.path.expanduser("~/torch_datasets/MNIST"),train=True,transform=transform)
dataloader = DataLoader(trainset,batch_size=64,shuffle=True)
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
net.eval()


def jsma_attack(model, image, target, theta=1, num_features=28*28,img_wh = 28):
    image = image.clone().detach().requires_grad_(True)
    target = torch.tensor([target])


    output = model(image)
    init_pred = output.max(1, keepdim=True)[1]

    if init_pred.item() == target.item():
        return image, False


    def saliency_map(image):
        logits = model.get_raw_logits(image)
        
        # def make_jacobian(logit):
        #     if image.grad is not None:
        #         image.grad.zero_()
        #     logit.backward(retain_graph=True)

        #     J = image.grad
        #     return J
        
        logits = logits.squeeze(0)
        
        J_all = jacobian(net.get_raw_logits, image).squeeze(0)


        J_target = J_all[target.item()]


        J_other_total = None
        for idx, logit in enumerate(logits):
            if idx != target.item():
                J_other = J_all[idx]
                if J_other_total is None:
                    J_other_total = J_other
                else:
                    J_other_total += J_other


        S = J_target * torch.abs(J_other_total)

        S[torch.logical_or(J_target < 0, J_other_total > 0)] = 0 # most normal boolean mask

        return S

    cur_image = image.detach().clone()
    
    for i in range(num_features):

        cur_image.requires_grad = True
        map = saliency_map(cur_image)

        cur_image.requires_grad = False

        best_index = torch.argmax(map).item()
        print(best_index)
        
        print(cur_image.size())
        cur_image[0][0][best_index // img_wh][best_index % img_wh] += theta 
        
        # best code
        print(torch.min(cur_image))
        print(torch.max(cur_image))
        print("minmax^^")

        cur_image = torch.clamp(cur_image,0,1)

        



        output = model(cur_image)
        final_pred = output.max(1, keepdim=True)[1]

        if final_pred.item() == target.item():

            return cur_image, True
    return image, False



def demo_jsma(dataloader, net): #TODO: prevent code reuse
    def get_images(dataloader):
        try:
            for images, labels, human_labels in dataloader:
                image = images
                label = labels
                human_label = human_labels
                break
            return image[0].unsqueeze(0), label[0].unsqueeze(0), human_label[0]
        except:
            for images, labels in dataloader:
                image = images
                label = labels
                break
            return image[0].unsqueeze(0), label[0].unsqueeze(0), ""


    def plot_images(orig_img, orig_pred, adv_img, adv_pred,real_label, human_label):
        fig, ax = plt.subplots(1, 4, figsize=(10, 5))
        def process_img(img):
            try:
                return img.squeeze().transpose(1, 2, 0)
            except ValueError:
                return img.squeeze()

        ax[0].imshow(process_img(orig_img), cmap='gray')
        ax[0].set_title(f'Original')
        ax[0].axis('off')

        ax[1].text(0.5, 0.5, f'Pred: {orig_pred}; Real: {real_label.item()} {human_label}', fontsize=12, ha='center')
        ax[1].set_title(f'Prediction')
        ax[1].axis('off')

        ax[2].imshow(process_img(adv_img), cmap='gray')
        ax[2].set_title(f'Adversarial')
        ax[2].axis('off')

        ax[3].text(0.5, 0.5, f'Pred: {adv_pred}; Real: {real_label.item()} {human_label}', fontsize=12, ha='center')
        ax[3].set_title(f'Prediction')
        ax[3].axis('off')

        plt.tight_layout()
        plt.show()

    net.eval()
    orig_image,orig_label,human_label = get_images(dataloader)

    with torch.no_grad():
        original_output = net(orig_image)
    original_pred = torch.argmax(original_output, dim=1).item()

    adversarial_image,success = jsma_attack(net,orig_image,(orig_label + 1) % 10)




    with torch.no_grad():
        adversarial_output = net(adversarial_image)
    adversarial_pred = torch.argmax(adversarial_output, dim=1).item()

    plot_images(orig_image.detach().cpu().numpy(), original_pred, adversarial_image.detach().cpu().numpy(), adversarial_pred, orig_label, human_label)

demo_jsma(dataloader, net)