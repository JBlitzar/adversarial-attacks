import matplotlib.pyplot as plt
import torch
import numpy as np
import tqdm

device = "mps" if torch.backends.mps.is_available() else "cpu"


def FGSM(image, label, net, criterion, epsilon=0.1, adaptive=False):
    correct = True
    epsilon_a = 0 if adaptive else epsilon

    while correct:
        net.eval()
        
        image.requires_grad = True

        logits = net(image)
        #logits = logits.long()
        label = label.long()
        loss = criterion(logits, label)

        loss.backward()

        result = image + epsilon_a * image.grad.sign()
        result = result.clamp(0,1)

        if not adaptive:
            return result
        
        correct = torch.argmax(net(result), dim=1).item() == label.item()
        epsilon_a += 0.001
    
    return result

def fgsm_acc_over_epsilon(dataloader, net, criterion, eps_step=0.05, eps_max=0.5):
    def get_fgsm_acc(epsilon, dataloader, net, criterion):
        net.eval()
        running_acc_sum = 0
        running_acc_amt = 0
        for batch, labels in tqdm.tqdm(dataloader, leave=False):
            batch = batch.to(device)
            labels = labels.to(device)

            adversarial_batch = FGSM(batch, labels, net, criterion, epsilon=epsilon, adaptive=False)
            with torch.no_grad():
                adversarial_output = net(adversarial_batch)

            adversarial_pred = torch.argmax(adversarial_output, dim=1) # todo: double check
            #print(adversarial_pred.size())

            running_acc_sum += torch.sum(adversarial_pred == labels).item()
            running_acc_amt += adversarial_pred.size(0)

        return running_acc_sum / running_acc_amt
    
    accuracies = []
    eps_range = np.linspace(0,eps_max, int(eps_max // eps_step))
    for eps in tqdm.tqdm(eps_range):

        fgsm_acc = get_fgsm_acc(eps, dataloader, net, criterion)
        accuracies.append(fgsm_acc)


    plt.plot(eps_range, accuracies)
    plt.show()



def demo_fgsm_over_eps(dataloader, net, criterion, eps_step=0.05,eps_max=0.5):
    
    def get_images(dataloader):
        try:
            for images, labels, human_labels in dataloader:
                image = images
                label = labels
                human_label = human_labels
                break
            return image[0].unsqueeze(0).to(device), label[0].unsqueeze(0).to(device), human_label[0]
        except:
            for images, labels in dataloader:
                image = images
                label = labels
                break
            return image[0].unsqueeze(0).to(device), label[0].unsqueeze(0).to(device), ""


    def plot_images(images, labels):
        fig, axes = plt.subplots(1, len(images), figsize=(20,20))
        def process_img(img):
            return img.squeeze().detach().cpu().numpy()

        for ax, image, label in zip(axes,images, labels):
            
            ax.imshow(process_img(image), cmap='gray')
            ax.set_title(f'{label}')
            ax.axis('off')


        plt.tight_layout()
        plt.show()

    net.eval()
    eps_range = np.linspace(0,eps_max, int(eps_max // eps_step))
    images_to_plot = []
    labels_to_plot = []
    for epsilon in eps_range:
        orig_image,orig_label,human_label = get_images(dataloader)

        with torch.no_grad():
            original_output = net(orig_image)
        original_pred = torch.argmax(original_output, dim=1).item()

        adversarial_image = FGSM(orig_image, orig_label, net, criterion, epsilon=epsilon, adaptive=False)


        with torch.no_grad():
            adversarial_output = net(adversarial_image)
        adversarial_pred = torch.argmax(adversarial_output, dim=1).item()


        labels_to_plot.append(f"{epsilon}: {original_pred} -> {adversarial_pred}")
        images_to_plot.append(adversarial_image)


    plot_images(images_to_plot, labels_to_plot)



def demo_fgsm(dataloader, net, criterion):
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

    adversarial_image = FGSM(orig_image, orig_label, net, criterion)


    with torch.no_grad():
        adversarial_output = net(adversarial_image)
    adversarial_pred = torch.argmax(adversarial_output, dim=1).item()

    plot_images(orig_image.detach().cpu().numpy(), original_pred, adversarial_image.detach().cpu().numpy(), adversarial_pred, orig_label, human_label)