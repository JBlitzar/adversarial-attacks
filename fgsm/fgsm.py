import matplotlib.pyplot as plt
import torch
import numpy as np

def FGSM(image, label, net, criterion, epsilon=0.2):
    net.eval()
    image.requires_grad = True

    logits = net(image)
    #logits = logits.long()
    label = label.long()
    loss = criterion(logits, label)

    loss.backward()

    return image + epsilon * image.grad.sign()

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