from matplotlib import pyplot as plt
import torch
import numpy as np
from utils.data_loaders import ColorizationDataset
from torch.utils.data import DataLoader
from models.model_architecture import ImageColorizationModel
from skimage import color
from glob import glob
from tqdm import tqdm

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count

def create_loss_meters():
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()

    return {'disc_GAN_loss': loss_D,
            'gen_GAN_loss': loss_G_GAN,
            'gen_L1_loss': loss_G_L1,
            'gen_loss': loss_G}

def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)

def lab_to_rgb(images):
    images[:, [0], :, :] = (images[:, [0], :, :] + 1.) * 50.
    images[:, [1, 2], :, :] = images[:, [1, 2], :, :] * 110.

    for img_ind in range(images.shape[0]):
        images[img_ind, :, :, :] = torch.tensor(color.lab2rgb(images[img_ind, :, :, :].permute(1, 2, 0))).permute(2, 0, 1)

    return images

def visualize(model, data, save = True, save_dir = None, epoch = None):
    model.generator.eval()
    with torch.no_grad():
        model.forward(data)
    model.generator.train()
    fake_images = model.gen_images.to('cpu')

    fake_imgs = lab_to_rgb(fake_images)
    real_imgs = lab_to_rgb(data)

    fig = plt.figure(figsize=(15, 8))
    L = real_imgs[:, [0], :, :]

    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i].permute(1, 2, 0))
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i].permute(1, 2, 0))
        ax.axis("off")

    # plt.show()

    if save:
        fig.savefig(f"{save_dir}/colorization_after_epoch_{epoch}.png")

def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")

def train_model(model, train_data_loader, epochs, model_checkpoint_after = 10, model_save_dir = None, prev_epochs = 0):
    for epoch in range(1, epochs + 1):
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to
        for data in tqdm(train_data_loader):
            model.optimize(data)
            update_losses(model, loss_meter_dict, count=data.size(0)) # function updating the log objects
        print(f"\nEpoch {epoch+1}/{epochs}")
        log_results(loss_meter_dict) # function to print out the losses
        visualize(model, data, save = True, save_dir = './outputs', epoch = prev_epochs + epoch) # function displaying the model's outputs
        # if epoch % model_checkpoint_after:
        torch.save(model.state_dict(), model_save_dir + f'_epoch_{prev_epochs + epoch}.pth')
    return

def main(IMAGE_PATHS, BATCH_SIZE, train_additional = True):
    dataset_split_name = 'train'
    images_path = np.random.choice(glob(IMAGE_PATHS.format(dataset_split_name)), 12000).tolist()
    train_dataset = ColorizationDataset(images_path, dataset_split_name)
    dataset_split_name = 'validation'
    images_path = glob(IMAGE_PATHS.format(dataset_split_name))
    val_dataset   = ColorizationDataset(images_path, dataset_split_name)
    
    train_data_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, pin_memory = True)
    val_data_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True, pin_memory = True)

    model = ImageColorizationModel()
    prev_epochs = 0
    model_list = sorted(glob('./models/*.pth'), key = lambda x: int(x.strip(".pth").split("_")[-1]))
    if train_additional and model_list:
        print(model_list[-1])
        prev_epochs = int(model_list[-1].split('_')[-1].split('.')[0])
        model.load_state_dict(torch.load(model_list[-1]))

    train_model(model, train_data_loader, 2, model_save_dir = './models/', prev_epochs = prev_epochs)
    torch.save(model.state_dict(), './models/_epoch_final.pth')


if __name__ == "__main__":
    IMAGE_PATHS = "./data/coco-2017/{}/data/*.jpg"
    BATCH_SIZE = 8
    main(IMAGE_PATHS, BATCH_SIZE)   
