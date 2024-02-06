from utils.nn_helpers import UNetGenerator, PatchDiscriminator, GANLoss
import torch
from torch import nn, optim

def init_model(model, device):
    def init_layer_weights(layer):
        layer_name = layer.__class__.__name__
        if "Conv" in layer_name:
            nn.init.kaiming_normal_(layer.weight.data, a = .2)
        elif "BatchNorm" in layer_name:
            nn.init.normal_(layer.weight.data, 1, .02)

    model.to(device)
    model.apply(init_layer_weights)
    return model
    
class ImageColorizationModel(nn.Module):
    def __init__(self, lr_generator = 2e-4, lr_discriminator = 2e-4, beta_1 = 0.5, beta_2 = 0.999, lambda_L1 = 100.):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.L1_lambda  = lambda_L1

        self.generator = UNetGenerator(1, 2, 4)
        self.discriminator = PatchDiscriminator(3)
        self.generator = init_model(self.generator, self.device)
        self.discriminator = init_model(self.discriminator, self.device)

        self.GAN_loss = GANLoss()
        self.L1_loss = nn.L1Loss()

        self.gen_optim = optim.Adam(self.generator.parameters(), lr = lr_generator, betas = (beta_1, beta_2))
        self.disc_optim = optim.Adam(self.discriminator.parameters(), lr = lr_discriminator, betas = (beta_1, beta_2))

    def set_requires_grad(self, model, set_flag = True):
        for param in model.parameters():
            param.requires_grad = set_flag

    def forward(self, x):
        self.real_images = x
        self.real_images = self.real_images.to(self.device)
        L, ab = x[:, [0], :, :], x[:, [1, 2], :, :]
        gen_images_ab = self.generator(L.to(self.device).detach()).to(self.device)
        self.gen_images = torch.cat([L.to(self.device), gen_images_ab], dim = 1)

    def backward_discriminator(self):
        all_images = torch.cat([self.real_images, self.gen_images], dim = 0).to(self.device)
        preds = self.discriminator(all_images.detach())
        image_real_ = torch.cat([torch.ones(self.real_images.shape[0], *preds.shape[1:]), torch.zeros(self.gen_images.shape[0], *preds.shape[1:])], dim = 0)
        self.disc_GAN_loss = self.GAN_loss(preds.to(self.device).detach(), image_real_.to(self.device).detach())
        self.disc_GAN_loss.backward()

    def backward_generator(self):
        # Images should be considered real by the discriminator
        preds = self.discriminator(self.gen_images)
        image_real_ = torch.ones(preds.shape)
        self.gen_GAN_loss = self.GAN_loss(preds.to(self.device).detach(), image_real_.to(self.device).detach())
        self.gen_L1_loss = self.L1_loss(self.gen_images[:, [1, 2], :, :], self.real_images[:, [1, 2], :, :]) * self.L1_lambda
        self.gen_loss = self.gen_GAN_loss + self.gen_L1_loss
        self.gen_loss.backward()

    def optimize(self, data):
        data.to(self.device)
        self.forward(data)
        self.discriminator.train()
        self.set_requires_grad(self.discriminator, True)
        self.disc_optim.zero_grad()
        self.backward_discriminator()
        self.disc_optim.step()

        self.generator.train()
        self.set_requires_grad(self.discriminator, False)
        self.gen_optim.zero_grad()
        self.backward_generator()
        self.gen_optim.step()