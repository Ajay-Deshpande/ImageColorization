from torch import nn
import torch

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(EncoderBlock, self).__init__()

        self.conv_1 = nn.Conv2d(in_c, out_c, kernel_size = (3, 3), stride = 1, padding = 1, bias = False)
        self.conv_2 = nn.Conv2d(out_c, out_c, (3, 3), 1, 1, bias = False)

    def forward(self, x):
        x = self.conv_1(x)
        x = nn.functional.relu(x)
        x = self.conv_2(x)
        x = nn.functional.relu(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(DecoderBlock, self).__init__()

        self.conv_T = nn.ConvTranspose2d(in_c, out_c, kernel_size = (2, 2), stride = 2, bias = False)
        self.conv_1 = nn.Conv2d(in_c, out_c, kernel_size = (3, 3), stride = 1, padding = 1, bias = False)
        self.conv_2 = nn.Conv2d(out_c, out_c, (3, 3), 1, 1, bias = False)

    def forward(self, x, skip_connection_output):
        x = self.conv_T(x)
        x = torch.cat([x, skip_connection_output], dim = 1)
        x = self.conv_1(x)
        x = nn.functional.relu(x)
        x = self.conv_2(x)
        x = nn.functional.relu(x)
        return x

class UNetGenerator(nn.Module):
    def __init__(self, input_channels, out_channels, num_encoder_blocks):
        super(UNetGenerator, self).__init__()

        self.encoder_blocks = []
        in_c = 64
        encoder1 = EncoderBlock(input_channels, in_c)
        self.encoder_blocks.append(encoder1)

        for _ in range(num_encoder_blocks - 1):
            self.encoder_blocks.append(EncoderBlock(in_c, in_c * 2))
            in_c *= 2

        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)

        self.mid_layer = EncoderBlock(in_c, in_c * 2)
        in_c *= 2
        self.decoder_blocks = []
        for _ in range(num_encoder_blocks):
            self.decoder_blocks.append(DecoderBlock(in_c, in_c // 2))
            in_c //= 2

        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)

        self.out = nn.Conv2d(in_c, out_channels, kernel_size = (3, 3), stride = 1, padding = 1)

    def forward(self, x):
        self.encoder_outputs = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            self.encoder_outputs.append(x)
            x = nn.MaxPool2d(kernel_size = (2,2), stride = 2)(x)

        self.encoder_outputs.reverse()
        x = self.mid_layer(x)

        for decoder_block, enc_output in zip(self.decoder_blocks, self.encoder_outputs):
            x = decoder_block(x, enc_output)

        x = self.out(x)
        return x

class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels, num_filters = 64, num_disc_blocks = 3):
        super(PatchDiscriminator, self).__init__()
        self.conv_1 = nn.Conv2d(input_channels, num_filters, kernel_size = (4, 4), stride = 2, padding = 1, bias = False)
        self.leaky_relu = nn.LeakyReLU(.2)
        self.layers = [self.conv_1, self.leaky_relu]
        for i in range(num_disc_blocks):
            self.layers.append(nn.Conv2d(num_filters * (2 ** i), num_filters * (2 ** (i + 1)), kernel_size = (4, 4), stride = 2, padding = 1, bias = False))
            self.layers.append(nn.BatchNorm2d(num_filters * (2 ** (i + 1))))
            self.layers.append(self.leaky_relu)
        self.layers.append(nn.Conv2d(num_filters * (2 ** (num_disc_blocks)), 1, kernel_size = (4, 4), stride = 1, padding = 1, bias = False))
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)

class GANLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.loss = nn.MSELoss()
        
    def __call__(self, preds, target_is_real):
        labels = torch.as_tensor(target_is_real, dtype = torch.float32)
        loss = self.loss(preds, labels)
        loss.requires_grad = True
        return loss