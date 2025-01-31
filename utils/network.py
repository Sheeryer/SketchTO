import numpy as np
import random
import torch
import torch.nn as nn

def set_seed(manualSeed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    np.random.seed(manualSeed)
    random.seed(manualSeed)

#%% Neural network
class TopNet(nn.Module):
    def __init__(self, nnSettings, inputDim, manualSeed=1234):
        self.inputDim = inputDim; # x and y coordn of the point ??好像不是，是每个点map到频域的向量
        self.outputDim = 1; # if material/void at the point
        super().__init__();
        self.layers = nn.ModuleList();
        # manualSeed = 1234; # NN are seeded manually
        set_seed(manualSeed);
        current_dim = self.inputDim;
        for lyr in range(nnSettings['numLayers']): # define the layers
            l = nn.Linear(current_dim, nnSettings['numNeuronsPerLyr']);
            nn.init.xavier_normal_(l.weight);
            nn.init.zeros_(l.bias);
            self.layers.append(l);
            current_dim = nnSettings['numNeuronsPerLyr'];
        self.layers.append(nn.Linear(current_dim, self.outputDim));
        self.bnLayer = nn.ModuleList();
        for lyr in range(nnSettings['numLayers']): # batch norm
            self.bnLayer.append(nn.BatchNorm1d(nnSettings['numNeuronsPerLyr']));

    def forward(self, x):
        m = nn.LeakyReLU();
        ctr = 0;
        for layer in self.layers[:-1]: # forward prop
            x = m(self.bnLayer[ctr](layer(x)));
            ctr += 1;
        rho = 0.01 + torch.sigmoid(self.layers[-1](x)).view(-1); # grab only the first output
        return  rho;



class PerfValuesNN(nn.Module):
    def __init__(self, input_dim, conv_layers, fc_layers, output_dim):
        super(PerfValuesNN, self).__init__()

        self.conv_layers = conv_layers
        self.fc_layers = fc_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        for i in range(len(conv_layers)):
            in_dim = input_dim if i==0 else conv_layers[i-1]
            out_dim = conv_layers[i]
            setattr(self, 'conv'+str(i), nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1))
            setattr(self, 'bn'+str(i), nn.BatchNorm2d(out_dim))

        self.leakyReLU = nn.LeakyReLU(negative_slope=0.1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层定义
        for i in range(len(fc_layers)):
            in_dim = conv_layers[-1] if i == 0 else fc_layers[i - 1]
            out_dim = fc_layers[i]
            setattr(self, 'fc'+str(i), nn.utils.weight_norm(nn.Linear(in_dim, out_dim)))

        self.predict = nn.Linear(fc_layers[-1], self.output_dim)
        self.th = nn.Tanh()


    def forward(self, x):

        for i in range(len(self.conv_layers)):
            conv = getattr(self, 'conv'+str(i))
            bn = getattr(self, 'bn' + str(i))
            x = conv(x)
            x = bn(x)
            x = self.leakyReLU(x)
            x = self.pool(x)

        # 假设卷积网络的输出为 x，大小为 (batch_size, channels, height, width)
        # 将 x 输入全局平均池化层
        x_pool = self.global_avg_pool(x)

        # 将池化后的特征图展平成一维向量
        x = x_pool.view(x_pool.size(0), -1)

        # 全连接层前向传播
        for i in range(len(self.fc_layers)):
            fc = getattr(self, 'fc'+str(i))
            x = fc(x)
            x = self.leakyReLU(x)

        x = self.th(self.predict(x))

        return x


class PerfImagesNN(nn.Module):
    def __init__(self, resolution, input_dim, conv_layers, latent_dim, deconv_layers, output_dim, cond_dim):
        super(PerfImagesNN, self).__init__()
        self.resolution = resolution
        self.input_dim = input_dim
        self.conv_layers = conv_layers
        self.latent_dim = latent_dim
        self.deconv_layers = deconv_layers
        self.output_dim = output_dim
        self.cond_dim = cond_dim

        for i in range(len(conv_layers)):
            in_dim = input_dim if i==0 else conv_layers[i-1]
            out_dim = conv_layers[i]
            setattr(self, 'conv'+str(i), nn.Conv2d(in_dim, out_dim, kernel_size=5, stride = 2, padding=1))
            setattr(self, 'bn'+str(i), nn.BatchNorm2d(out_dim))

        # Fully connected layers for mean and log variance
        l = conv_layers[-1]*(resolution//2**len(conv_layers))**2
        self.fc_mu = nn.Linear(l, latent_dim)
        self.fc_logvar = nn.Linear(l, latent_dim)

        # Fully connected layer for decoder input
        self.fc_dec = nn.Linear(latent_dim + cond_dim, l)

        # Decoder: 4 transposed convolutional (deconvolutional) layers
        for i in range(len(deconv_layers)):
            input_dim = conv_layers[-1] if i==0 else deconv_layers[i-1]
            out_dim = deconv_layers[i]
            setattr(self, 'deconv'+str(i), nn.ConvTranspose2d(input_dim, out_dim,
                                                              kernel_size=5, stride=2, padding=1))
            setattr(self, 'dbn'+str(i), nn.BatchNorm2d(out_dim))


        self.deconv_last = nn.ConvTranspose2d(deconv_layers[-1], output_dim, kernel_size=4, stride=2, padding=1)

        self.leakyReLU = nn.LeakyReLU(negative_slope=0.1)

    def encode(self, x):
        for i in range(len(self.conv_layers)):
            conv = getattr(self, 'conv'+str(i))
            bn = getattr(self, 'bn'+str(i))
            x = self.leakyReLU(bn(conv(x)))

        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond):
        z = torch.cat([z, cond], dim=1)

        x = self.fc_dec(z)
        t = self.resolution//2**len(self.conv_layers)
        x = x.view(x.size(0), self.conv_layers[-1], t, t)  # Reshape

        for i in range(len(self.deconv_layers)):
            deconv = getattr(self, 'deconv' + str(i))
            dbn = getattr(self, 'dbn' + str(i))
            x = self.leakyReLU(dbn(deconv(x)))

        x = torch.sigmoid(self.deconv_last(x))  # Use sigmoid for final activation
        return x

    def forward(self, x, cond):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, cond), mu, logvar

