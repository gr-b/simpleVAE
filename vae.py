import torch
import torch.nn as nn
import torch.nn.functional as F

bottleneck_size = 2 # n means, n log variances

# VAE has 2 major differences from an autoencoder:
# 1. VAE encoder has two outputs: the means and the standard deviations for each component
#    these are used during training to forward pass vectors _slightly_ unpredictibly different 
#    from what the regular autoencoder would have, resulting in a smoother latent space. 
# 2. KL loss. All this does is encourage the means to be 0 and the standard deviations to be 1.


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.mean_head = nn.Linear(64, bottleneck_size)
        self.std_head  = nn.Linear(64, bottleneck_size)

        # Decoder
        self.d_fc1 = nn.Linear(bottleneck_size, 64)
        self.d_fc2 = nn.Linear(64, 128)
        self.d_fc3 = nn.Linear(128, 512)
        self.d_fc4 = nn.Linear(512, 784)
        self.d_sig = nn.Sigmoid()

    def forward_encoder(self, x):
        x = x.view(-1, 28*28)        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.mean_head(x), self.std_head(x)

    # Backprop can't flow through a random node.
    # So this wouldn't work if we sampled from a gaussian with mean mu and std e^0.5logvar
    # But instead, we can sample from a gaussian to get a z-score, then use that to create our sample
    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        offset = torch.randn_like(std)
        return mu + offset*std

    def forward_decoder(self, z):
        x = F.relu(self.d_fc1(z))
        x = F.relu(self.d_fc2(x))
        x = F.relu(self.d_fc3(x))
        x = self.d_fc4(x)
        x = self.d_sig(x)
        return x

    def forward(self, x):
       mu, logvar = self.forward_encoder(x)
       z = self.sample(mu, logvar)
       x_prime = self.forward_decoder(z)
       return x_prime, mu, logvar # Also pass latent dim forward so we can calculate loss all in one place


            


