import torch
import torch.nn as nn
import math
import torch.nn.functional as F
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout,noise_std,gene_dropout,logvar_min,logvar_max):
        super().__init__()
        self.noise_std = noise_std
        self.gene_dropout = gene_dropout
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        # encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.LayerNorm(hidden_dim))
            encoder_layers.append(nn.GELU())
            encoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        # decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.LayerNorm(h))
            decoder_layers.append(nn.GELU())
            decoder_layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.decoder = nn.Sequential(*decoder_layers)
        self.u = nn.Linear(prev_dim, input_dim)
        self.logvar = nn.Linear(prev_dim, input_dim)
    def perturb(self, x):
        if self.training:
            if self.noise_std>0.0:
                x = x + torch.randn_like(x) * self.noise_std
            if self.gene_dropout > 0.0:
                keep = 1.0 - self.gene_dropout
                mask = (torch.rand_like(x) < keep).float()
                x = x * mask / keep
        return x
    def forward(self, x):
        x_perturbed = self.perturb(x)
        z = self.encoder(x_perturbed)
        h = self.decoder(z)
        u = self.u(h)
        raw = self.logvar(h)
        var = F.softplus(raw) + 1e-4
        logvar = torch.log(var)
        logvar = torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)
        return u, logvar,z
    def guassian_NLL(self,u,logvar,target,reduction='mean'):
        inverse_var = torch.exp(-logvar)
        NLL = 0.5*((target-u)**2*inverse_var+logvar+ math.log(2 * math.pi))
        if reduction=='mean':
            return NLL.mean()
        elif reduction=='sum':
            return NLL.sum()
        else:
            return  NLL
    def decorrelation_loss(self,z):
        z_centered = z-z.mean(dim=0,keepdim=True)
        N =  z_centered.size(0)
        if N<=1:
            return z.new_tensor(0.0,requires_grad=True)
        cov = (z_centered.T @ z_centered)/(N-1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag ** 2).mean()



