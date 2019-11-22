import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.autograd import Variable
import itertools
from utils import *

# latent_dim = 3
# 真正的gaussian 与 fake gaussian 训练模型次数的比例
real_gaussian_times = 5

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 20), nn.Tanh())
        self.fc2 = nn.Sequential(nn.Linear(20, latent_dim))
    def forward(self, x):
        y = self.fc1(x)
        return self.fc2(y)
class Decoder(nn.Module):
    def __init__(self, output_dim, latent_dim):
        super(Decoder, self).__init__()
        self.fc3 = nn.Sequential(nn.Linear(latent_dim, 20), nn.Tanh())
        self.fc4 = nn.Sequential(nn.Linear(20, output_dim))
    def forward(self, x):
        y = self.fc3(x)
        return self.fc4(y)

class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()
        self.d1 = nn.Sequential(nn.Linear(latent_dim, 10), nn.Tanh())
        self.d2 = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())
    def forward(self, x):
        y = self.d1(x)
        return self.d2(y)
    
class Estimation(nn.Module):
    def __init__(self, latent_dim, n_gmm = 2, add_recon_error=True):
        super(Estimation, self).__init__()
  
        self.es1 = nn.Sequential(nn.Linear(latent_dim, 10), nn.Tanh())
        self.es2 = nn.Dropout(p=0.5)
        self.es3 = nn.Linear(10, n_gmm)
        self.es4 = nn.Softmax(dim = 1)
    def forward(self, x):
        y = self.es1(x)
        y = self.es2(y)
        y = self.es3(y)
        return self.es4(y)

class RAAEBFA(nn.Module):
    """Residual Block."""
    def __init__(self, input_dim, latent_dim, BFA = 0.5, n_gmm = 2, add_recon_error=True):
        '''
        BFA: balance feature adaptation parameter: 0.5 default
        n_gmm: GMM parameter
        add_recon_error : bool value, if need to add reconstruction error
        '''
        super(RAAEBFA, self).__init__()        
        self.latent_dim = latent_dim
        self.BFA = BFA
        self.gmm_k = n_gmm
        self.add_recon_error = add_recon_error
        # encoder
        self.encoder = Encoder(input_dim, latent_dim)
        # decoder
        self.decoder = Decoder(input_dim, latent_dim)
        # estimator
        if self.add_recon_error:
            self.estimation = Estimation(latent_dim+2, n_gmm=self.gmm_k, add_recon_error=self.add_recon_error)
            self.register_buffer("phi", torch.zeros(n_gmm))
            self.register_buffer("mu", torch.zeros(n_gmm,latent_dim+2))
            self.register_buffer("cov", torch.zeros(n_gmm,latent_dim+2,latent_dim+2))
        else:
            self.estimation = Estimation(latent_dim, n_gmm=self.gmm_k, add_recon_error=self.add_recon_error)
            self.register_buffer("phi", torch.zeros(n_gmm))
            self.register_buffer("mu", torch.zeros(n_gmm,latent_dim))
            self.register_buffer("cov", torch.zeros(n_gmm,latent_dim,latent_dim))
        # discriminator 
        self.discriminator = Discriminator(latent_dim)
        
        #register buffer
        

    def relative_euclidean_distance(self, a, b):
        return (a-b).norm(2, dim=1) / a.norm(2, dim=1)

    def reconstruction_step(self, x):
        z = self.encoder(x)
        dec = self.decoder(z)
        rec_cosine = F.cosine_similarity(x, dec, dim=1)
        rec_euclidean = self.relative_euclidean_distance(x, dec)
        rec_cosine = rec_cosine.unsqueeze(-1)
        rec_euclidean = rec_euclidean.unsqueeze(-1)
        Normlize = torch.nn.BatchNorm1d(rec_cosine.size(1))
        rec_cosine = Normlize(rec_cosine)
        Normlize = torch.nn.BatchNorm1d(rec_euclidean.size(1))
        rec_euclidean = Normlize(rec_euclidean)
        if self.add_recon_error:
            gmm_input = torch.cat([(1 - self.BFA) * z, self.BFA * rec_cosine, self.BFA * rec_euclidean], dim=1)
        else:
            gmm_input = z
        
        gamma = self.estimation(gmm_input)
        return z, dec, gmm_input, gamma

    def regularization_step(self, x):
        return      
        
    def forward(self, x):
        z = self.encoder(x)
        dec = self.decoder(z)
        rec_cosine = F.cosine_similarity(x, dec, dim=1)
        rec_euclidean = self.relative_euclidean_distance(x, dec)
        rec_cosine = rec_cosine.unsqueeze(-1)
        rec_euclidean = rec_euclidean.unsqueeze(-1)
        Normlize = torch.nn.BatchNorm1d(rec_cosine.size(1))
        if torch.cuda.is_available():
            Normlize = Normlize.cuda()
        rec_cosine = Normlize(rec_cosine)
        Normlize = torch.nn.BatchNorm1d(rec_euclidean.size(1))
        if torch.cuda.is_available():
            Normlize = Normlize.cuda()
        rec_euclidean = Normlize(rec_euclidean)
        if self.add_recon_error:
            gmm_input = torch.cat([(1 - self.BFA) * z, self.BFA * rec_cosine, self.BFA * rec_euclidean], dim=1)
        else:
            gmm_input = z
        gamma = self.estimation(gmm_input)
        return z, dec, gmm_input, gamma

    def compute_gmm_params(self, gmm_input, gamma):
        N = gamma.size(0)
        # K
        sum_gamma = torch.sum(gamma, dim=0)

        # K
        phi = (sum_gamma / N)

        self.phi = phi.data

 
        # K x D
        mu = torch.sum(gamma.unsqueeze(-1) * gmm_input.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        # z = N x D
        # mu = K x D
        # gamma N x K

        # z_mu = N x K x D
        z_mu = (gmm_input.unsqueeze(1)- mu.unsqueeze(0))

        # z_mu_outer = N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim = 0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data

        return phi, mu, cov
        
    def compute_energy(self, gmm_input, phi=None, mu=None, cov=None, size_average=True):
        if phi is None:
            phi = to_var(self.phi)
        if mu is None:
            mu = to_var(self.mu)
        if cov is None:
            cov = to_var(self.cov)

        k, D, _ = cov.size()

        z_mu = (gmm_input.unsqueeze(1)- mu.unsqueeze(0))

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        for i in range(k):
            # K x D x D
            cov_k = cov[i] + to_var(torch.eye(D)*eps)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))

            #det_cov.append(np.linalg.det(cov_k.data.cpu().numpy()* (2*np.pi)))
            #det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
            det_cov.append((torch.cholesky(cov_k.cpu() * (2 * np.pi)).diag().prod()).unsqueeze(0))
            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        if torch.cuda.is_available():
            det_cov = torch.cat(det_cov).cuda()
        else:
            det_cov = torch.cat(det_cov)
        #det_cov = to_var(torch.from_numpy(np.float32(np.array(det_cov))))

        # N x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        # for stability (logsumexp)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val)

        # sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (det_cov).unsqueeze(0), dim = 1) + eps)
        sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim = 1) + eps)
        # sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt((2*np.pi)**D * det_cov)).unsqueeze(0), dim = 1) + eps)


        if size_average:
            sample_energy = torch.mean(sample_energy)
        return sample_energy, cov_diag


    
    def reconstruction_step_loss_function(self, x, x_hat, z, gmm_input, gamma, lambda_energy, lambda_cov_diag):

        recon_error = torch.mean((x - x_hat) ** 2)
        # gmm loss
        phi, mu_gmm, cov = self.compute_gmm_params(gmm_input, gamma)
        sample_energy, cov_diag = self.compute_energy(gmm_input, phi, mu_gmm, cov)
        
        # sparse kl loss
        # rho_hat = torch.sum(z, dim=0, keepdim=True)
        # sparsity_penalty = self.sparse_Beta * self.sparse_KL_loss(rho_hat)
        #print('sparse loss', sparsity_penalty.item())
        total_energy = lambda_energy * sample_energy
        regular_loss = lambda_cov_diag * cov_diag
        loss = recon_error + total_energy + regular_loss
        
        return loss, sample_energy, total_energy, regular_loss
    
    def regularization_step_loss_function(self, x):
        # Discriminator step
        # 
        self.encoder.eval() 
        for _ in range(real_gaussian_times):
            
            z_fake_gaussian = self.encoder(x)
            z_real_gaussian = torch.randn(z_fake_gaussian.size())
            if torch.cuda.is_available():
                z_real_gaussian = z_real_gaussian.cuda()
            d_real = self.discriminator(z_real_gaussian)
            d_fake = self.discriminator(z_fake_gaussian)
            discriminator_loss = -(torch.mean(d_real) - torch.mean(d_fake))
            
        # generator
        z_fake = self.encoder(x)
        D_fake = self.discriminator(z_fake)
        
        generator_loss = -torch.mean(D_fake)
            
        return discriminator_loss, generator_loss
        