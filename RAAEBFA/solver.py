import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from torch.autograd import grad
from torch.autograd import Variable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *
from data_loader import *
import IPython
from tqdm import tqdm


from MD.model import RAAEBFA

class Solver(object):
    DEFAULTS = {}   
    def __init__(self, train_data_loader, test_data_loader, config, input_dim, latent_dim, name='', BFA=0.5):
        # Data loader
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.gmm_k = 4
        self.BFA = BFA
        self.name = name
        # Build tensorboard if use
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def build_model(self):
        # Define model
        self.raaebfa = RAAEBFA(self.input_dim, self.latent_dim, BFA = self.BFA, n_gmm = self.gmm_k, add_recon_error=True)

        # Optimizers
        self.decoder_optimizer = torch.optim.Adam(self.raaebfa.decoder.parameters(), lr=self.lr)
        self.discriminator_optimizer = torch.optim.Adam(self.raaebfa.discriminator.parameters(), lr=self.lr)
        self.encoder_optimizer = torch.optim.Adam(self.raaebfa.encoder.parameters(), lr=self.lr)
        self.estimator_optimizer = torch.optim.Adam(self.raaebfa.estimation.parameters(), lr = self.lr)
        # self.generator_optimizer = torch.optim.Adam(self.raaebfa.encoder.parameters(), lr=self.lr)

        # Print networks
        self.print_network(self.raaebfa, 'RAAEBFA')

        if torch.cuda.is_available():
            self.raaebfa.cuda()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):
        self.raaebfa.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_raaebfa.pth'.format(self.pretrained_model))))

        print("phi", self.raaebfa.phi,"mu",self.raaebfa.mu, "cov",self.raaebfa.cov)

        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def train_epoch(self):
        epoch_loss = 0
        reconstruction_loss = 0
        discriminator_epoch_loss = 0
        for i, (input_data, _) in enumerate(tqdm(self.train_data_loader)):
                input_data = self.to_var(input_data)
                #print(input_data.size())
                total_loss, sample_energy, recon_error, cov_diag, discriminator_loss, generator_loss = self.raaebfa_step(input_data)
                # Logging
                loss = {}
                loss['total_loss with gmm'] = total_loss.item()
                loss['sample_energy'] = sample_energy.item()
                loss['discriminator_loss'] = discriminator_loss.item()
                loss['recon_loss'] = recon_error
                loss['conv_diag'] = cov_diag.item()
                loss['generator_loss'] = generator_loss.item()
                epoch_loss += total_loss.item()
                reconstruction_loss += recon_error.item()
                discriminator_epoch_loss += discriminator_loss
                log = 'train Loss Infor:'
                if (i+1) % self.log_step == 0:
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)
        return epoch_loss / (i+1), reconstruction_loss / (i+1), discriminator_epoch_loss / (i+1)
    def train_total(self):
        result = []
        train_loss = []
        train_reconstruction_loss = []
        train_discriminator_loss = []
        for i, e in enumerate(range(self.num_epochs)):
            epoch_loss, recon_loss, discriminator_loss = self.train_epoch()
            train_loss.append(epoch_loss)
            train_reconstruction_loss.append(recon_loss)
            train_discriminator_loss.append(discriminator_loss)
            epoch_result = self.test()
            result.append(epoch_result)
        plt.figure()
        plt.subplot(311)
        plt.plot(train_loss)
        plt.subplot(312)
        plt.plot(train_reconstruction_loss)
        plt.subplot(313)
        plt.plot(train_discriminator_loss)
        plt.savefig('./result/'+ self.name +'loss.png')
        return np.array(result)

    def train(self):
        iters_per_epoch = len(self.train_data_loader)

        # Start with trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[0])
        else:
            start = 0

        # Start training
        iter_ctr = 0
        start_time = time.time()
        self.ap_global_train = np.array([0,0,0])
        train_loss = []
        train_reconstruction_loss = []
        train_discriminator_loss = []
        for e in range(start, self.num_epochs):
            epoch_loss = 0
            reconstruction_loss = 0
            discriminator_epoch_loss = 0
            for i, (input_data, _) in enumerate(tqdm(self.train_data_loader)):
                iter_ctr += 1
                start = time.time()

                input_data = self.to_var(input_data)
                #print(input_data.size())
                total_loss, sample_energy, recon_error, cov_diag, discriminator_loss, generator_loss = self.raaebfa_step(input_data)
                # Logging
                loss = {}
                loss['total_loss with gmm'] = total_loss.item()
                loss['sample_energy'] = sample_energy.item()
                loss['discriminator_loss'] = discriminator_loss.item()
                loss['recon_loss'] = recon_error
                loss['conv_diag'] = cov_diag.item()
                loss['generator_loss'] = generator_loss.item()
                epoch_loss += total_loss.item()
                reconstruction_loss += recon_error.item()
                discriminator_epoch_loss += discriminator_loss
                if (i+1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    total_time = ((self.num_epochs*iters_per_epoch)-(e*iters_per_epoch+i)) * elapsed/(e*iters_per_epoch+i+1)
                    epoch_time = (iters_per_epoch-i)* elapsed/(e*iters_per_epoch+i+1)
                    
                    epoch_time = str(datetime.timedelta(seconds=epoch_time))
                    total_time = str(datetime.timedelta(seconds=total_time))
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    # 这里没弄明白啥意思，需要再看看
                    lr_tmp = []
                    for param_group in self.decoder_optimizer.param_groups:
                        lr_tmp.append(param_group['lr'])
                    tmplr = np.squeeze(np.array(lr_tmp))

                    log = "Elapsed {}/{} -- {} , Epoch [{}/{}], Iter [{}/{}], lr {}".format(
                        elapsed,epoch_time,total_time, e+1, self.num_epochs, i+1, iters_per_epoch, tmplr)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)        
            train_loss.append(epoch_loss / (i+1))
            train_reconstruction_loss.append(reconstruction_loss / (i+1))
            train_discriminator_loss.append(discriminator_epoch_loss / (i+1))
            torch.save(self.raaebfa.state_dict(), os.path.join(self.model_save_path, '{}_raaebfa.pth'.format(e+1)))
        plt.figure()
        plt.subplot(311)
        plt.plot(train_loss)
        plt.subplot(312)
        plt.plot(train_reconstruction_loss)
        plt.subplot(313)
        plt.plot(train_discriminator_loss)
        plt.savefig('./result/'+ self.name +'loss.png')

    def raaebfa_step(self, input_data):
        self.raaebfa.train()
        # reconstruction step 
        z, dec, gmm_input, gamma = self.raaebfa(input_data)
        total_loss, sample_energy, recon_error, cov_diag = self.raaebfa.reconstruction_step_loss_function(input_data, dec, z, gmm_input, gamma, self.lambda_energy, self.lambda_cov_diag)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.estimator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

        total_loss = Variable(total_loss, requires_grad = True)
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.raaebfa.parameters(), 5)
        self.estimator_optimizer.step()
        self.decoder_optimizer.step()
        self.encoder_optimizer.step()
        

        # regularization step    
        self.raaebfa.encoder.eval()
        discriminator_loss, generator_loss = self.raaebfa.regularization_step_loss_function(input_data)
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        # self.reset_grad()
        
        self.raaebfa.encoder.train()
        self.encoder_optimizer.zero_grad()
        generator_loss.backward()
        self.encoder_optimizer.step()       
        return total_loss, sample_energy, recon_error, cov_diag, discriminator_loss, generator_loss
    def test(self):
        print("======================TEST MODE======================")
        self.raaebfa.eval()
        # calculate train_phi, train_mu, train_cov
        N = 0
        mu_sum = 0
        cov_sum = 0
        gamma_sum = 0

        for it, (input_data, labels) in enumerate(self.train_data_loader):
            input_data = self.to_var(input_data)
            z, dec, gmm_input, gamma = self.raaebfa(input_data)
            phi, mu_gmm, cov = self.raaebfa.compute_gmm_params(gmm_input, gamma)
            batch_gamma_sum = torch.sum(gamma, dim=0)
            
            gamma_sum += batch_gamma_sum
            mu_sum += mu_gmm * batch_gamma_sum.unsqueeze(-1) # keep sums of the numerator only
            cov_sum += cov * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1) # keep sums of the numerator only
            
            N += input_data.size(0)
            
        train_phi = gamma_sum / N
        train_mu = mu_sum / gamma_sum.unsqueeze(-1)
        train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        # caculate anomaly score
        test_energy = []
        test_labels = []
        test_z = []
        for it, (input_data, labels) in enumerate(self.test_data_loader):
            input_data = self.to_var(input_data)
            z, dec, gmm_input, gamma = self.raaebfa(input_data)
            sample_energy, cov_diag = self.raaebfa.compute_energy(gmm_input, train_phi, train_mu, train_cov, size_average=False)
            test_energy.append(sample_energy.data.cpu().numpy())
            test_z.append(gmm_input.data.cpu().numpy())
            test_labels.append(labels.numpy())

        # print(test_energy)
        test_energy = np.concatenate(test_energy,axis=0)
        test_z = np.concatenate(test_z,axis=0)
        test_labels = np.concatenate(test_labels,axis=0)

        # 排序后的 80%的分位数
        percent = self.test_data_loader.dataset.test_PN_percent()
        print(percent)
        thresh = np.percentile(test_energy, percent) 
        print("Threshold :", thresh)

        pred = (test_energy > thresh).astype(int)
        pred[pred == 0] = -1
        gt = test_labels.astype(int)
        print('predict', np.sum(pred))
        print('get labels', np.sum(gt))
        from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
        accuracy = accuracy_score(gt,pred)
        precision, recall, f_score, support = prf(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(accuracy, precision, recall, f_score))        
        return accuracy, precision, recall, f_score
    