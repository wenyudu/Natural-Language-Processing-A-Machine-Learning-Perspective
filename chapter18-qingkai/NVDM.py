# A simple implementation of neural variational document model (NVDM).

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


def block(in_d, out_d):
    layers = [nn.Linear(in_d, out_d), nn.Softplus()]
    return layers


def rsample(mean, log_variance):
    epsilon = torch.randn_like(mean)
    return mean + epsilon * (log_variance / 2).exp()


class NVDM(nn.Module):
    def __init__(self, input_dim, inter_dim, hid_dim):
        super(NVDM, self).__init__()
        self.encoder = nn.Sequential(
            *block(input_dim, inter_dim), *block(inter_dim, inter_dim))
        self.mean = nn.Linear(inter_dim, hid_dim)
        self.log_variance = nn.Linear(inter_dim, hid_dim)
        self.decoder = nn.Linear(hid_dim, input_dim)
        self.prior_mean = nn.Parameter(torch.zeros(1, hid_dim), requires_grad=False)
        self.prior_log_variance = nn.Parameter(torch.full((1, hid_dim), np.log(1)), requires_grad=False)
        self.hid_dim = hid_dim

    def forward(self, x, mask):
        # encoder as equation 18.49
        enc = self.encoder(x)
        mean = self.mean(enc)
        log_variance = self.log_variance(enc)

        # repameterisation trick
        z = rsample(mean, log_variance)

        # decoder
        recon = self.decoder(z)

        # reconstruction loss
        logits = F.log_softmax(recon, dim=-1)
        recon_loss = -(logits * x).sum(1)

        # kld
        logvar_division = self.prior_log_variance - log_variance
        var_division = (log_variance - self.prior_log_variance).exp()
        diff = mean - self.prior_mean
        diff_term = diff.pow(2) / self.prior_log_variance.exp()
        kld = 0.5 * ((var_division + diff_term +
                      logvar_division).sum(-1) - self.hid_dim)

        # loss is composed of two parts:
        loss = (recon_loss + kld) * mask
        return loss.mean()


def train(nvdm, train_url, optimizer, batch_size=64, training_epochs=1000):
    train_set, train_count = utils.data_set(train_url)
    for epoch in range(training_epochs):
        train_batches = utils.create_batches(len(train_set), batch_size)
        loss_sum = 0.0
        for idx_batch in train_batches:
            data_batch, count_batch, mask = utils.fetch_data(train_set, train_count, idx_batch, 2000)
            data_batch = torch.FloatTensor(data_batch)
            mask = torch.FloatTensor(mask)
            loss = nvdm(data_batch, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        print(loss_sum/len(train_batches))


def main():
    train_url = os.path.join("train.feat")
    nvdm = NVDM(2000, 500, 50)
    optimizer = torch.optim.Adam(nvdm.parameters(), 5e-4)
    train(nvdm, train_url, optimizer)

if __name__ == "__main__":
    main()
