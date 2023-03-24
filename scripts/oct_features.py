from sklearn.decomposition import PCA
from autoencoder import MDNemulator_polyfit
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, dataset, TensorDataset
import torch


class extract_features():
    def __init__(self, archive):
        self.archive=archive
        
    def _pca_features(self,n_components, atr):
        pca = PCA(n_components=n_components)
        pca_comp = pca.fit_transform(getattr(self.archive, atr))
        print(f'{n_components} PCA components explain {pca.explained_variance_ratio_.sum()} of the variance')
        
        self.pca_comp = pca_comp
        return pca_comp
    
    def _autoencoder_features(self,ncomp,atr,lr=1e-3, epochs=100):
        X_dat = getattr(self.archive, atr)
        input_size = X_dat.shape[1]
        
        trainig_dataset = TensorDataset(torch.Tensor(X_dat))
        loader_train = DataLoader(trainig_dataset, batch_size=100, shuffle = True)
        
        autoencoder = MDNemulator_polyfit(learning_rate=lr,epochs=epochs,ncomp=ncomp, input_size=input_size)
        trainer = Trainer(max_epochs=epochs,enable_progress_bar=False)

        trainer.fit(autoencoder, train_dataloaders=loader_train)

        X_dat_latent,X_dat_out = autoencoder(torch.Tensor(X_dat))
        
        encoder_comp = X_dat_latent.detach().cpu().numpy()
        self.encoder_comp=encoder_comp
        return X_dat_out.detach().cpu().numpy(), encoder_comp
    
    
        
        