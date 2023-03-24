import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch.utils.data import DataLoader, dataset, TensorDataset
from torch import nn, optim
from torch.optim import lr_scheduler

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import pytorch_lightning as pl



class autoencoder(pl.LightningModule):
    def __init__(self,learning_rate,epochs,ncomp, input_size, nhidden=3):
        super().__init__()
        self.automatic_optimization = True
        self.learning_rate = learning_rate
        self.epochs=epochs
        self.ncomp=ncomp
        self.loss = nn.L1Loss()
        
        
        params_encoder = np.linspace(input_size,ncomp,nhidden)
        modules = []
        for k in range(nhidden-1):
            modules.append(nn.Linear(int(params_encoder[k]) ,int(params_encoder[k+1])))
            modules.append(nn.LeakyReLU(0.2))  
        self.encoder = nn.Sequential(*modules)
        
        params_decoder = np.linspace(ncomp,input_size,nhidden)
        modules = []
        for k in range(nhidden-1):
            modules.append(nn.Linear(int(params_decoder[k]) ,int(params_decoder[k+1])))
            modules.append(nn.LeakyReLU(0.2))  
        self.decoder = nn.Sequential(*modules)
        
        
    def forward(self, xinp):
        xlatent = self.encoder(xinp)
        xout = self.decoder(xlatent)

        return xlatent,xout  
    
    def training_step(self, batch, batch_idx):
        xinp = batch
        #print(xinp[0])
        
        xlatent,xout  = self(xinp[0])#.cuda()
        loss = self.loss(xout,xinp[0])

        self.log('train_loss', loss, prog_bar=True)
        
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, logger=False)
    
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=int(0.75*self.epochs), gamma=0.1)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
