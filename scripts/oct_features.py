from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class extract_features():
    def __init__(self, archive):
        self.archive=archive
        
    def _pca_features(self,n_components, atr):
        pca = PCA(n_components=n_components)
        pca_comp = pca.fit_transform(getattr(self.archive, atr))
        print(f'{n_components} PCA components explain {pca.explained_variance_ratio_.sum()} of the variance')
        
        self.pca_comp = pca_comp
        return pca_comp
    
    def _LDA_features(self,atr):
        lda = LDA()
        lda_comp = lda.fit_transform(getattr(self.archive, atr))
        print('2 LDA components explain {lda.explained_variance_ratio_.sum()} of the variance')
        
        self.lda_comp = lda_comp
        return lda_comp
    
    
        
        