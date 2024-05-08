#!/usr/bin/env python
# coding: utf-8

# ## Preparation

import sys
print(sys.path)

#-*- coding : utf-8-*-
# coding:unicode_escape
import warnings
warnings.filterwarnings("ignore")

# import ST_utils
# import train_STAligner
import STAligner

# # the location of R (used for the mclust clustering)
# import os
# os.environ['R_HOME'] = "D:\\anaconda\envs\STAligner\Lib\R"
# os.environ['R_USER'] = "D:\\anaconda\envs\STAligner\Lib\site-packages\rpy2"

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri

import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.linalg

import torch
used_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(used_device)

sample_names = ["10X","slide","stereo"]
input_dir = '/home/lixiangyu/wjk/data/mouseOB/25um'
output_dir = '/home/lixiangyu/wjk/data/mouseOB/25um/STAligner/'
experiment_name = 'MouseOlfactoryBulb25um'


# ## Load Data

from scipy import sparse

Batch_list = []
adj_list = []

for section_id in sample_names:
    print(section_id)

    adata = sc.read_h5ad(input_dir + section_id+'.h5ad')
    print(adata)
    df = adata.obs[['x','y']].astype('float32')
    adata.obsm['spatial'] = df.values
    print(adata.obs.head())
    
    # make spot name unique
    adata.obs_names = [x + '_' + section_id for x in adata.obs_names]

    # Constructing the spatial network
    if(section_id == '10X'):
        STAligner.Cal_Spatial_Net(adata, rad_cutoff=200)
    elif(section_id == 'slide'):
        STAligner.Cal_Spatial_Net(adata, rad_cutoff=50)
    else:
        STAligner.Cal_Spatial_Net(adata, rad_cutoff=3)
    # the spatial network are saved in adata.uns[‘adj’]
    # STAligner.Stats_Spatial_Net(adata) # plot the number of spatial neighbors

    # Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]

    adj_list.append(adata.uns['adj'])
    Batch_list.append(adata)


# ## Concat the scanpy objects for multiple slices

adata_concat = ad.concat(Batch_list, label="slice_name", keys=sample_names)
# adata_concat.obs['Ground Truth'] = adata_concat.obs['Ground Truth'].astype('category')
adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
print('adata_concat.shape: ', adata_concat.shape)


# ## Concat the spatial network for multiple slices
adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1,len(sample_names)):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
adata_concat.uns['edgeList'] = np.nonzero(adj_concat)


# ## Running STAligner
adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh = 50, device=used_device)

# # save embedding
np.savetxt(output_dir + experiment_name + '_STAligner.csv', adata_concat.obsm['STAligner'], delimiter=",")
np.savetxt(output_dir + experiment_name + '_STAGATE.csv', adata_concat.obsm['STAGATE'], delimiter=",")

# # save h5ad

print(type(adata_concat))
print(adata_concat.obsm)
print(adata_concat)

adata_concat.obs = adata_concat.obs.astype('str')
#  tuple 不能保存为h5ad
adata_concat.uns['edgeList'] = list(adata_concat.uns['edgeList'])

print(adata_concat.isbacked)
adata_concat.filename = output_dir + experiment_name + '.h5ad'
print(adata_concat.isbacked)

