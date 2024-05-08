#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")

import STAligner

import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.linalg
import os

import torch
used_device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

sample_names = ["embryo1-2","embryo1-5","embryo2-2","embryo2-5","embryo3-2","embryo3-5"]
input_dir = '/home/lixiangyu/wjk/data/MouseAtlas/'
output_dir = '/home/lixiangyu/wjk/data/MouseAtlas/STAligner/'
experiment_name = 'all'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

Batch_list = []
adj_list = []

section_ids = sample_names
for section_id in section_ids:
    print(section_id)
    adata = sc.read_h5ad(input_dir + section_id + ".h5ad")

    STAligner.Cal_Spatial_Net(adata, rad_cutoff=50)

    # Normalization
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000) #ensure enough common HVGs in the combined matrix
    adata = adata[:, adata.var['highly_variable']]

    adj_list.append(adata.uns['adj'])
    Batch_list.append(adata)
    
adata_concat = ad.concat(Batch_list, label="batch", keys=section_ids)
adata_concat.obs["batch_name"] = adata_concat.obs["batch"].astype('category')
print('adata_concat.shape: ', adata_concat.shape)

adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1,len(section_ids)):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, device=used_device)

# # save embedding
np.savetxt(output_dir + experiment_name + '_STAligner.csv', adata_concat.obsm['STAligner'], delimiter=",")
np.savetxt(output_dir + experiment_name + '_STAGATE.csv', adata_concat.obsm['STAGATE'], delimiter=",")

adata_concat.obs = adata_concat.obs.astype('str')
#  tuple 不能保存为h5ad
adata_concat.uns['edgeList'] = list(adata_concat.uns['edgeList'])

print(adata_concat.isbacked)
adata_concat.filename = output_dir + experiment_name + '.h5ad'
print(adata_concat.isbacked)
