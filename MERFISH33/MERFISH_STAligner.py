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

sample_names = [str(i) for i in range(33)]
input_dir = '/home/lixiangyu/wjk/data/MERFISH/'
output_dir = '/home/lixiangyu/wjk/data/MERFISH/STAligner/'
experiment_name = 'MERFISH'

Batch_list = []
adj_list = []

section_ids = [str(i) for i in range(33)]
for section_id in section_ids:
    print(section_id)
    adata = sc.read_h5ad(os.path.join(input_dir + 'merfish_mouse_brain_slice' + section_id + ".h5ad"))
    
    import numpy as np
    from scipy.sparse import csr_matrix
    adata.obsm['spatial'].drop(columns=['Z'], inplace = True)
    adata.X = csr_matrix(adata.X)
    
    # make spot name unique
    adata.obs_names = [x + '_' + section_id for x in adata.obs_names]

    STAligner.Cal_Spatial_Net(adata, rad_cutoff=50)

    # Normalization
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000) #ensure enough common HVGs in the combined matrix
    adata = adata[:, adata.var['highly_variable']]

    adj_list.append(adata.uns['adj'])
    Batch_list.append(adata)
    
adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids)
adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
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
