#!/usr/bin/env python
# coding: utf-8

import os 

print(os.getcwd())#显示当前路径
os.chdir('/home/lixiangyu/benchmark/DeepST/DeepST-main/deepst')#更改路径，''里面为更改的路径
print(os.getcwd())#显示当前路径

from DeepST import run
import matplotlib.pyplot as plt
from pathlib import Path
import scanpy as sc
import pandas as pd
import numpy as np
from anndata import AnnData


data_name_list = ['HCC-5A', 'HCC-5B', 'HCC-5C', 'HCC-5D']
data_path = '/home/lixiangyu/benchmark/data/LICA/'
save_path = '/home/lixiangyu/benchmark/data/LICA/DeepST/'


deepen = run(save_path = save_path, 
	task = "Integration",
	pre_epochs = 800, 
	epochs = 1000, 
	use_gpu = True,
	)

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from scipy.sparse import issparse,csr_matrix
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler
from torch.utils.data import TensorDataset

import matplotlib.pyplot as plt

from pathlib import Path, PurePath
from typing import Optional, Union
from anndata import AnnData
import numpy as np
from PIL import Image
import pandas as pd
# import stlearn
from _compat import Literal
import scanpy
import scipy

from matplotlib.image import imread
import json


###### Generate an augmented list of multiple datasets
augement_data_list = []
graph_list = []
from scipy.sparse import csr_matrix
for i in range(len(data_name_list)):
	print(data_name_list[i])
	adata = deepen._get_adata(platform="Visium", data_path=data_path, data_name=data_name_list[i])
	adata = deepen._get_image_crop(adata, data_name=data_name_list[i])
	adata = deepen._get_augment(adata, spatial_type="LinearRegress")
	graph_dict = deepen._get_graph(adata.obsm["spatial"], distType = "KDTree")
	augement_data_list.append(adata)
	graph_list.append(graph_dict)

print('single adata OK!')

######## Synthetic Datasets and Graphs
multiple_adata, multiple_graph = deepen._get_multiple_adata(adata_list = augement_data_list, data_name_list = data_name_list, graph_list = graph_list)

print('multiple adata OK!')
###### Enhanced data preprocessing
print('Enhanced data preprocessing')
data = deepen._data_process(multiple_adata, pca_n_comps = 200)


deepst_embed = deepen._fit(
		data = data,
		graph_dict = multiple_graph,
		domains = multiple_adata.obs["batch"].values,  ##### Input to Domain Adversarial Model
		n_domains = len(data_name_list))
np.savetxt(os.path.join(save_path, "deepst_embedding.csv"), deepst_embed, delimiter=",")

n_domains = 6

multiple_adata.obsm["DeepST_embed"] = deepst_embed
multiple_adata = deepen._get_cluster_data(multiple_adata, n_domains=n_domains, priori = True)


sc.pp.neighbors(multiple_adata, use_rep='DeepST_embed')
sc.tl.umap(multiple_adata)
sc.pl.umap(multiple_adata, color=["DeepST_refine_domain","batch_name"])
plt.savefig(os.path.join(save_path, f'{"_".join(data_name_list)}_umap.pdf'), bbox_inches='tight', dpi=300)


for data_name in data_name_list:
	adata = multiple_adata[multiple_adata.obs["batch_name"]==data_name]
	sc.pl.spatial(adata, color='DeepST_refine_domain', frameon = False, spot_size=150)
	plt.savefig(os.path.join(save_path, f'{data_name}_domains.pdf'), bbox_inches='tight', dpi=300)


print(multiple_adata.isbacked)
multiple_adata.filename = save_path + '/LICA.h5ad'
print(multiple_adata.isbacked)


