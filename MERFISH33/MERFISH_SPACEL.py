import scanpy as sc
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import SPACEL
from SPACEL import Scube
    
sample_names = [str(i) for i in range(33)]
input_dir = '/home/lixiangyu/benchmark/data/MERFISH/'
output_dir = '/home/lixiangyu/benchmark/data/MERFISH/SPACEL/'
experiment_name = 'MERFISH'


adata_list = []
# meta = list()
for dataset in sample_names:
    print(dataset)
    adata = sc.read_h5ad(os.path.join(input_dir + 'merfish_mouse_brain_slice' + dataset + ".h5ad"))
    adata_list.append(adata)

Scube.align(adata_list,
      cluster_key='spatial_domain',
      n_neighbors = 15,
      n_threads=10,
      p=2,
      write_loc_path=output_dir+'aligned_coordinates.csv'
     )
sc.concat(adata_list).write(output_dir+'merfish_mouse_brain.h5ad')

coo = pd.DataFrame()
for i in range(len(adata_list)):
    loc = adata_list[i].obsm['spatial_aligned'].copy()
    loc['Z'] = i
    loc['celltype_colors'] = adata_list[i].obs['spatial_domain'].replace(dict(zip([4,1,6,2,3,5,0],['#aad466', '#f4ed27','#f9973f','#e76f5a','#40ecd4', '#a62098', '#a4bcda'])))
    coo = pd.concat([coo,loc],axis=0)
coo.to_csv(output_dir+'aligned_coordinates_colors.csv')