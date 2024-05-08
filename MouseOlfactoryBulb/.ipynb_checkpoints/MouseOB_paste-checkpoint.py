import sys
print(sys.path)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import paste as pst
import ot

sample_names = ["10X","slide","stereo"]
input_dir = './'
output_dir = './'
experiment_name = 'MouseOB'

adata_list = []
# meta = list()
for dataset in sample_names:
    print(dataset)
    adata = sc.read_h5ad(input_dir + dataset+'.h5ad')
    df = adata.obs[['x','y']].astype('float32')
    adata.obsm['spatial'] = df.values
    sc.pp.filter_genes(adata, min_counts = 15)
    sc.pp.filter_cells(adata, min_counts = 100)
    adata_list.append(adata)

n_colors = 10
print(n_colors)

import seaborn as sns
slice_colors = sns.color_palette(n_colors = n_colors).as_hex()

import time
start = time.time()

pis = []
for i in range(len(adata_list)-1):
    pi_temp = pst.pairwise_align(adata_list[i], adata_list[i+1], backend = ot.backend.TorchBackend(), use_gpu = True)
    # pi_temp = pst.pairwise_align(adata_list[i], adata_list[i+1])
    pis.append(pi_temp)
    
print('Runtime: ' + str(time.time() - start))

new_slices = pst.stack_slices_pairwise(adata_list, pis)

# save coor
new_coord = new_slices[0].obsm['spatial']
cells = new_slices[0].obs_names
for i in range(1, len(adata_list)):
    new_coord = np.vstack((new_coord, new_slices[i].obsm['spatial']))
    cells = np.hstack((cells, new_slices[i].obs_names))

new_coord=pd.DataFrame(new_coord,index=cells,columns=['x','y'])
new_coord.to_csv(output_dir + "paste_coord_" + experiment_name + ".csv")


slices = adata_list

initial_slice = slices[0].copy()
lmbda = len(slices)*[1/len(slices)]

pst.filter_for_common_genes(slices)

b = []
for i in range(len(slices)):
    b.append(pst.match_spots_using_spatial_heuristic(slices[0].X.todense(), slices[i].X.todense()))


start = time.time()
## Possible to pass in an initial pi (as keyword argument pis_init) 
## to improve performance, see Tutorial.ipynb notebook for more details.
# center_slice, pis = pst.center_align(initial_slice, slices, lmbda) 

# center_slice, pis = pst.center_align(initial_slice, slices, lmbda, random_seed = 5, backend = ot.backend.TorchBackend(), use_gpu = True)
center_slice, pis = pst.center_align(initial_slice, slices, lmbda, pis_init = b, backend = ot.backend.TorchBackend(), use_gpu = True) 

print('Runtime: ' + str(time.time() - start))


W = center_slice.uns['paste_W']
H = center_slice.uns['paste_H']


np.savetxt(output_dir + experiment_name +"_W.csv", W, delimiter=",")
np.savetxt(output_dir + experiment_name +"_H.csv", H, delimiter=",")


center, new_slices = pst.stack_slices_center(center_slice, slices, pis)

center_color = 'orange'
# slices_colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3']
slice_colors = sns.color_palette(n_colors = n_colors).as_hex()

plt.figure(figsize=(7,7))
pst.plot_slice(center,center_color,s=400)
for i in range(len(new_slices)):
    pst.plot_slice(new_slices[i],slice_colors[i],s=400)

plt.legend(handles=[mpatches.Patch(color=slice_colors[0], label='1'),mpatches.Patch(color=slice_colors[1], label='2')])
plt.gca().invert_yaxis()
plt.axis('off')
plt.show()
