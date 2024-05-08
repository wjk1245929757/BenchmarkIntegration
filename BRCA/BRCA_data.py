import scanpy as sc
import pandas as pd

samples = ['anterior1', 'anterior2', 'posterior1', 'posterior2']
input_dir = 'G:/dataset/2_saggital/input/'
for sample in samples:
    file_path = input_dir + sample
    adata_temp = sc.read_visium(file_path)
    adata_temp.var_names_make_unique()
    adata_temp.obs_names_make_unique()
    meta = pd.read_csv(file_path+ '/truth.csv', index_col = 0)
    meta.rename(index=lambda s: s.replace(".", "-").split("-",1)[1], inplace=True)
    adata_temp.obs = pd.concat([adata_temp.obs, meta],axis=1)
    adata_temp.write( file_path + '/' + sample + '.h5ad' )
    
    
# BRCA

from pathlib import Path
import json
import anndata as ad
from anndata import AnnData
from matplotlib.image import imread

def add_visium_image(
    path: Path | str,
    adata: AnnData = None,
    genome: str | None = None,
    *,
    library_id: str | None = None,
    load_images: bool | None = True,
    source_image_path: Path | str | None = None,
) -> AnnData:
    
    path = Path(path)
    adata.uns["spatial"] = dict()

    from h5py import File

    if library_id is None:
        library_id = 'BRCA'

    adata.uns["spatial"][library_id] = dict()

    if load_images:
        tissue_positions_file = (
            path / "spatial/tissue_positions.csv"
            if (path / "spatial/tissue_positions.csv").exists()
            else path / "spatial/tissue_positions_list.csv"
        )
        files = dict(
            tissue_positions_file=tissue_positions_file,
            scalefactors_json_file=path / "spatial/scalefactors_json.json",
            hires_image=path / "spatial/tissue_hires_image.png",
            lowres_image=path / "spatial/tissue_lowres_image.png",
        )

        # check if files exists, continue if images are missing
        for f in files.values():
            if not f.exists():
                if any(x in str(f) for x in ["hires_image", "lowres_image"]):
                    logg.warning(
                        f"You seem to be missing an image file.\n"
                        f"Could not find '{f}'."
                    )
                else:
                    raise OSError(f"Could not find '{f}'")

        adata.uns["spatial"][library_id]["images"] = dict()
        for res in ["hires", "lowres"]:
            try:
                adata.uns["spatial"][library_id]["images"][res] = imread(
                    str(files[f"{res}_image"])
                )
            except Exception:
                raise OSError(f"Could not find '{res}_image'")

        # read json scalefactors
        adata.uns["spatial"][library_id]["scalefactors"] = json.loads(
            files["scalefactors_json_file"].read_bytes()
        )

#         adata.uns["spatial"][library_id]["metadata"] = {
#             k: (str(attrs[k], "utf-8") if isinstance(attrs[k], bytes) else attrs[k])
#             for k in ("chemistry_description", "software_version")
#             if k in attrs
#         }

        # read coordinates
        positions = pd.read_csv(
            files["tissue_positions_file"],
            header=0 if tissue_positions_file.name == "tissue_positions.csv" else None,
            index_col=0,
        )
        positions.columns = [
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_col_in_fullres",
            "pxl_row_in_fullres",
        ]

        adata.obs = adata.obs.join(positions, how="left")

        adata.obsm["spatial"] = adata.obs[
            ["pxl_row_in_fullres", "pxl_col_in_fullres"]
        ].to_numpy()
        adata.obs.drop(
            columns=["pxl_row_in_fullres", "pxl_col_in_fullres"],
            inplace=True,
        )

        # put image path in uns
        if source_image_path is not None:
            # get an absolute path
            source_image_path = str(Path(source_image_path).resolve())
            adata.uns["spatial"][library_id]["metadata"]["source_image_path"] = str(
                source_image_path
            )

    return adata


samples = ['1142243F', '1160920F']
input_dir = 'G:/dataset/3_BRCA/input/'
# adata = load_data_merge([input_dir+sample+'.h5ad' for sample in samples], samples)
for sample in samples:
    adata_temp = sc.read_h5ad(input_dir+sample+'.h5ad')
    adata_temp = add_visium_image(input_dir+sample, adata_temp)
    print(adata_temp)
