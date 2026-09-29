# Data Preparation

## Download slides from TCGA

To download the WSIs from TCGA, first download and install the [GDC Data Transfer Tool](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool).

Our download script `download_files_single.py` downloads the WSIs based on manifest files. The manifest files we used for Histo-7M are provided in `tcga_misc`. 

### Example usage:
```bash
python -u download_files_single.py --project acc --type tissue
```

You can also download manifest files for each disease type by yourself from [TCGA Site](https://portal.gdc.cancer.gov/projects?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22projects.program.name%22%2C%22value%22%3A%5B%22TCGA%22%5D%7D%7D%5D%7D). The GDC Data Transfer Tool also supports command lines such as:
```bash
gdc-client download -m manifest.<cancer-type>.txt
```

## Download slides from CPTAC

### Downloading TCIA Faspex Packages Using Aspera CLI

The [Cancer Imaging Archive (TCIA)](https://wiki.cancerimagingarchive.net/) contains various datasets, including the Clinical Proteomic Tumor Analysis Consortium (CPTAC) pathology slide collections. 
These collections offer bulk downloads of pathology slides for specific cancer types through Aspera download packages.

We provide the slides we used in Histo-7M in `cptac_misc/comment/`.

## Download slides from PAIP

PAIP platform is currently unavailable. We will provide the tiled images soon.


## Tile slides
```bash
python wsi_tiling.py  \
  --dataset <path of dataset containing svs>  \
  --output <path to store tiled patches>  \
  --scale 20 --patch_size 1024 --num_threads 16
```

Multi-process version (set `--ntasks` to any number > 1 and `--cpus-per-task=1`):
```bash
srun python -u wsi_tiling_mp.py --patch_size 1024 \
        --dataset /path_to_your_dataset/tcga_slides/acc \
        --output /path_to_your_output/tcga_patches3/acc --scale 20 --save_original 
```