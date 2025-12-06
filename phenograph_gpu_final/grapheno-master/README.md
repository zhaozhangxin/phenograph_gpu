# grapheno
A GPU-accelerated implementation of [PhenoGraph](https://github.com/jacoblevine/PhenoGraph) using [NVIDIA RAPIDS](https://github.com/rapidsai) for fast single-cell phenotyping.


## Modifications by Zhangxin Zhao (2025/12/06)

This version has been modified to support newer RAPIDS versions (RAPIDS 23.12 with CUDA 12.8 compatibility) and limited GPU memory configurations (Tested on NVIDIA RTX A5000 GPU):

- **cluster.py**: Fixed compatibility issues with cuGraph API changes in RAPIDS 23.12
  - Updated Jaccard coefficient column naming (handles `source/destination` vs `src/dst` variations)
  - Improved error handling for graph construction edge cases
  
- Implemented chunked processing strategy for large datasets that exceed GPU memory limits
- Added centroid-based cluster merging across chunks to maintain global cluster consistency
- Workflow:
  1. Process data in manageable chunks using grapheno clustering
  2. Extract cluster centroids from each chunk
  3. Merge similar clusters across chunks using hierarchical clustering
  4. Assign unified cluster labels to all data points

## Installation
Install RAPIDS into new environment:
```bash
conda create -n phenoGPU_test -c rapidsai -c nvidia -c conda-forge \
    cudf=23.12 \
    cugraph=23.12 \
    cuml=23.12 \
    python=3.10 \
    cuda-version=12.0 \
    -y
```

Activate the new conda environment:
```bash
conda activate phenoGPU_test
```

Install scikit-learn and scipy:
```bash
mamba install scikit-learn scipy
```

Install grapheno:
```bash
pip install git+https://gitlab.com/eburling/grapheno.git
```

Install holoviews for visualization:
```bash
conda install -c pyviz holoviews bokeh
```


## Usage
See `demo_phenographGPU.ipynb` for a complete example of chunked processing on large datasets.
### Key Parameters
- `n_neighbors`: Number of nearest neighbors (default: 30, reduce for memory constraints)
- `chunk_size`: Data points per chunk (recommended: 10,000 for the total runtime of approximate 11 minutes--> An attempt was made to adjust the value to 20,000; however, the processing time remained unchanged, and the visualization demonstrated minimal improvement.)
- `distance_threshold`: Centroid merging threshold (default: 2.0, adjust based on data scale)
- `min_size`: Minimum cluster size (default: 5)
### Visualization
The notebook includes examples using cuML's TSNE and UMAP for cluster visualization:
Generally speaking, the visual effect of UMAP is better and the clustering is more distinct.

