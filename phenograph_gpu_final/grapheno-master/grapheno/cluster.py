import os
import time
import cuml
import cudf
import cugraph
import dask_cudf
import cupy as cp
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import cugraph.dask.comms.comms as Comms
from cuml.neighbors import NearestNeighbors as NN
from cuml.dask.neighbors import NearestNeighbors as DaskNN

import numpy as np
from tqdm.notebook import tqdm

def generate_dummy_data(n_samples = 5000000,
                        n_features = 20,
                        centers = 30,
                        cluster_std=3.0):
    X, y = cuml.make_blobs(n_samples, n_features, centers, cluster_std)
    columns = [f'feature{i+1}' for i in range(n_features)]
    df = cudf.DataFrame(X, columns=columns).astype('float32')
    df['label'] = y.astype(int)
    df.to_csv('dummy_data.csv', index=False)
    
    
def start_cluster():
    cluster = LocalCUDACluster()
    client = Client(cluster)
    Comms.initialize(p2p=True)
    return cluster, client


def kill_cluster(cluster, client):
    Comms.destroy()
    client.close()
    cluster.close()
    
    
def compute_and_cache_knn_edgelist(input_csv_path, 
                                   knn_edgelist_path, 
                                   features, 
                                   n_neighbors, 
                                   client=None):
    
    print(f'Computing and caching {n_neighbors}NN '
          f'edgelist: {knn_edgelist_path}')
    
    if client:
        chunksize = cugraph.dask.get_chunksize(input_csv_path)
        X = dask_cudf.read_csv(input_csv_path, chunksize=chunksize)
        X = X.loc[:, features].astype('float32')
        model = DaskNN(n_neighbors=n_neighbors+1, client=client)
    else:
        X = cudf.read_csv(input_csv_path)
        X = X.loc[:, features].astype('float32')
        model = NN(n_neighbors=n_neighbors+1)
    
    model.fit(X)
    
    n_vertices = X.shape[0].compute() if client else X.shape[0]
    
    # exclude self index
    knn_edgelist = model.kneighbors(X, return_distance=False).loc[:, 1:]  
    if client: # gather from GPUs and make index a contiguous range
        knn_edgelist = knn_edgelist.compute().reset_index(drop=True)
    knn_edgelist = knn_edgelist.melt(var_name='knn', value_name='dst')
    knn_edgelist = knn_edgelist.reset_index().rename(columns={'index':'src'})
    knn_edgelist = knn_edgelist.loc[:, ['src', 'dst']]
    knn_edgelist['src'] = knn_edgelist['src'] % n_vertices # avoids transpose
    knn_edgelist.to_parquet(knn_edgelist_path)
    
    
def compute_and_cache_jac_edgelist(knn_edgelist_path, 
                                   jac_edgelist_path, 
                                   distributed=False):
    
    print(f'Computing and caching jaccard edgelist: {jac_edgelist_path}')
    knn_graph = load_knn_graph(knn_edgelist_path, distributed)
    jac_graph = cugraph.jaccard(knn_graph)
    jac_graph.to_parquet(jac_edgelist_path)
    
    
def load_knn_graph(knn_edgelist_path, distributed=False):
    G = cugraph.Graph()
    if distributed:
        knn_edgelist = dask_cudf.read_parquet(knn_edgelist_path, 
                                              split_row_groups=True)
        G.from_dask_cudf_edgelist(knn_edgelist, source='src', destination='dst')
    else:
        knn_edgelist = cudf.read_parquet(knn_edgelist_path)
        G.from_cudf_edgelist(knn_edgelist, source='src', destination='dst')
    return G


def load_jac_graph(jac_edgelist_path, distributed=False):
    G = cugraph.Graph()
    if distributed:
        jac_edgelist = dask_cudf.read_parquet(jac_edgelist_path, 
                                              split_row_groups=True)
    else:
        jac_edgelist = cudf.read_parquet(jac_edgelist_path)
    
    # RAPIDS 23.12 fix: Check actual column names and rename if needed
    print(f"DEBUG: Jaccard edgelist columns: {jac_edgelist.columns.tolist()}")
    
    # Handle different column naming conventions
    if 'first' in jac_edgelist.columns and 'second' in jac_edgelist.columns:
        jac_edgelist = jac_edgelist.rename(columns={'first': 'src', 'second': 'dst'})
    elif 'source' in jac_edgelist.columns and 'destination' in jac_edgelist.columns:
        jac_edgelist = jac_edgelist.rename(columns={'source': 'src', 'destination': 'dst'})
    
    if distributed:
        G.from_dask_cudf_edgelist(jac_edgelist, source='src', destination='dst', edge_attr='jaccard_coeff')
    else:
        G.from_cudf_edgelist(jac_edgelist, source='src', destination='dst', edge_attr='jaccard_coeff')
    return G


def sort_by_size(clusters, min_size):
    """
    Relabel clustering in order of descending cluster size.
    New labels are consecutive integers beginning at 0
    Clusters that are smaller than min_size are assigned to -1.
    Copied from https://github.com/jacoblevine/PhenoGraph.
    
    Parameters
    ----------
    clusters: array
        Either numpy or cupy array of cluster labels.
    min_size: int
        Minimum cluster size.
    Returns
    -------
    relabeled: cupy array
        Array of cluster labels re-labeled by size.
        
    """
    relabeled = cp.zeros(clusters.shape, dtype=int)
    _, counts = cp.unique(clusters, return_counts=True)
    # sizes = cp.array([cp.sum(clusters == x) for x in cp.unique(clusters)])
    o = cp.argsort(counts)[::-1]
    for i, c in enumerate(o):
        if counts[c] > min_size:
            relabeled[clusters == c] = i
        else:
            relabeled[clusters == c] = -1
    return relabeled


################################################
# NOTES #
# Jaccard similarity and Leiden clustering don't
# have distributed GPU implementations yet,
# but they probably will soon, at which point
# it will be worth loading graphs using
# dask_cudf edgelists. As of RAPIDS 22.08 there
# is a distributed GPU implementation of Louvain
# if you run out of memory on single GPU
# computation of Leiden clustering. Note that
# such changes to RAPIDS will likely require
# reworking this code to accomodate, but should
# not be too much, e.g. change cugraph.jaccard
# to cugraph.dask.jaccard, etc.
################################################
def cluster(input_csv_path,
            features,
            n_neighbors=30,
            distributed_knn = True,
            distributed_graphs = False,
            min_size=10):
    
    tic = time.time()

    # client=None
    # if any([distributed_knn, distributed_graphs]):
    #     print('Initializing distributed GPU cluster...')
    #     cluster, client = start_cluster()
    #     print(f'Cluster started in {(time.time()-tic):.2f} seconds...')

    knn_edgelist_path = os.path.basename(input_csv_path).rsplit('.', 1)[0]
    knn_edgelist_path = f'{knn_edgelist_path}_{n_neighbors}NN_edgelist.parquet'

    jac_edgelist_path = os.path.basename(knn_edgelist_path).rsplit('.', 1)[0]
    jac_edgelist_path = f'{jac_edgelist_path}_jaccard.parquet'

    subtic = time.time()

    if os.path.exists(jac_edgelist_path):

        print(f'Loading cached jaccard edgelist into graph: {jac_edgelist_path}')

        # if not distributed_graphs:
        #     kill_cluster(cluster, client)

        jac_graph = load_jac_graph(jac_edgelist_path, 
                                   distributed_graphs)

        print(f'Jaccard graph loaded in {(time.time()-subtic):.2f} seconds...')

    elif os.path.exists(knn_edgelist_path):

        print('Loading cached kNN edgelist for Jaccard graph '
              f'computation: {knn_edgelist_path}')

        # if not distributed_graphs:
        #     kill_cluster(cluster, client)

        compute_and_cache_jac_edgelist(knn_edgelist_path, 
                                       jac_edgelist_path, 
                                       distributed_graphs)

        jac_graph = load_jac_graph(jac_edgelist_path, 
                                   distributed_graphs)

        print('Jaccard graph computed, cached, and reloaded in '
              f'{(time.time()-subtic):.2f} seconds...')

    else:
        
        if distributed_knn:
            print('Starting distributed cluster for kNN computation...')
            
            # Disable UCX completely for RAPIDS 23.12
            os.environ['DASK_UCX__ENABLE'] = 'False'
            os.environ['DASK_RMM__POOL_SIZE'] = '1GB'
            
            cluster = LocalCUDACluster(
                n_workers=2,
                threads_per_worker=1,
                protocol='tcp',  # Force TCP, not UCX
                silence_logs=False,
                rmm_pool_size='1GB',
                rmm_managed_memory=False,
                jit_unspill=False,
                local_directory='/tmp/dask-worker-space',
                death_timeout='600s'
            )
            client = Client(
                cluster,
                timeout='600s',
                heartbeat_interval='60s'
            )
            
            try:
                # DO NOT initialize Comms with p2p=True for TCP
                compute_and_cache_knn_edgelist(input_csv_path, 
                                               knn_edgelist_path, 
                                               features, 
                                               n_neighbors, 
                                               client)
            finally:
                client.close()
                cluster.close()
        else:
            print('Using single GPU for kNN computation...')
            compute_and_cache_knn_edgelist(input_csv_path, 
                                           knn_edgelist_path, 
                                           features, 
                                           n_neighbors, 
                                           None)

        print(f'{n_neighbors}NN edgelist computed and cached in '
              f'{(time.time()-subtic):.2f} seconds...')

        subtic = time.time()

        # if not distributed_graphs:
        #     kill_cluster(cluster, client)

        compute_and_cache_jac_edgelist(knn_edgelist_path, 
                                       jac_edgelist_path, 
                                       distributed_graphs)

        jac_graph = load_jac_graph(jac_edgelist_path, 
                                   distributed_graphs)

        print('Jaccard graph computed, cached, and reloaded in '
              f'{(time.time()-subtic):.2f} seconds...')

    subtic = time.time()

    print('Computing Leiden clustering over Jaccard graph...')
    clusters, modularity = cugraph.leiden(jac_graph)
    print(f'Leiden clustering completed in {(time.time()-subtic):.2f} seconds...')
    print(f'Clusters detected: {len(clusters.partition.unique())}')
    print(f'Clusters modularity: {modularity}')
        
    clusters = clusters.sort_values(by='vertex').partition.values
    clusters = sort_by_size(clusters, min_size)

    out_parquet_path = input_csv_path.rsplit('.', 1)[0]
    out_parquet_path = f'{out_parquet_path}_{n_neighbors}NN_leiden.parquet'
    print(f'Writing output dataframe: {out_parquet_path}')
    
    df = cudf.read_csv(input_csv_path)
    df['cluster'] = clusters
    df.to_parquet(out_parquet_path)
    df = cudf.read_parquet(out_parquet_path)
    print(f'Grapheno completed in {(time.time()-tic):.2f} seconds!')
    
    return df