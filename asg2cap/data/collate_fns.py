from torch.utils.data._utils import collate
import torch
import numpy as np
from thexp.utils.timing import timeit



# collate.default_collate()

def convert_batch_sparse_matrix_collate_fn(batch_data: dict):
    timeit.mark('convert')
    batch_size = len(batch_data)
    _sample = batch_data[0]
    max_nodes, _ = _sample['attn_fts'].shape
    num_rels = len(_sample['edge_sparse_matrices'])

    rel_edges = np.zeros((batch_size, num_rels, max_nodes, max_nodes), dtype=np.float32)
    for i, _sample in enumerate(batch_data):
        for j, edge_sparse_matrix in enumerate(_sample['edge_sparse_matrices']):
            rel_edges[i, j] = edge_sparse_matrix.todense()

        _sample['flow_edges'] = _sample['flow_sparse_matrix'].toarray().astype(np.float32)

    [i.pop('edge_sparse_matrices') for i in batch_data]
    [i.pop('flow_sparse_matrix') for i in batch_data]
    batch_data = collate.default_collate(batch_data)
    batch_data['rel_edges'] = torch.tensor(rel_edges)
    timeit.mark('convert_e')
    return batch_data
