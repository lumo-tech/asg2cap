from typing import Union, Iterable, List
from scipy import sparse
from collections import Counter
import numpy as np
import h5py
import json

from thexp import Delegate
from thexp.frame.builder import X, Y, ID, _Value
from data import transforms

from thexp import Logger

log = Logger()


class ASGLoadDelegate(Delegate):

    def __init__(self,
                 region_id: List[str], json_fs: List[str], hdf5_fs: List[str],
                 mp_fts: np.ndarray,
                 mp_ids: List[int],
                 max_attn_len=10,
                 pixel_reduce=1,
                 num_rels=6
                 ):

        self.sent2int_func = transforms.Sent2Int()
        self.pad_feature_func = transforms.FitFeatureLen(max_attn_len)
        self.pad_sents_func = transforms.FitSentIDLen()

        self.region_ids = region_id
        self.json_fs = json_fs

        # self.region_id_graph_maps = []
        #
        # _size = len(json_fs)
        # log.info('load jsons')
        # for i, f in enumerate(json_fs):
        #     log.inline(i, _size)
        #     with open(f, 'r') as r:
        #         region_id_graph_map = json.load(r)  # type:dict
        #         self.region_id_graph_maps.append(region_id_graph_map)

        self.hdf5_fs = hdf5_fs

        self.mp_fts = mp_fts
        self.mp_ids = mp_ids

        self.max_attn_len = max_attn_len
        self.offset = pixel_reduce
        self.num_rels = num_rels

    def add_obj_attr_edge(self, edges, obj_node_id, attr_node_id):
        edges.append([obj_node_id, attr_node_id, 0])
        edges.append([attr_node_id, obj_node_id, 1])

    def add_rel_subj_edge(self, edges, rel_node_id, subj_node_id):
        edges.append([subj_node_id, rel_node_id, 2])
        edges.append([rel_node_id, subj_node_id, 3])

    def add_rel_obj_edge(self, edges, rel_node_id, obj_node_i):
        edges.append([rel_node_id, obj_node_i, 4])
        edges.append([obj_node_i, rel_node_id, 5])

    def add_flow_edge(self, edge, obj_id, sub_id, edge_id):
        edge.append((sub_id, edge_id))
        edge.append((edge_id, obj_id))

    def __len__(self):
        return len(self.region_ids)

    def __call__(self, index, builder=None) -> Union[X, Y, ID, None, Iterable[_Value]]:
        region_id = self.region_ids[index]
        with open(self.json_fs[index], 'r') as r:
            region_id_graph_map = json.load(r)
        # region_id_graph_map = self.region_id_graph_maps[index]

        region_graph = region_id_graph_map[region_id]
        region_caption = region_graph['phrase']  # type:str

        with h5py.File(self.hdf5_fs[index], 'r') as f:
            key = list(f.keys())[0]
            box_features = f[key][:]
            boxs = f[key].attrs['boxes']

            _box_ft_map = {tuple(box): ft for box, ft in zip(boxs, box_features)}

        attn_features, node_types, attr_order_idxs = [], [], []

        # 用于 relationship 中 obj 和 sub 的寻找和构件
        _obj_id_box_map = {}
        # 用于为每个图的边添加 id
        _obj_id_to_graph_id = {}
        # 用于在 o,a,r 之间构建边
        _edges, _flow_edges = [], []

        _n = 0
        for obj in region_graph['objects']:
            # 将每个 obj 对应的feature 加到 attn_features 中
            # 将每个 obj 的 attr 按照 obj 的feature 加到 attn_features 中
            _obj_id = obj['object_id']

            _box = (obj['x'], obj['y'], obj['x'] + obj['w'] - self.offset, obj['y'] + obj['h'] - self.offset)
            _obj_id_box_map[_obj_id] = _box
            _obj_id_to_graph_id[_obj_id] = _n
            _box_features = _box_ft_map[_box]

            attn_features.append(_box_features)
            attr_order_idxs.append(0)
            node_types.append(1)

            _n += 1
            if _n >= self.max_attn_len:
                break

            for ia, attr in enumerate(obj['attributes']):  # 构建 attr 的 feature 和 edge 的边
                attn_features.append(_box_features)  # attr 的 feature 和 obj 的 feature 一样
                attr_order_idxs.append(ia + 1)
                node_types.append(1)

                self.add_obj_attr_edge(_edges, _obj_id_to_graph_id[_obj_id], _n)
                self.add_flow_edge(_flow_edges, _obj_id_to_graph_id[_obj_id], _obj_id_to_graph_id[_obj_id], _n)
                _n += 1
                if _n >= self.max_attn_len:
                    break
            if _n >= self.max_attn_len:
                break

        if _n < self.max_attn_len:
            for graph in region_graph['relationships']:
                # 将每个relations 相关的两个 obj 获取feature 加 到 attn_ft 中
                _obj_box = _obj_id_box_map[graph['object_id']]
                _sub_box = _obj_id_box_map[graph['subject_id']]
                _box = (min(_obj_box[0], _sub_box[0]), min(_obj_box[1], _sub_box[1]),
                        max(_obj_box[2], _sub_box[2]), max(_obj_box[3], _sub_box[3]))
                _rel_box_features = _box_ft_map[_box]

                attn_features.append(_rel_box_features)
                attr_order_idxs.append(2)
                node_types.append(2)

                self.add_rel_subj_edge(_edges, _n, _obj_id_to_graph_id[graph['subject_id']])
                self.add_rel_obj_edge(_edges, _n, _obj_id_to_graph_id[graph['object_id']])
                self.add_flow_edge(_flow_edges,
                                   _obj_id_to_graph_id[graph['object_id']],
                                   _obj_id_to_graph_id[graph['subject_id']],
                                   _n)
                _n += 1
                if _n >= self.max_attn_len:
                    break

        # padding features
        attn_features, attn_mask = self.pad_feature_func(np.array(attn_features, np.float32))
        node_types = node_types[:self.max_attn_len] + [0] * max(0, self.max_attn_len - len(node_types))
        attr_order_idxs = attr_order_idxs[:self.max_attn_len] + [0] * max(0, self.max_attn_len - len(attr_order_idxs))

        node_types = np.array(node_types, np.long)
        attr_order_idxs = np.array(attr_order_idxs, np.long)

        # mp feature
        mp_feature = self.mp_fts[self.mp_ids[index]]

        # sent label
        caption_ids = self.sent2int_func(region_caption)
        caption_ids, caption_masks = self.pad_sents_func(caption_ids)
        caption_ids = np.array(caption_ids, dtype=np.long)
        caption_masks = np.array(caption_masks)

        # 构建 edge 和 flow_edge 的稀疏矩阵
        if len(_edges) > 0:  # 构建
            _src_nodes, _tgt_nodes, _edge_types = tuple(zip(*_edges))
            _src_nodes = np.array(_src_nodes, np.long)
            _tgt_nodes = np.array(_tgt_nodes, np.long)
            _edge_types = np.array(_edge_types, np.long)
            _edge_counter = Counter(
                [(tgt_node, edge_type) for tgt_node, edge_type in zip(_tgt_nodes, _edge_types)])
            _edge_norms = np.array(
                [1 / _edge_counter[(tgt_node, edge_type)] for tgt_node, edge_type in zip(_tgt_nodes, _edge_types)],
                np.float32)
        else:
            _tgt_nodes = _src_nodes = _edge_types = _edge_norms = np.array([])

        # build python sparse matrix
        edge_sparse_matrices = []
        for i in range(self.num_rels):
            idxs = (_edge_types == i)
            edge_sparse_matrices.append(
                sparse.coo_matrix((_edge_norms[idxs], (_tgt_nodes[idxs], _src_nodes[idxs])),
                                  shape=(self.max_attn_len, self.max_attn_len)))

        # add end flow loop
        _flow_src_nodes = set([x[0] for x in _flow_edges])
        for j in range(_n):
            if j not in _flow_src_nodes:
                _flow_edges.append((j, j))  # end loop
        # flow order graph
        _flow_src_nodes, _flow_tgt_nodes = tuple(zip(*_flow_edges))
        _flow_src_nodes = np.array(_flow_src_nodes, np.long)
        _flow_tgt_nodes = np.array(_flow_tgt_nodes, np.long)
        # normalize by src (collumn)
        _flow_counter = Counter(_flow_src_nodes)
        _flow_edge_norms = np.array(
            [1 / _flow_counter[src_node] for src_node in _flow_src_nodes])

        flow_sparse_matrix = sparse.coo_matrix((_flow_edge_norms, (_flow_tgt_nodes, _flow_src_nodes)),
                                               shape=(self.max_attn_len, self.max_attn_len))

        return (
            self.ID_(index, 'index'),
            self.ID_(region_id, 'region_ids'),
            self.X_(mp_feature, 'mp_fts'),
            self.X_(attn_features, 'attn_fts'),
            self.X_(attn_mask, 'attn_masks'),
            self.X_(node_types, 'node_types'),
            self.X_(attr_order_idxs, 'attr_order_idxs'),
            self.X_(edge_sparse_matrices, 'edge_sparse_matrices'),
            self.X_(flow_sparse_matrix, 'flow_sparse_matrix'),
            self.Y_(caption_ids, 'caption_ids'),
            self.Y_(caption_masks, 'caption_masks'),
        )
