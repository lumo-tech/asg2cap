from typing import Dict, Callable, Tuple
import json
import os

from PIL import Image
from thexp import globs
from thexp.decorators import regist_func
from thexp.base_classes import llist
import numpy as np

# globs.add_value('datasets', 'path/to/all_datasets/', level=globs.LEVEL.globals)
root = globs['datasets']

datasets = {
    # 'cifar10': cifar10,
}  # type:Dict[str,Callable[[str],Tuple[llist,llist]]]

split_map = {
    'train': 'trn',
    'eval': 'val',
    'test': 'tst',
}


@regist_func(datasets)
def mscoco(split='train', set_root='MSCOCO'):
    """
    加载流程：
        根据 public_split 加载相应的 json
        根据 json 生成 region_id 和 hdf5 file 的对应关系
        一个 region_id 对应一个 sample
    :param split:
    :return:
    """
    _split_key = split_map.get(split, split)

    _coco_root = os.path.abspath(os.path.join(root, set_root))
    _region_root = os.path.join(_coco_root, 'annotation/controllable/regionfiles')
    _split_root = os.path.join(_coco_root, 'annotation/controllable/public_split')

    _hdf5_root = os.path.join(_coco_root, 'ordered_feature/SA/X_101_32x8d/objrels')

    region_file = os.path.join(_region_root, f"{_split_key}_names.npy")
    split_file = os.path.join(_split_root, f"{_split_key}_names.npy")
    mp_file = os.path.join(_coco_root, 'ordered_feature/MP/resnet101.ctrl/', f'{_split_key}_ft.npy')
    # assert os.path.exists(region_file), "{} not exists".format(region_file)

    _ordered_split_files = np.load(split_file)  # 该顺序和 mp_fts 的顺序一致
    _imageid_mpid_map = {
        x[:-4].split("_")[-1]: i
        for i, x in enumerate(_ordered_split_files)
    }

    xss = np.load(region_file)
    mp_fts = np.load(mp_file)

    region_ids = []
    json_fs = []
    hdf5_fs = []

    mp_ids = []
    for rel_path, region_id in xss:  # type:str,str
        image_id = rel_path.split('_')[-1]

        region_ids.append(region_id)
        json_fs.append(os.path.join(_region_root, f'{rel_path}.json'))

        rel_path2 = rel_path.replace('/', '_')
        # key = train2014_COCO_train2014_000000357413.jpg
        hdf5_file = os.path.join(_hdf5_root, f'{rel_path2}.jpg.hdf5', )
        if not os.path.exists(hdf5_file):
            hdf5_file = os.path.join(_hdf5_root, '../o2/objrels', f'{rel_path2}.jpg.hdf5', )

        hdf5_fs.append(hdf5_file)

        mp_ids.append(_imageid_mpid_map[image_id])

    # 每个 region id 属于一个 image id ，每个 image id 对应 一个下标，该下标对应到 mp_fts 中的一个 feature

    return region_ids, json_fs, hdf5_fs, mp_ids, mp_fts
