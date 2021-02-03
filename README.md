# asg2cap

Reimplement https://github.com/cshizhe/asg2cap

## 数据集构建方式

### MSCOCO

由 npy 文件、json 文件、hdf5 文件三类文件组成。

```
.
├── annotation
│   ├── controllable
│   │   ├── cleaned_tst_region_descriptions.json
│   │   ├── int2word.npy
│   │   ├── public_split # 图片文件名，切分方式 
│   │   │   ├── trn_names.npy 
│   │   │   ├── tst_names.npy
│   │   │   └── val_names.npy
│   │   ├── regionfiles #  object ID  对应的图片，冗余信息，在 json 文件中完全包括了
│   │   │   ├── trn_names.npy
│   │   │   ├── tst_names.npy
│   │   │   └── val_names.npy
│   │   │   ├── train2014 # 每个json 是一个 dict，level1 每个key 是 object ID，对应的 value 是相应anno
│   │   │   │   ├── COCO_train2014_000000000009.json
│   │   │   │   ├── COCO_train2014_000000000025.json
│   │   │   │   ├── COCO_train2014_000000000030.json
│   │   │   ├── val2014
│   │   │   │   ├── COCO_val2014_000000000042.json
│   │   │   │   ├── COCO_val2014_000000000073.json
│   │   │   │   ├── COCO_val2014_000000000074.json
│   │   │   │   ├── COCO_val2014_000000000133.json
│   │   │   │   ├── COCO_val2014_000000000136.json
│   │   │   │   ├── COCO_val2014_000000040011.json
│   │   └── word2int.json
├── dir.txt
├── ordered_feature
│   ├── MP
│   │   └── resnet101.ctrl
│   │       ├── trn_ft.npy
│   │       ├── tst_ft.npy
│   │       └── val_ft.npy
│   └── SA
│       └── X_101_32x8d
│           ├── o2
│           │   └── objrels
│           │       ├── train2014_COCO_train2014_000000000036.jpg.hdf5
│           │       ├── train2014_COCO_train2014_000000579757.jpg.hdf5
│           │       ├── train2014_COCO_train2014_000000579785.jpg.hdf5
│           │       ├── val2014_COCO_val2014_000000000042.jpg.hdf5
│           │       ├── val2014_COCO_val2014_000000000073.jpg.hdf5
│           │       ├── val2014_COCO_val2014_000000000133.jpg.hdf5
│           └── objrels
│               ├── train2014_COCO_train2014_000000000009.jpg.hdf5
│               ├── train2014_COCO_train2014_000000581909.jpg.hdf5
│               ├── train2014_COCO_train2014_000000581921.jpg.hdf5
│               ├── val2014_COCO_val2014_000000000074.jpg.hdf5
│               ├── val2014_COCO_val2014_000000000139.jpg.hdf5
│               └── val2014_COCO_val2014_000000581929.jpg.hdf5
└── results
    
```

### json - objectID 字段

```json
{
  "515385": { // 这一 ID 是 region id，其 value 字段表示的是该 region 的 object 和 relationship
    "objects": [
      {
        "object_id": 1639311, // 指 515385 这个 object 和 1639311 的关系
        "name": "boy",
        "attributes": [
          "small"
        ],
        "x": 245,
        "y": 182,
        "w": 115,
        "h": 173
      },
      {
        "object_id": "",
        ...
      },
      ...
    ],
    "relationships": [
      {
        "relationship_id": 867254,
        "name": "on",
        "subject_id": 1639311,
        "object_id": 1639310
      },
      {
        "relationship_id": ...,
        "name": "on",
        "subject_id": 1639311,
        "object_id": 1639312
      }
    ],
    "phrase": "a small boy on some grass and a frisbee"
  },
  "407180": {
    "objects": [
      {
        "object_id": 1639311,
        "name": "boy",
        "attributes": [
          "small"
        ],
        "x": 245,
        "y": 182,
        "w": 115,
        "h": 173
      },
      {
        "object_id": 1639310,
        "name": "grass",
        "attributes": [],
        "x": 91,
        "y": 158,
        "w": 545,
        "h": 242
      },
      {
        "object_id": 1639312,
        "name": "frisbee",
        "attributes": [],
        "x": 286,
        "y": 236,
        "w": 123,
        "h": 75
      }
    ],
    "relationships": [
      {
        "relationship_id": 867254,
        "name": "on",
        "subject_id": 1639311,
        "object_id": 1639310
      },
      {
        "relationship_id": 867255,
        "name": "on",
        "subject_id": 1639311,
        "object_id": 1639312
      }
    ],
    "phrase": "a small boy on some grass and a frisbee"
  }
}


```

# Sample 构建

每张图的每个区域都会有一个 region_id，根据 region_id 获取得到 region_graph （key=region_id 的 value）和 region_caption（value 下的 phrase 字段）

对每一个region_graph，预训练好的目标检测模型会给出多个 ROI/box（区域坐标 x,y,w,h），以及每个 box 的 feature。

 - attn_fn ：每个 box 的 feature 合在一起
 - 


# 数据字段解释

数据一共有三大部分，包括：

- public_split，存储 train/val/test 的文件名列表，以 npy 格式进行组织
- region_file，存储

## TODOs

两个