from thexp import Trainer
from data import delegate
from trainers import GlobalParams
from thexp.contrib.data import splits
from thexp import DatasetBuilder

from data.transforms import ToTensor
from data.dataxy import datasets
from data.collate_fns import convert_batch_sparse_matrix_collate_fn

toTensor = ToTensor()


class DatasetMixin(Trainer):
    def datasets(self, params: GlobalParams):
        raise NotImplementedError()


class BaseSupDatasetMixin(DatasetMixin):
    """Base Supervised Dataset"""

    @staticmethod
    def _build_datasets(datas, params: GlobalParams):
        region_ids, json_fs, hdf5_fs, mp_ids, mp_fts = datas
        asg_del = delegate.ASGLoadDelegate(region_id=region_ids,
                                           json_fs=json_fs, hdf5_fs=hdf5_fs,
                                           mp_fts=mp_fts, mp_ids=mp_ids,
                                           max_attn_len=params.max_attn_len, pixel_reduce=params.pixel_reduce)
        loader = (
            DatasetBuilder()
                .add_delegate(asg_del)
                .zip_mode()
                .DataLoader(batch_size=params.batch_size, collate_fn=convert_batch_sparse_matrix_collate_fn)
        )
        return loader

    def datasets(self, params: GlobalParams):
        dataset_fn = datasets[params.dataset]

        train_dataloader = BaseSupDatasetMixin._build_datasets(dataset_fn('train',
                                                                          'asg2cap/ControllableImageCaption/MSCOCO'),
                                                               params)
        val_dataloader = BaseSupDatasetMixin._build_datasets(dataset_fn('val',
                                                                        'asg2cap/ControllableImageCaption/MSCOCO'),
                                                             params)
        test_dataloader = BaseSupDatasetMixin._build_datasets(dataset_fn('test',
                                                                         'asg2cap/ControllableImageCaption/MSCOCO'),
                                                              params)

        self.regist_databundler(train=train_dataloader,
                                eval=val_dataloader,
                                test=test_dataloader)

        self.to(self.device)
