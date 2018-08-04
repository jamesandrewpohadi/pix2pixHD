import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

def CreateDatasetMirror(opt):
    dataset = None
    from data.aligned_dataset import AlignedDatasetMirror
    dataset = AlignedDatasetMirror()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

def CreateOneDataset(opt, image_label_path, image_inst_path):
    dataset = None
    from data.aligned_dataset import OneDataset
    dataset = OneDataset(image_label_path, image_inst_path)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

class CustomDatasetDataLoaderMirror(CustomDatasetDataLoader):
    def name(self):
        return 'CustomDatasetDataLoaderMirror'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDatasetMirror(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

class OneDatasetDataLoader(BaseDataLoader):
    
    def name(self):
        return 'OneDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)

    def load_data(self, image_label_path, image_inst_path):
        self.dataset = CreateOneDataset(self.opt, image_label_path, image_inst_path)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.nThreads))
        return self.dataloader
