
def CreateDataLoader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader

def CreateDataLoaderMirror(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader

def CreateOneDataLoader(opt):
    from data.custom_dataset_data_loader import OneDatasetDataLoader
    data_loader = OneDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
