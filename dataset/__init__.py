from dataset.dataset_3dpw import create_datasets as create_3dpw_datasets, create_full_datasets as create_3dpwfull_datasets, create_ablation_datasets
from dataset.dataset_expi import create_datasets as create_expi_datasets


def create_datasets(config, **args):

    if config.dataset == "3dpw":
        return create_3dpw_datasets(config, **args)

    elif config.dataset == "3dpwfull":
        return create_3dpwfull_datasets(config, **args)
    
    elif config.dataset == "expi":
        return create_expi_datasets(config, **args)
    
    elif config.dataset.startswith("3dpw_ablation"):
        return create_ablation_datasets(config, **args)
