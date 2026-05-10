import kagglehub

def get_dataset_path():
    path = kagglehub.dataset_download("julioazancort/basic-electronic-components")
    return path