import kagglehub

def get_dataset_path():
    path = kagglehub.dataset_download("aryaminus/electronic-components")
    return path