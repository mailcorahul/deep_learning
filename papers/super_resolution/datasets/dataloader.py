from torch.utils.data import Dataset


class SRCNN_Loader(Dataset):
    """ Data Loader for SRCNN """
    
    def __init__(self, root_dir):
        self.root_dir = root_dir




