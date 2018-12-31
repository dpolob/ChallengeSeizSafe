from torch.utils.data import Dataset

class Data3DSet(Dataset):
    """
        Coge el dataset en bruto y le aplica las siguientes operaciones
    """
    def __init__(self, data):
        super(Data3DSet, self).__init__()
        self.data = data
        print(self.data.shape)

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, index):
        return self.data[:, index, :]

