from dataset import Dataset


class DummyDataset(Dataset):
    def __init__(self, name, mode, input_shape, output_shape):
        super(DummyDataset, self).__init__(name, mode)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.tr_xs, self.tr_ys = [], []
        self.te_xs, self.te_ys = [], []
