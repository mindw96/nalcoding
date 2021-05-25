import matplotlib.pyplot as plt
import numpy as np

from dataset import Dataset


class AutoencodeDataset(Dataset):
    def __init__(self, name, mode, train_ratio=1.0):
        self.train_ratio = train_ratio
        super(AutoencodeDataset, self).__init__(name, mode)

    def get_autoencode_data(self, batch_size, nth):
        xs, ys = self.get_train_data(batch_size, nth)
        return xs

    @property
    def train_count(self):
        return int(len(self.tr_xs) * self.train_ratio)

    @property
    def autoencode_count(self):
        return len(self.tr_xs)


class MnistAutoDataset(AutoencodeDataset):
    def __init__(self, train_ratio=0.1):
        super(MnistAutoDataset, self).__init__('mnist', 'select', train_ratio)

        tr_x_path = 'mnist/train-images-idx3-ubyte'
        tr_y_path = 'mnist/train-labels-idx1-ubyte'

        xs = np.fromfile(tr_x_path, dtype='uint8')[16:]
        ys = np.fromfile(tr_y_path, dtype='uint8')[8:]

        xs = xs.reshape([-1, 28 * 28])
        ys = np.eye(10)[ys]

        self.shuffle_data(xs, ys)

    def visualize(self, xs, estimates, answers):
        self.dump_text(answers, estimates)
        self.dump_image_data(xs)

    def autoencode_visualize(self, xs, rep, estimates, answers):
        self.dump_text(answers, estimates)
        self.dump_image_data(xs)
        self.dump_image_data(rep)

    def hash_result_visualize(self, images):
        self.dump_image_data(images)

    def dump_text(self, answers, estimates):
        ans = np.argmax(answers, axis=1)
        est = np.argmax(estimates, axis=1)
        print('정답', ans, ' vs. ', '추정', est)

    def dump_image_data(self, images):
        show_cnt = len(images)
        fig, axes = plt.subplots(1, show_cnt, figsize=(show_cnt, 1))

        for n in range(show_cnt):
            plt.subplot(1, show_cnt, n + 1)
            plt.imshow(images[n].reshape(28, 28), cmap='Greys_r')
            plt.axis('off')

        plt.draw()
        plt.show()
