import PIL.Image
import numpy as np
from matplotlib import pyplot as plt

from dataset import Dataset


class GanDataset(Dataset):
    def visualize(self, xs):
        show_cnt = len(xs)
        fig, axes = plt.subplots(1, show_cnt, figsize=(show_cnt, 1))

        for n in range(show_cnt):
            plt.subplot(1, show_cnt, n + 1)
            if xs[n].shape[0] == 28 * 28:
                plt.imshow(xs[n].reshape(28, 28), cmap='Greys_r')
            else:
                plt.imshow(xs[n].reshape([32, 32, 3]))
            plt.axis('off')

        plt.draw()
        plt.show()

    def train_prt_result(self, epoch, costs, accs, acc, time1, time2):
        dcost, gcost = np.mean(costs, axis=0)
        dacc, gacc = acc
        print('    Epoch {}: cost={:5.3f}/{:5.3f} acc={:5.3f}/{:5.3f} ({}/{} secs)'.format(epoch, dcost, gcost, dacc,
                                                                                           gacc, time1, time2))

    def test_prt_result(self, name, acc, time):
        dacc, gacc = acc
        print('Model {} test report: accuracy = {:5.3f}/{:5.3f}, ({} secs)\n'.format(name, dacc, gacc, time))


class GanDatasetPicture(GanDataset):
    def __init__(self, fname):
        super(GanDatasetPicture, self).__init__('pic_' + fname, 'binary')

        pic_path = 'pictures/' + fname
        jpgfile = PIL.Image.open(pic_path)
        pixels = np.array(jpgfile)

        hn = pixels.shape[0] // 32
        wn = pixels.shape[1] // 32

        pieces = pixels[0:hn * 32, 0:wn * 32, 0:3]
        pieces = pieces.reshape([hn, 32, wn, 32, 3])
        pieces = pieces.transpose([0, 2, 1, 3, 4])
        pieces = pieces.reshape([-1, 32 * 32 * 3])

        pieces = pieces / 255.0

        self.shuffle_data(pieces, pieces)


class GanDatasetMnist(GanDataset):
    def __init__(self, name, max_cnt=0, nums=None):
        super(GanDatasetMnist, self).__init__(name, 'binary')

        tr_x_path = '../ch13/mnist/train-images-idx3-ubyte'
        tr_y_path = '../ch13/mnist/train-labels-idx1-ubyte'

        images = np.fromfile(tr_x_path, dtype='uint8')[16:]
        labels = np.fromfile(tr_y_path, dtype='uint8')[8:]

        images = images.reshape([-1, 28 * 28])
        images = (images - 127.5) / 127.5

        if max_cnt == 0: max_cnt = len(images)

        if nums is None:
            xs = images[:max_cnt]
        else:
            ids = []
            for n in range(len(images)):
                if labels[n] in nums: ids.append(n)
                if len(ids) >= max_cnt: break
            xs = images[ids]

        self.shuffle_data(xs, xs)
