import numpy as np
from matplotlib import pyplot as plt

import mathutil
from dataset import Dataset


class EndecDataset(Dataset):
    def __init__(self, name, mode_enc, mode_dec):
        self.mode_enc = mode_enc
        self.mode_dec = mode_dec
        super(EndecDataset, self).__init__(name, mode_dec)

    def set_train_mode(self, mode):
        if mode == 'both':
            self.mode = self.mode_dec
            self.tr_xs, self.va_xs, self.te_xs = self.data[0]
            self.tr_ys, self.va_ys, self.te_ys = self.data[2]
        elif mode == 'encoder':
            self.mode = self.mode_enc
            self.tr_xs, self.va_xs, self.te_xs = self.data[0]
            self.tr_ys, self.va_ys, self.te_ys = self.data[1]
        elif mode == 'decoder':
            self.mode = self.mode_dec
            self.tr_xs, self.va_xs, self.te_xs = self.data[1]
            self.tr_ys, self.va_ys, self.te_ys = self.data[2]

    def shuffle_data(self, xs, ts, ys, tr_ratio=0.8, va_ratio=0.05):
        data_count = len(xs)

        tr_cnt = int(data_count * tr_ratio / 10) * 10
        va_cnt = int(data_count * va_ratio)
        te_cnt = data_count - (tr_cnt + va_cnt)

        tr_from, tr_to = 0, tr_cnt
        va_from, va_to = tr_cnt, tr_cnt + va_cnt
        te_from, te_to = tr_cnt + va_cnt, data_count

        indices = np.arange(data_count)
        np.random.shuffle(indices)

        idx_tr = indices[tr_from:tr_to]
        idx_va = indices[va_from:va_to]
        idx_te = indices[te_from:te_to]

        data_xs = [xs[idx_tr], xs[idx_va], xs[idx_te]]
        data_ts = [ts[idx_tr], ts[idx_va], ts[idx_te]]
        data_ys = [ys[idx_tr], ys[idx_va], ys[idx_te]]

        self.data = [data_xs, data_ts, data_ys]

        self.input_shape = xs[0].shape
        self.output_shape = ys[0].shape

        return idx_tr, idx_va, idx_te

    def forward_postproc(self, output, y):
        if self.mode != 'seqselect':
            return super(EndecDataset, self).forward_postproc(output, y)
        size = y.shape[-1]
        y_flat = y[:, 1:, :].reshape(-1, size)
        output_flat = output[:, 1:, :].reshape(-1, size)
        entropy = mathutil.softmax_cross_entropy_with_logits(y_flat, output_flat)
        loss = np.mean(entropy)
        aux = [y_flat, output_flat, y]

        return loss, aux

    def backprop_postproc(self, G_loss, aux):
        if self.mode != 'seqselect':
            return super(EndecDataset, self).backprop_postproc(G_loss, aux)

        y_flat, output_flat, y = aux
        shape = output_flat.shape

        g_loss_entropy = np.ones(shape) / np.prod(shape)
        g_entropy_output = mathutil.softmax_cross_entropy_with_logits_derv(y_flat, output_flat)

        G_entropy = g_loss_entropy * G_loss
        G_output_flat = g_entropy_output * G_entropy

        mb_size, timesteps, timefeat = y.shape

        G_output = np.zeros(y.shape)
        G_output[:, 0, :] = y[:, 0, :]
        G_output[:, 1:, :] = G_output_flat.reshape([mb_size, timesteps - 1, timefeat])

        return G_output

    def eval_accuracy(self, x, y, output):
        if self.mode != 'seqselect':
            return super(EndecDataset, self).eval_accuracy(x, y, output)

        size = y.shape[-1]
        y_flat = y[:, 1:, :].reshape(-1, size)
        output_flat = output[:, 1:, :].reshape(-1, size)

        acc0 = super(EndecDataset, self).eval_accuracy(x, y_flat, output_flat, 'select')

        estimate = np.argmax(output[:, 1:, :], axis=2)
        answer = np.argmax(y[:, 1:, :], axis=2)
        correct = np.equal(estimate, answer)
        hap = np.sum(correct, axis=1)
        all_correct = np.equal(hap, y.shape[1] - 1)
        acc1 = np.mean(all_correct)

        return [acc0, acc1]

    def train_prt_result(self, epoch, costs, accs, acc, time1, time2):
        if self.mode != 'seqselect':
            return super(EndecDataset, self).train_prt_result(epoch, costs, accs, acc, time1, time2)

        acc_pair = np.mean(accs, axis=0)
        print('Epoch {}: cost={:5.3f}, accuracy={:5.3f}+{:5.3f}/{:5.3f}+{:5.3f} ({}/{} secs)'.format(epoch,
               np.mean(costs), acc_pair[0], acc_pair[1], acc[0], acc[1], time1, time2))

    def test_prt_result(self, name, acc, time):
        if self.mode != 'seqselect':
            return super(EndecDataset, self).test_prt_result(name, acc, time)

        print('Model {} test report: accuracy = {:5.3f}+{:5.3f}, ({} secs)\n'.format(self.name, acc[0], acc[1], time))


class MnistEngDataset(EndecDataset):
    def __init__(self):
        super(MnistEngDataset, self).__init__('mnist_eng', 'select', 'seqselect')

        images, labels = self.load_data()
        digits, words = self.set_captions(labels)

        self.shuffle_data(images, digits, words)

    def set_captions(self, ys):
        self.target_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

        words = np.zeros([10, 7, 27])
        words[:, 0, 0] = 6
        words[:, 1:, 0] = 1.0

        for n in range(10):
            word = self.target_names[n]
            for m in range(len(word)):
                alpha_idx = ord(word[m]) - 96
                words[n, m + 1, :] = np.eye(27)[alpha_idx]

        captions = words[ys]
        digits = np.eye(10)[ys]

        return digits, captions

    def visualize(self, xs, code, estimates, answers):
        self.dump_image_data(xs)
        for n in range(len(xs)):
            astr = self.eng_prob_to_caption(answers[n])
            estr = self.eng_prob_to_caption(estimates[n])
            cstr = np.argmax(code[n])
            print('{} : {}(code: {})'.format(astr, estr, cstr))

    def dump_image_data(self, images):
        show_cnt = len(images)
        fig, axes = plt.subplots(1, show_cnt, figsize=(5, 5))
        for n in range(show_cnt):
            plt.subplot(1, show_cnt, n + 1)
            plt.imshow(images[n].reshape(28, 28), cmap='Greys_r')
            plt.axis('off')
        plt.draw()
        plt.show()

    def eng_prob_to_caption(self, probs):
        word = ''
        idxs = np.argmax(probs[1:, :], axis=1)
        for n in range(int(probs[0, 0])):
            if idxs[n] == 0:
                word = word + '#'
            else:
                word = word + chr(idxs[n] + 96)
        return word

    def load_data(self):
        tr_x_path = '../ch13/mnist/train-images-idx3-ubyte'
        tr_y_path = '../ch13/mnist/train-labels-idx1-ubyte'

        images = np.fromfile(tr_x_path, dtype='uint8')[16:]
        labels = np.fromfile(tr_y_path, dtype='uint8')[8:]

        return images.reshape([-1, 28 * 28]), labels


class MnistKorDataset(EndecDataset):
    def __init__(self, cnt):
        super(MnistKorDataset, self). \
            __init__('mnist_kor', 'regression', 'seqselect')

        self.alphabet = ['#', '영', '일', '이', '삼', '사', '오', \
                         '육', '칠', '팔', '구', '십', '백', '천']

        nums = np.arange(0, 100000) % (10 ** cnt)

        images, labels = self.load_data()

        self.images = self.conv_num_to_images(nums, images, labels, cnt)
        self.onehots = self.gen_caption_onehots(nums, cnt)

        self.shuffle_data(self.images, nums.reshape([-1, 1]), self.onehots)

    def conv_num_to_images(self, nums, xs, ys, cnt):
        data_cnt, xs_cnt = len(nums), len(xs)
        images = np.zeros([data_cnt, cnt + 1, 28 * 28])
        images[:, 0, 0] = cnt
        for n in range(data_cnt):
            digits = nums[n]
            for m in range(cnt, 0, -1):
                digit = digits % 10
                digits = digits // 10
                while True:
                    k = np.random.randint(xs_cnt)
                    if ys[k] == digit:
                        images[n, m, :] = xs[k]
                        break

        return images.reshape([data_cnt, cnt + 1, 28, 28, 1])

    def gen_caption_onehots(self, nums, cnt):
        data_cnt, alpha_cnt = 10 ** cnt, len(self.alphabet)
        onehots = np.zeros([data_cnt, cnt * 2 + 1, alpha_cnt])
        onehots[:, 0, 0] = cnt * 2
        onehots[:, 1:, 0] = 1.0
        for n in range(data_cnt):
            caption = self.gen_caption(n, cnt)
            onehots[n, 1:, :] = self.conv_to_onehot(caption + '#', cnt)

        return onehots[nums]

    def gen_caption(self, num, cnt):
        if num == 0: return '영'
        tousands, hundreds, tens, ones = '', '', '', ''
        if cnt >= 4: tousands = self.gen_digit_str(num // 1000, '천')
        if cnt >= 3: hundreds = self.gen_digit_str(num // 100 % 10, '백')
        if cnt >= 2: tens = self.gen_digit_str(num // 10 % 10, '십')
        if cnt >= 1: ones = self.gen_digit_str(num % 10, '')
        return tousands + hundreds + tens + ones

    def gen_digit_str(self, digit, unit):
        if digit == 0: return ''
        if digit == 1 and unit != '': return unit
        return self.alphabet[digit + 1] + unit

    def conv_to_onehot(self, caption, cnt):
        alpha_cnt = len(self.alphabet)
        onehots = np.zeros([cnt * 2, alpha_cnt])
        for n in range(len(caption)):
            idx = self.alphabet.index(caption[n])
            onehots[n] = np.eye(alpha_cnt)[idx]
        return onehots

    def visualize(self, xs, code, estimates, answers):
        for n in range(len(xs)):
            self.dump_image_data(xs[n, 1:])
            astr = self.prob_to_caption(answers[n])
            estr = self.prob_to_caption(estimates[n])
            print('{} : {}'.format(astr, estr))

    def prob_to_caption(self, probs):
        word = ''
        idxs = np.argmax(probs[1:, :], axis=1)
        for n in range(int(probs[0, 0])):
            word = word + self.alphabet[idxs[n]]
        return word

    def load_data(self):
        tr_x_path = '../ch13/mnist/train-images-idx3-ubyte'
        tr_y_path = '../ch13/mnist/train-labels-idx1-ubyte'

        images = np.fromfile(tr_x_path, dtype='uint8')[16:]
        labels = np.fromfile(tr_y_path, dtype='uint8')[8:]

        return images.reshape([-1, 28 * 28]), labels

    def dump_image_data(self, images):
        show_cnt = len(images)
        fig, axes = plt.subplots(1, show_cnt, figsize=(5, 5))
        for n in range(show_cnt):
            plt.subplot(1, show_cnt, n + 1)
            plt.imshow(images[n].reshape(28, 28), cmap='Greys_r')
            plt.axis('off')
        plt.draw()
        plt.show()