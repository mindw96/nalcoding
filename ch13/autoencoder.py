import time

import numpy as np

from ch12.rnn_ext_model import RnnExtModel


class Autoencoder(RnnExtModel):
    def __init__(self, name, dataset, hconfigs, show_maps=False, l2_decay=0, l1_decay=0, dump_structure=False,
                 fix_encoder=False):
        self.fix_encoder = fix_encoder
        super(Autoencoder, self).__init__(name, dataset, hconfigs, show_maps, l2_decay, l1_decay, dump_structure)

    def init_parameters(self, hconfigs):
        econf = hconfigs['encoder']
        dconf = hconfigs['decoder']
        hconf = hconfigs['supervised']

        in_shape = self.dataset.input_shape

        pme, code_shape = self.build_subnet(econf, in_shape)
        pmd, represent_shape = self.build_subnet(dconf, code_shape)
        pmh, hidden_shape = self.build_subnet(hconf, code_shape)

        self.econfigs, self.dconfigs, self.hconfigs = econf, dconf, hconf
        self.pm_encoder, self.pm_decoder, self.pm_hiddens = pme, pmd, pmh

        output_cnt = int(np.prod(self.dataset.output_shape))
        self.seqout = False

        if len(hconf) > 0 and self.get_layer_type(hconf[-1]) in ['rnn', 'lstm']:
            if self.get_conf_param(hconf[-1], 'outseq', True):
                self.seqout = True
                hidden_shape = hidden_shape[1:]
                output_cnt = int(np.prod(self.dataset.output_shape[1:]))

        self.pm_output, _ = self.alloc_layer_param(hidden_shape, output_cnt)

    def build_subnet(self, hconfigs, prev_shape):
        pms = []

        for hconfig in hconfigs:
            pm, prev_shape = self.alloc_layer_param(prev_shape, hconfig)
            pms.append(pm)

        return pms, prev_shape

    def autoencode(self, epoch_count=10, batch_size=10, learning_rate=0.001, report=0):
        self.learning_rate = learning_rate

        batch_count = self.dataset.autoencode_count // batch_size
        time1 = time2 = int(time.time())

        if report != 0:
            print('Model {} autoencode started:'.format(self.name))

        for epoch in range(epoch_count):
            costs = []
            accs = []
            self.dataset.shuffle_train_data(batch_size*batch_count)
            for n in range(batch_count):
                trX = self.dataset.get_autoencode_data(batch_size, n)
                cost, acc = self.autoencode_step(trX)
                costs.append(cost)
                accs.append(acc)

            if report > 0 and (epoch+1) % report == 0:
                acc_mean = np.mean(accs)
                time3 = int(time.time())
                tm1, tm2 = time3 - time2, time3 - time1
                self.dataset.train_prt_result(epoch+1, costs,accs, tm1, tm2)
                time2=time3

        tm_total = int(time.time()) - time1
        if report != 0:
            print('Model {} autoencode ended in {} secs'.format(self.name, tm_total))

    def autoencode_step(self, x):
        self.is_training = True

        hidden, aux_encoder, aux_decoder = self.forward_autoencode(x)

        diff = hidden - x
        square = np.square(diff)
        loss = np.mean(square)

        mse = np.mean(np.square(hidden - x))
        accuracy = 1 - np.sqrt(mse) / np.mean(x)

        g_loss_square = np.ones(x.shape) / np.prod(x.shape)
        g_square_diff = 2 * diff
        g_diff_output = 1

        G_loss =1.0
        G_square = g_loss_square * G_loss
        G_diff = g_square_diff * G_square
        G_hidden = g_diff_output * G_diff

        self.backprop_autoencode(G_hidden, aux_encoder, aux_decoder)

        self.is_training = False

        return loss, accuracy
