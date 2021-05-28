import numpy as np

from ch13.autoencoder import Autoencoder
from ch14.encoder_decoder import EncoderDecoder


class Gan(EncoderDecoder):
    def init_parameters(self, hconfigs):
        gconf = hconfigs['generator']
        dconf = hconfigs['discriminor']

        if not isinstance(gconf[0], list):
            gconf = [gconf]
        if not isinstance(dconf[0], list):
            dconf = [dconf]

        self.seed_shape = hconfigs['seed_shape']
        input_shape = self.dataset.input_shape

        pmg, gen_shape = self.build_subnet(gconf, self.seed_shape)
        pmd, bin_shape = self.build_subnet(dconf, input_shape)

        assert tuple(gen_shape) == tuple(input_shape)
        assert tuple(bin_shape) == tuple([1])

        self.gconfigs, self.dconfigs = gconf, dconf
        self.pm_gen, self.pm_dis = pmg, pmd

        self.seqout = False
        self.pm_output = None

    def train_step(self, x, y):
        self.is_training = True

        d_loss = self.train_discriminor(x)
        g_loss = self.train_generator(len(x))

        self.is_training = False

        return [d_loss, g_loss], 0

    def train_discriminor(self, real_x):
        mb_size = len(real_x)

        fake_x, _ = self.forward_generator(mb_size)

        mixed_x = np.vstack([real_x, fake_x])
        output, aux_dis = self.forward_discriminor(mixed_x)

        y = np.zeros([2 * mb_size, 1])
        y[0:mb_size, 0] = 1.0

        d_loss, aux_pp = self.forward_postproc(output, y)

        G_loss = 1.0
        G_output = self.backprop_postproc(G_loss, aux_pp)
        self.backprop_discriminor(G_output, aux_dis)

        return d_loss

    def train_generator(self, mb_size):
        fake_x, aux_gen = self.forward_generator(mb_size)

        output, aux_dis = self.forward_discriminor(fake_x)
        y = np.ones([mb_size, 1])

        g_loss, aux_pp = self.forward_postproc(output, y)

        G_loss = 1.0
        G_output = self.backprop_postproc(G_loss, aux_pp)

        self.is_training = False
        G_fake_x = self.backprop_discriminor(G_output, aux_dis)
        self.is_training = True
        self.backprop_generator(G_fake_x, aux_gen)

        return g_loss

    def forward_discriminor(self, x):
        hidden = x
        aux_dis = []

        for n, hconfig in enumerate(self.dconfigs):
            hidden, aux = self.forward_layer(hidden, hconfig, self.pm_dis[n])
            aux_dis.append(aux)

        return hidden, aux_dis

    def backprop_discriminor(self, G_hidden, aux_dis):
        for n in reversed(range(len(self.dconfigs))):
            hconfig, pm, aux = self.dconfigs[n], self.pm_dis[n], aux_dis[n]
            G_hidden = self.backprop_layer(G_hidden, hconfig, pm, aux)
        return G_hidden

    def forward_generator(self, mb_size):
        hidden = np.random.uniform(-1.0, 1.0, size=[mb_size] + self.seed_shape)
        aux_gen = []

        for n, hconfig in enumerate(self.gconfigs):
            hidden, aux = self.forward_layer(hidden, hconfig, self.pm_gen[n])
            aux_gen.append(aux)

        return hidden, aux_gen

    def backprop_generator(self, G_hidden, aux_gen):
        for n in reversed(range(len(self.gconfigs))):
            hconfig, pm, aux = self.gconfigs[n], self.pm_gen[n], aux_gen[n]
            G_hidden = self.backprop_layer(G_hidden, hconfig, pm, aux)
        return G_hidden

    def update_param(self, pm, key, G_key):
        if not self.is_training: return

        super(Gan, self).update_param(pm, key, G_key)

    def eval_accuracy(self, real_x, y, output=None):
        mb_size = len(real_x)

        fake_x, _ = self.forward_generator(mb_size)
        mixed_x = np.vstack([real_x, fake_x])
        output, aux_dis = self.forward_discriminor(mixed_x)

        y = np.zeros([2 * mb_size, 1])
        y[0:mb_size] = 1.0
        d_acc = self.dataset.eval_accuracy(mixed_x, y, output)

        fake_x, _ = self.forward_generator(mb_size)
        output, aux_dis = self.forward_discriminor(fake_x)

        y = np.ones([mb_size, 1])
        g_acc = self.dataset.eval_accuracy(fake_x, y, output)

        return [d_acc, g_acc]

    def visualize(self, num):
        real_x, _ = self.dataset.get_visualize_data(num)
        fake_x, _ = self.forward_generator(num)
        self.dataset.visualize(np.vstack([real_x, fake_x]))
