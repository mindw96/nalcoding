from ch15.dataset_gan import GanDatasetPicture, GanDatasetMnist
from ch15.gan import Gan

dset_pic_gogh = GanDatasetPicture('gogh.jpg')
dset_pic_jungsun = GanDatasetPicture('jungsun.jpg')
print(dset_pic_gogh)
print(dset_pic_jungsun)

conf_pic = {
    'seed_shape': [16],
    'generator': [['full', {'width': 64}],
                  ['full', {'width': 32 * 32 * 3, 'actfunc': 'sigmoid'}]],
    'discriminor': [['full', {'width': 64}],
                    ['full', {'width': 1, 'actfunc': 'none'}]]
}

# gan_pic_gogh = Gan("gan_pic_gogh", dset_pic_gogh, conf_pic)
# gan_pic_gogh.exec_all(epoch_count=100, report=1)

# gan_pic_jungsun = Gan("gan_pic_jungsun", dset_pic_jungsun, conf_pic)
# gan_pic_jungsun.exec_all(epoch_count=100, report=1)

dset_gan_mnist_full = GanDatasetMnist('dset_gan_mnist_full', 4000)
dset_gan_mnist_68 = GanDatasetMnist('dset_gan_mnist_68', 2000, [6, 8])
dset_gan_mnist_8 = GanDatasetMnist('dset_gan_mnist_8', 1000, [8])
print(dset_gan_mnist_full)
print(dset_gan_mnist_68)
print(dset_gan_mnist_8)

conf_gan_mnist = {
    'seed_shape':  [16],
    'generator':   [['full', {'width':64}],
                    ['full', {'width':784, 'actfunc':'tanh'}]],
    'discriminor': [['full', {'width':64}],
                    ['full', {'width':1, 'actfunc':'none'}]]
}

# gan_mnist_68 = Gan("gan_mnist_68", dset_gan_mnist_68, conf_gan_mnist)
# gan_mnist_68.exec_all(epoch_count=100, report=1)

# gan_mnist_no_adam = Gan("gan_mnist_no_adam", dset_gan_mnist_8, conf_gan_mnist)
# gan_mnist_no_adam.use_adam = False
# gan_mnist_no_adam.exec_all(epoch_count=100, report=1)

# gan_mnist_adam = Gan("gan_mnist_adam", dset_gan_mnist_8, conf_gan_mnist)
# gan_mnist_adam.use_adam = True
# gan_mnist_adam.exec_all(epoch_count=100, report=1)

gan_mnist_full = Gan("gan_mnist_full", dset_gan_mnist_full, conf_gan_mnist)
gan_mnist_full.exec_all(epoch_count=100, report=1)