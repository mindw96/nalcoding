from ch13.autoencoder import Autoencoder
from ch13.dataset_auto_mnist import MnistAutoDataset

mset_all = MnistAutoDataset(train_ratio=1.00)
mset_1p = MnistAutoDataset(train_ratio=0.01)

conf_mlp = [['full', {'width': 10}]]

# mnist_mlp_all = RnnExtModel('mnist_mlp_all', mset_all, conf_mlp)
# mnist_mlp_all.exec_all(epoch_count=10, report=1)

# mnist_mlp_1p = RnnExtModel('mnist_mlp_1p', mset_1p, conf_mlp)
# mnist_mlp_1p.exec_all(epoch_count=10, report=1)

conf_auto = {
    'encoder': [['full', {'width': 10}]],
    'decoder': [['full', {'width': 784}]],
    'supervised': [['full', {'width': 10}]]
}

# mnist_auto_1 = Autoencoder('mnist_auto_1', mset_1p, conf_auto)
# mnist_auto_1.autoencode(epoch_count=10, report=1)
# mnist_auto_1.exec_all(epoch_count=10, report=1)

# mnist_auto_all = Autoencoder('mnist_auto_all', mset_all, conf_auto)
# mnist_auto_all.autoencode(epoch_count=10, report=1)
# mnist_auto_all.exec_all(epoch_count=10, report=1)

# mnist_auto_fix = Autoencoder('mnist_auto_fix', mset_1p, conf_auto, fix_encoder=True)
# mnist_auto_fix.autoencode(epoch_count=10, report=1)
# mnist_auto_fix.exec_all(epoch_count=10, report=1)

conf_auto_2 = {
    'encoder': [['full', {'width':64}], ['full', {'width':10}]],
    'decoder': [['full', {'width':64}], ['full', {'width':784}]],
    'supervised': [['full', {'width':10}]]
}

mnist_auto_2 = Autoencoder('mnist_auto_2', mset_all, conf_auto_2)
mnist_auto_2.autoencode(epoch_count=50, report=1)
mnist_auto_2.exec_all(epoch_count=10, report=1)

conf_hash_1 = {
    'encoder': [['full', {'width':10, 'actfunc':'sigmoid'}]],
    'decoder': [['full', {'width':784}]],
    'supervised': []
}

# mnist_hash_1 = Autoencoder('mnist_hash_1', mset_1p, conf_hash_1)
# mnist_hash_1.autoencode(epoch_count=10, report=1)
# mnist_hash_1.semantic_hasing_index()
# mnist_hash_1.semantic_hasing_search()

