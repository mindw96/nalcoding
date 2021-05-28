from ch14.dataset_endec_mnist import MnistEngDataset
from ch14.dataset_endec_mnist import MnistKorDataset
from ch14.encoder_decoder import EncoderDecoder

mnist_eng = MnistEngDataset()

conf_eng1 = {
    'encoder': [['full', {'width': 10}]],
    'decoder': [['lstm', {'recur_size': 32, 'inseq': False,
                          'outseq': True, 'timesteps': 6}],
                ['seqwrap', ['full', {'width': 27, 'actfunc': 'none'}]]]
}

# encdec_eng1 = EncoderDecoder('encdec_eng1', mnist_eng, conf_eng1)
# encdec_eng1.exec_1_step(epoch_count=10, report=1)

conf_eng2 = {
    'encoder': [['full', {'width': 32}],
                ['batch_normal'],
                ['full', {'width': 10}]],
    'decoder': [['lstm', {'recur_size': 32, 'inseq': False,
                          'outseq': True, 'timesteps': 6}],
                ['seqwrap', ['full', {'width': 27, 'actfunc': 'none'}]]]
}

# encdec_eng2 = EncoderDecoder('encdec_eng2', mnist_eng, conf_eng2)
# encdec_eng2.exec_1_step(epoch_count=10, report=1)

# encdec_eng2_2 = EncoderDecoder('encdec_eng2_2', mnist_eng, conf_eng2)
# encdec_eng2_2.exec_2_steps(epoch_count=10, report=1)

mnist_kor2 = MnistKorDataset(2)

conf_kor2 = {
    'encoder': [['seqwrap', ['full', {'width': 32}]],
                ['batch_normal'],
                ['full', {'width': 10}]],
    'decoder': [['lstm', {'recur_size': 32, 'inseq': False,
                          'outseq': True, 'timesteps': 4}],
                ['seqwrap', ['full', {'width': 14, 'actfunc': 'none'}]]]
}

# encdec_kor2 = EncoderDecoder('encdec_kor2', mnist_kor2, conf_kor2)
# encdec_kor2.exec_1_step(epoch_count=10, report=1)

mnist_kor3 = MnistKorDataset(3)

conf_kor3 = {
    'encoder': [['seqwrap', ['full', {'width': 32}]],
                ['batch_normal'],
                ['full', {'width': 10}]],
    'decoder': [['lstm', {'recur_size': 32, 'inseq': False,
                          'outseq': True, 'timesteps': 6}],
                ['seqwrap', ['full', {'width': 14, 'actfunc': 'none'}]]]
}

# encdec_kor3 = EncoderDecoder('encdec_kor3', mnist_kor3, conf_kor3)
# encdec_kor3.exec_1_step(epoch_count=10, report=1)

mnist_kor4 = MnistKorDataset(4)

conf_kor4 = {
    'encoder': [['seqwrap', ['full', {'width': 32}]],
                ['batch_normal'],
                ['full', {'width': 10}]],
    'decoder': [['lstm', {'recur_size': 64, 'inseq': False,
                          'outseq': True, 'timesteps': 8}],
                ['seqwrap', ['full', {'width': 14, 'actfunc': 'none'}]]]
}

encdec_kor4 = EncoderDecoder('encdec_kor4', mnist_kor4, conf_kor4)
encdec_kor4.exec_1_step(epoch_count=50, report=1)
