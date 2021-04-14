from dataset_automata import AutomataDataset
from rnn_basic_model import RnnBasicModel

ad = AutomataDataset()

am_4 = RnnBasicModel('am_4', ad, ['rnn', {'recur_size': 4, 'outseq': False}])
am_16 = RnnBasicModel('am_16', ad, ['rnn', {'recur_size': 16, 'outseq': False}])
am_64 = RnnBasicModel('am_64', ad, ['rnn', {'recur_size': 64, 'outseq': False}])

# am_4.exec_all(epoch_count=10, report=1)
# am_16.exec_all(epoch_count=10, report=1)
am_64.exec_all(epoch_count=10, report=1)
