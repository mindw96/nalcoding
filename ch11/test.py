from dataset_sounds import UrbanSoundDataset
from rnn_lstm_model import RnnLstmModel

usd_10_10 = UrbanSoundDataset(10, 10)
usd_10_100 = UrbanSoundDataset(10, 100)

conf_basic = ['rnn', {'recur_size': 20, 'outseq': False}]
conf_lstm = ['lstm', {'recur_size': 20, 'outseq': False}]
conf_state = ['lstm', {'recur_size': 20, 'outseq': False, 'use_state': True}]

us_basic_10_10 = RnnLstmModel('us_basic_10_10', usd_10_10, conf_basic)
us_lstm_10_10 = RnnLstmModel('us_lstm_10_10', usd_10_10, conf_lstm)
us_state_10_10 = RnnLstmModel('us_state_10_10', usd_10_10, conf_state)

us_basic_10_100 = RnnLstmModel('us_basic_10_100', usd_10_100, conf_basic)
us_lstm_10_100 = RnnLstmModel('us_lstm_10_100', usd_10_100, conf_lstm)
us_state_10_100 = RnnLstmModel('us_state_10_100', usd_10_100, conf_state)

# us_basic_10_10.exec_all(epoch_count=10, report=1)
# us_lstm_10_10.exec_all(epoch_count=10, report=1, show_cnt=0)
# us_state_10_10.exec_all(epoch_count=10, report=1, show_cnt=0)

us_state_10_100.exec_all(epoch_count=100, report=10, show_cnt=0)
