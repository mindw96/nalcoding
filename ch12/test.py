from dataset_videoshot import VideoShotDataset
from rnn_ext_model import RnnExtModel

vsd = VideoShotDataset(['AStarIsBorn1937.mp4'])
print(vsd)

conf1 = \
    [['seqwrap', ['avg', {'stride': 30}],
      ['conv', {'ksize': 3, 'chn': 12}],
      ['full', {'width': 16}]],
     ['lstm', {'recur_size': 8}]]

vsm1 = RnnExtModel('vsm1', vsd, conf1)
vsm1.exec_all(epoch_count=10, report=2, show_cnt=3)

conf2 = \
    [['seqwrap', ['avg', {'stride': 30}],
      ['conv', {'ksize': 3, 'chn': 12}],
      ['full', {'width': 16}]],
     ['lstm', {'recur_size': 8}],
     ['lstm', {'recur_size': 4}]]

vsm2 = RnnExtModel('vsm2', vsd, conf2)
vsm2.exec_all(epoch_count=10, report=2, show_cnt=3)

conf3 = \
    [['seqwrap', ['conv', {'ksize': 1, 'chn': 4}],
      ['max', {'stride': 2}],
      ['conv', {'ksize': 1, 'chn': 8}],
      ['max', {'stride': 3}],
      ['conv', {'ksize': 1, 'chn': 16}],
      ['avg', {'stride': 5}],
      ['full', {'width': 12}]],
     ['lstm', {'recur_size': 8}]]

vsm3 = RnnExtModel('vsm3', vsd, conf3)
vsm3.exec_all(epoch_count=10, report=2, show_cnt=3)

vsm1.exec_all(epoch_count=40, report=20, show_cnt=0)
vsm2.exec_all(epoch_count=40, report=20, show_cnt=0)
vsm3.exec_all(epoch_count=40, report=20, show_cnt=0)
