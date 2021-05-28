from ch05.flowerdataset import FlowersDataset
from cnn_reg_model import CnnRegModel

fd = FlowersDataset([96, 96], [96, 96, 3])

# fm1 = CnnRegModel('flowers_model_1', fd, [30, 10])
# fm1.exec_all(epoch_count=10, report=1, show_params=True)

# fm2 = CnnRegModel('flowers_model_2', fd, [30, 10], l2_decay=0.1)
# fm2.exec_all(epoch_count=10, report=1, show_cnt=3, show_params=True)

# fm3 = CnnRegModel('flowers_model_3', fd, [30, 10], l1_decay=0.01)
# fm3.exec_all(epoch_count=10, report=1, show_cnt=3, show_params=True)

cnn_og = [['conv', {'ksize': 3, 'chn': 6}],
               ['max', {'stride': 2}],
               ['conv', {'ksize': 3, 'chn': 12}],
               ['max', {'stride': 2}],
               ['conv', {'ksize': 3, 'chn': 24}],
               ['avg', {'stride': 3}]]

cnn_dropout = [['conv', {'ksize': 3, 'chn': 6}],
               ['max', {'stride': 2}],
               ['dropout', {'keep_prob': 0.6}],
               ['conv', {'ksize': 3, 'chn': 12}],
               ['max', {'stride': 2}],
               ['dropout', {'keep_prob': 0.6}],
               ['conv', {'ksize': 3, 'chn': 24}],
               ['avg', {'stride': 3}],
               ['dropout', {'keep_prob': 0.6}]]

cnn_batch_normal = [['conv', {'ksize': 3, 'chn': 6}],
                    ['max', {'stride': 2}],
                    ['batch_normal'],
                    ['conv', {'ksize': 3, 'chn': 12}],
                    ['max', {'stride': 2}],
                    ['batch_normal'],
                    ['conv', {'ksize': 3, 'chn': 24}],
                    ['avg', {'stride': 3}]]

noise_std = 0.01
cnn_noise = [['noise', {'type': 'normal', 'mean': 0, 'std': noise_std}],
             ['conv', {'ksize': 3, 'chn': 6}],
             ['max', {'stride': 2}],
             ['noise', {'type': 'normal', 'mean': 0, 'std': noise_std}],
             ['conv', {'ksize': 3, 'chn': 12}],
             ['max', {'stride': 2}],
             ['noise', {'type': 'normal', 'mean': 0, 'std': noise_std}],
             ['conv', {'ksize': 3, 'chn': 24}],
             ['avg', {'stride': 3}]]

fcnn1 = CnnRegModel('flowers_mlp_noise', fd, [['noise', {'type': 'normal', 'mean': 0, 'std': noise_std}], 30,['noise', {'type': 'normal', 'mean': 0, 'std': noise_std}], 10, ['noise', {'type': 'normal', 'mean': 0, 'std': noise_std}], 3])
fcnn1.exec_all(epoch_count=10, report=1, show_params=True)
