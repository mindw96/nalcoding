from ch5.flowerdataset import FlowersDataset
from ch6.dataset_office31 import Office31Dataset
from cnn_basic_model import CnnBasicModel

fd = FlowersDataset([96, 96], [96, 96, 3])
od = Office31Dataset([96, 96], [96, 96, 3])

# fm1 = CnnBasicModel('flowers_model_1', fd, [30,10])
# fm1.exec_all(epoch_count=10, report=1)

# fm2 = CnnBasicModel('flowrs_model_2', fd, [['full', {'width':30}]], ['full', {'width':10}])
# fm2.use_adam = False
# fm2.exec_all(epoch_count=10, report=1)

# fm3 = CnnBasicModel('flowers_model_3', fd,
#                     [['conv', {'ksize': 5, 'chn': 6}], ['max', {'stride': 4}], ['conv', {'ksize': 3, 'chn': 12}],
#                      ['avg', {'stride': 2}]], True)
# fm3.exec_all(epoch_count=10, report=1)

fm4 = CnnBasicModel('flowers_model_4', fd,
                    [['conv', {'ksize': 3, 'chn': 6}],
                     ['max', {'stride': 2}],
                     ['conv', {'ksize': 3, 'chn': 12}],
                     ['max', {'stride': 2}],
                     ['conv', {'ksize': 3, 'chn': 24}],
                     ['avg', {'stride': 3}]], True)
fm4.exec_all(epoch_count=10, report=1)
