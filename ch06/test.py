from mlp_model import MlpModel
from adam_model import AdamModel
from dataset_office31 import Office31Dataset

od = Office31Dataset()

# om1 = MlpModel('office_model_1', od, [10])
# om1.exec_all(epoch_count=20, report=10)

# om2 = AdamModel('office_model_2', od, [64, 32, 10])
# om2.exec_all(epoch_count=50,report=10, learning_rate=0.0001)

om3 = AdamModel('office_model_3', od, [128, 64, 32, 10])
om3.use_adam = True
om3.exec_all(epoch_count=100, report=10, learning_rate=0.0001)