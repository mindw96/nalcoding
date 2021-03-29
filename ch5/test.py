import mlp_model
import flowerdataset


fd = flowerdataset.FlowersDataset()
fm = mlp_model.MlpModel('flowers_model_1', fd, [50, 25, 10])
fm.exec_all(epoch_count=100, report=10)
