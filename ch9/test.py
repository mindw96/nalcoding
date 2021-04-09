from ch5.flowerdataset import FlowersDataset
from ch9.cnn_ext_model import CnnExtModel
from ch9.dataset_dummy import DummyDataset

imagenet = DummyDataset('imagenet', 'select', [299, 299, 3], 200)

CnnExtModel.set_macro('v3_preproc',
                      ['serial',
                       ['conv', {'ksize': 3, 'stride': 2, 'chn': 32, 'padding': 'VALID'}],
                       ['conv', {'ksize': 3, 'chn': 32, 'padding': 'VALID'}],
                       ['conv', {'ksize': 3, 'chn': 64, 'padding': 'SAME'}],
                       ['max', {'ksize': 3, 'stride': 2, 'padding': 'VALID'}],
                       ['conv', {'ksize': 3, 'chn': 80, 'padding': 'VALID'}],
                       ['conv', {'ksize': 3, 'chn': 192, 'padding': 'VALID'}],
                       ['conv', {'ksize': 3, 'chn': 2, 'padding': 'VALID'}]])

CnnExtModel.set_macro('v3_inception1',
                      ['parallel',
                       ['conv', {'ksize': 1, 'chn': 64}],
                       ['serial',
                        ['conv', {'ksize': 1, 'chn': 48}],
                        ['conv', {'ksize': 5, 'chn': 64}]],
                       ['serial',
                        ['conv', {'ksize': 1, 'chn': 64}],
                        ['conv', {'ksize': 3, 'chn': 96}],
                        ['conv', {'ksize': 3, 'chn': 96}]],
                       ['serial',
                        ['avg', {'ksize': 3, 'stride': 1}],
                        ['conv', {'ksize': 1, 'chn': '#chn'}]]])

CnnExtModel.set_macro('v3_resize1',
                      ['parallel',
                       ['conv', {'ksize': 3, 'stride': 2, 'chn': 384}],
                       ['serial',
                        ['conv', {'ksize': 1, 'chn': 64}],
                        ['conv', {'ksize': 3, 'chn': 96}],
                        ['conv', {'ksize': 3, 'stride': 2, 'chn': 96}]],
                       ['max', {'ksize': 3, 'stride': 2}]])

CnnExtModel.set_macro('v3_inception2',
                      ['parallel',
                       ['conv', {'ksize': 1, 'chn': 192}],
                       ['serial',
                        ['conv', {'ksize': [1, 1], 'chn': '#chn'}],
                        ['conv', {'ksize': [1, 7], 'chn': '#chn'}],
                        ['conv', {'ksize': [7, 1], 'chn': 192}]],
                       ['serial',
                        ['conv', {'ksize': [1, 1], 'chn': '#chn'}],
                        ['conv', {'ksize': [7, 1], 'chn': '#chn'}],
                        ['conv', {'ksize': [1, 7], 'chn': '#chn'}],
                        ['conv', {'ksize': [7, 1], 'chn': '#chn'}],
                        ['conv', {'ksize': [1, 7], 'chn': 192}]],
                       ['serial',
                        ['avg', {'ksize': 3, 'stride': 1}],
                        ['conv', {'ksize': 1, 'chn': 192}]]])

CnnExtModel.set_macro('v3_resize2',
                      ['parallel',
                       ['serial',
                        ['conv', {'ksize': 1, 'chn': 192}],
                        ['conv', {'ksize': 3, 'stride': 2, 'chn': 320}]],
                       ['serial',
                        ['conv', {'ksize': [1, 1], 'chn': 192}],
                        ['conv', {'ksize': [1, 7], 'chn': 192}],
                        ['conv', {'ksize': [7, 1], 'chn': 192}],
                        ['conv', {'ksize': [3, 3], 'stride': [2, 2], 'chn': 192}]],
                       ['max', {'ksize': 3, 'stride': 2}]])

CnnExtModel.set_macro('v3_inception3',
                      ['parallel',
                       ['conv', {'ksize': 1, 'chn': 320}],
                       ['serial',
                        ['conv', {'ksize': [3, 3], 'chn': 384}],
                        ['parallel',
                         ['conv', {'ksize': [1, 3], 'chn': 384}],
                         ['conv', {'ksize': [3, 1], 'chn': 384}]]],
                       ['serial',
                        ['conv', {'ksize': [1, 1], 'chn': 448}],
                        ['conv', {'ksize': [3, 3], 'chn': 384}],
                        ['parallel',
                         ['conv', {'ksize': [1, 3], 'chn': 384}],
                         ['conv', {'ksize': [3, 1], 'chn': 384}]]],
                       ['serial',
                        ['avg', {'ksize': 3, 'stride': 1}],
                        ['conv', {'ksize': 1, 'chn': 192}]]])

CnnExtModel.set_macro('v3_postproc',
                      ['serial',
                       ['avg', {'stride': 8}],
                       ['dropout', {'keep_prob': 0.7}]])

CnnExtModel.set_macro('inception_V3',
                      ['serial',
                       ['custom', {'name': 'v3_preproc'}],
                       ['custom', {'name': 'v3_inception1', 'args': {'#chn': 32}}],
                       ['custom', {'name': 'v3_inception1', 'args': {'#chn': 64}}],
                       ['custom', {'name': 'v3_inception1', 'args': {'#chn': 64}}],
                       ['custom', {'name': 'v3_resize1'}],
                       ['custom', {'name': 'v3_inception2', 'args': {'#chn': 128}}],
                       ['custom', {'name': 'v3_inception2', 'args': {'#chn': 160}}],
                       ['custom', {'name': 'v3_inception2', 'args': {'#chn': 160}}],
                       ['custom', {'name': 'v3_inception2', 'args': {'#chn': 192}}],
                       ['custom', {'name': 'v3_resize2'}],
                       ['custom', {'name': 'v3_inception3'}],
                       ['custom', {'name': 'v3_inception3'}],
                       ['custom', {'name': 'v3_postproc'}]])

inception_v3 = CnnExtModel('inception_V3', imagenet, [['custom', {'name': 'inception_V3'}]], dump_structure=True)

fd = FlowersDataset([96, 96], [96, 96, 3])

# 위에서 만든 Inception V3는 GPU 가속을 안하면 처리 시간이 매우 오래 걸리기 때문에 데이터셋에 맞춘 개량 모델을 만든다.
CnnExtModel.set_macro('flower_preproc',
                      ['serial',
                       ['conv', {'ksize': 3, 'stride': 2, 'chn': 6, 'actions': '#act'}]])

CnnExtModel.set_macro('flower_inception1',
                      ['parallel',
                       ['conv', {'ksize': 1, 'chn': 4, 'actions': '#act'}],
                       ['conv', {'ksize': 3, 'chn': 6, 'actions': '#act'}],
                       ['serial',
                        ['conv', {'ksize': 3, 'chn': 6, 'actions': '#act'}],
                        ['conv', {'ksize': 3, 'chn': 6, 'actions': '#act'}]],
                       ['serial',
                        ['avg', {'ksize': 3, 'stride': 1}],
                        ['conv', {'ksize': 1, 'chn': 4, 'actions': '#act'}]]])

CnnExtModel.set_macro('flower_resize',
                      ['parallel',
                       ['conv', {'ksize': 3, 'stride': 2, 'chn': 12, 'actions': '#act'}],
                       ['serial',
                        ['conv', {'ksize': 3, 'chn': 12, 'actions': '#act'}],
                        ['conv', {'ksize': 3, 'stride': 2, 'chn': 12, 'actions': '#act'}]],
                       ['avg', {'ksize': 3, 'stride': 2}]])

CnnExtModel.set_macro('flower_inception2',
                      ['parallel',
                       ['conv', {'ksize': 1, 'chn': 8, 'actions': '#act'}],
                       ['serial',
                        ['conv', {'ksize': [3, 3], 'chn': 8, 'actions': '#act'}],
                        ['parallel',
                         ['conv', {'ksize': [1, 3], 'chn': 8, 'actions': '#act'}],
                         ['conv', {'ksize': [3, 1], 'chn': 8, 'actions': '#act'}]]],
                       ['serial',
                        ['conv', {'ksize': [1, 1], 'chn': 8, 'actions': '#act'}],
                        ['conv', {'ksize': [3, 3], 'chn': 8, 'actions': '#act'}],
                        ['parallel',
                         ['conv', {'ksize': [1, 3], 'chn': 8, 'actions': '#act'}],
                         ['conv', {'ksize': [3, 1], 'chn': 8, 'actions': '#act'}]]],
                       ['serial',
                        ['avg', {'ksize': 3, 'stride': 1}],
                        ['conv', {'ksize': 1, 'chn': 8, 'actions': '#act'}]]])

CnnExtModel.set_macro('flower_postproc',
                      ['serial',
                       ['avg', {'stride': 6}],
                       ['dropout', {'keep_prob': 0.7}]])

CnnExtModel.set_macro('inception_flower',
                      ['serial',
                       ['custom', {'name': 'flower_preproc', 'args': {'#act': '#act'}}],
                       ['custom', {'name': 'flower_inception1', 'args': {'#act': '#act'}}],
                       ['custom', {'name': 'flower_resize', 'args': {'#act': '#act'}}],
                       ['custom', {'name': 'flower_inception1', 'args': {'#act': '#act'}}],
                       ['custom', {'name': 'flower_resize', 'args': {'#act': '#act'}}],
                       ['custom', {'name': 'flower_inception2', 'args': {'#act': '#act'}}],
                       ['custom', {'name': 'flower_resize', 'args': {'#act': '#act'}}],
                       ['custom', {'name': 'flower_inception2', 'args': {'#act': '#act'}}],
                       ['custom', {'name': 'flower_postproc', 'args': {'#act': '#act'}}]])

conf_flower_LA = ['custom', {'name': 'inception_flower', 'args': {'#act': 'LA'}}]
model_flower_LA = CnnExtModel('model_flower_LA', fd, conf_flower_LA, dump_structure=True)

model_flower_LA.exec_all(report=1)
