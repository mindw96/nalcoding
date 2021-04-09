from ch5.flowerdataset import FlowersDataset
from ch9.cnn_ext_model import CnnExtModel
from ch9.dataset_dummy import DummyDataset

imagenet = DummyDataset('imagenet', 'select', [224, 224, 3], 1000)

CnnExtModel.set_macro('p24',
                      ['serial',
                       ['loop', {'repeat': '#repeat'}, ['conv', {'ksize': 3, 'chn': '#chn'}]],
                       ['max', {'stride': 2}]])

CnnExtModel.set_macro('vgg_19',
                      ['serial',
                       ['custom', {'name': 'p24', 'args': {'#repeat': 2, '#chn': 64}}],
                       ['custom', {'name': 'p24', 'args': {'#repeat': 2, '#chn': 128}}],
                       ['custom', {'name': 'p24', 'args': {'#repeat': 4, '#chn': 256}}],
                       ['custom', {'name': 'p24', 'args': {'#repeat': 4, '#chn': 512}}],
                       ['custom', {'name': 'p24', 'args': {'#repeat': 4, '#chn': 512}}],
                       ['loop', {'repeat': 2}, ['full', {'width': 4096}]]])

vgg19 = CnnExtModel('vgg_19', imagenet,
                    ['custom', {'name': 'vgg_19'}], dump_structure=True)

CnnExtModel.set_macro('pn',
                      ['serial',
                       ['conv', {'ksize': 3, 'stride': 2, 'chn': '#n', 'actions': '#act'}],
                       ['loop', {'repeat': '#cnt1'},
                        ['conv', {'ksize': 3, 'chn': '#n', 'actions': '#act'}]]])

CnnExtModel.set_macro('plain_34',
                      ['serial',
                       ['conv', {'ksize': 7, 'stride': 2, 'chn': 64, 'actions': '#act'}],
                       ['max', {'stride': 2}],
                       ['loop', {'repeat': 6}, ['conv', {'ksize': 3, 'chn': 64, 'actions': '#act'}]],
                       ['custom', {'name': 'pn', 'args': {'#cnt1': 7, '#n': 128, '#act': '#act'}}],
                       ['custom', {'name': 'pn', 'args': {'#cnt1': 11, '#n': 256, '#act': '#act'}}],
                       ['custom', {'name': 'pn', 'args': {'#cnt1': 5, '#n': 512, '#act': '#act'}}],
                       ['avg', {'stride': 7}]])

plain_34 = CnnExtModel('plain_34', imagenet,
                       ['custom', {'name': 'plain_34', 'args': {'#act': 'LA'}}], dump_structure=True)

CnnExtModel.set_macro('rf',
                      ['add', {'x': True},
                       ['serial', ['conv', {'ksize': 3, 'chn': '#n', 'actions': '#act'}],
                        ['conv', {'ksize': 3, 'chn': '#n', 'actions': '#act'}]]])

CnnExtModel.set_macro('rh',
                      ['add', {'x': False},
                       ['serial', ['conv', {'ksize': 3, 'stride': 2, 'chn': '#n', 'actions': '#act'}],
                        ['conv', {'ksize': 3, 'chn': '#n', 'actions': '#act'}]],
                       ['avg', {'stride': 2}]])

CnnExtModel.set_macro('rfull',
                      ['serial',
                       ['loop', {'repeat': '#cnt'},
                        ['custom', {'name': 'rf', 'args': {'#n': '#n', '#act': '#act'}}]]])

CnnExtModel.set_macro('rhalf',
                      ['serial',
                       ['custom', {'name': 'rh', 'args': {'#n': '#n', '#act': '#act'}}],
                       ['loop', {'repeat': '#cnt1'},
                        ['custom', {'name': 'rf', 'args': {'#n': '#n', '#act': '#act'}}]]])

CnnExtModel.set_macro('residual_34',
                      ['serial',
                       ['conv', {'ksize': 7, 'stride': 2, 'chn': 64, 'actions': '#act'}],
                       ['max', {'stride': 2}],
                       ['custom', {'name': 'rfull', 'args': {'#cnt': 3, '#n': 64, '#act': '#act'}}],
                       ['custom', {'name': 'rhalf', 'args': {'#cnt1': 3, '#n': 128, '#act': '#act'}}],
                       ['custom', {'name': 'rhalf', 'args': {'#cnt1': 5, '#n': 256, '#act': '#act'}}],
                       ['custom', {'name': 'rhalf', 'args': {'#cnt1': 2, '#n': 512, '#act': '#act'}}],
                       ['avg', {'stride': 7}]])

residual_34 = CnnExtModel('residual_34', imagenet,
                          ['custom', {'name': 'residual_34', 'args': {'#act': 'LA'}}], dump_structure=True)

CnnExtModel.set_macro('bf',
                      ['add', {'x': True},
                       ['serial',
                        ['conv', {'ksize': 1, 'chn': '#n1', 'actions': '#act'}],
                        ['conv', {'ksize': 3, 'chn': '#n1', 'actions': '#act'}],
                        ['conv', {'ksize': 1, 'chn': '#n4', 'actions': '#act'}]]])

CnnExtModel.set_macro('bh',
                      ['add', {'x': False},
                       ['serial',
                        ['conv', {'ksize': 1, 'stride': 2, 'chn': '#n1', 'actions': '#act'}],
                        ['conv', {'ksize': 3, 'chn': '#n1', 'actions': '#act'}],
                        ['conv', {'ksize': 1, 'chn': '#n4', 'actions': '#act'}]],
                       ['avg', {'stride': 2}]])

CnnExtModel.set_macro('bfull',
                      ['serial',
                       ['loop', {'repeat': '#cnt'},
                        ['custom', {'name': 'bf', 'args': {'#n1': '#n1', '#n4': '#n4',
                                                           '#act': '#act'}}]]])

CnnExtModel.set_macro('bhalf',
                      ['serial',
                       ['custom', {'name': 'bh', 'args': {'#n1': '#n1', '#n4': '#n4',
                                                          '#act': '#act'}}],
                       ['loop', {'repeat': '#cnt1'},
                        ['custom', {'name': 'bf', 'args': {'#n1': '#n1', '#n4': '#n4',
                                                           '#act': '#act'}}]]])

CnnExtModel.set_macro('bottleneck_152',
                      ['serial',
                       ['conv', {'ksize': 7, 'stride': 2, 'chn': 64, 'actions': '#act'}],
                       ['max', {'ksize': 3, 'stride': 2}],
                       ['custom', {'name': 'bfull', 'args': {'#cnt': 3, '#n1': 64, '#n4': 256, '#act': '#act'}}],
                       ['custom', {'name': 'bhalf', 'args': {'#cnt1': 7, '#n1': 128, '#n4': 512,
                                                             '#act': '#act'}}],
                       ['custom', {'name': 'bhalf', 'args': {'#cnt1': 35, '#n1': 256, '#n4': 1024,
                                                             '#act': '#act'}}],
                       ['custom', {'name': 'bhalf', 'args': {'#cnt1': 2, '#n1': 512, '#n4': 2048,
                                                             '#act': '#act'}}],
                       ['avg', {'stride': 7}]])

bottleneck_152 = CnnExtModel('bottleneck_152', imagenet,
                             ['custom', {'name': 'bottleneck_152', 'args': {'#act': 'LAB'}}],
                             dump_structure=True)

fd = FlowersDataset([64, 64], [64, 64, 3])

# CnnExtModel.set_macro('residual_flower',
#     ['serial',
#         ['conv', {'ksize':7, 'stride':2, 'chn':16, 'actions':'#act'}],
#         ['max', {'stride':2}],
#         ['custom', {'name':'rfull', 'args':{'#cnt':2, '#n':16, '#act':'#act'}}],
#         ['custom', {'name':'rhalf', 'args':{'#cnt1':1, '#n':32, '#act':'#act'}}],
#         ['custom', {'name':'rhalf', 'args':{'#cnt1':1, '#n':64, '#act':'#act'}}],
#         ['avg', {'stride':4}]])
#
# residual_flower = CnnExtModel('residual_flower', fd,
#       ['custom', {'name':'residual_flower', 'args':{'#act':'LAB'}}],
#                   dump_structure=True)
# residual_flower.exec_all(epoch_count=10, report=2)

CnnExtModel.set_macro('bottleneck_flower',
                      ['serial',
                       ['conv', {'ksize': 7, 'stride': 2, 'chn': 16, 'actions': '#act'}],
                       ['max', {'ksize': 3, 'stride': 2}],
                       ['custom', {'name': 'bfull', 'args': {'#cnt': 1, '#n1': 16, '#n4': 64,
                                                             '#act': '#act'}}],
                       ['custom', {'name': 'bhalf', 'args': {'#cnt1': 2, '#n1': 32, '#n4': 128,
                                                             '#act': '#act'}}],
                       ['custom', {'name': 'bhalf', 'args': {'#cnt1': 1, '#n1': 64, '#n4': 256,
                                                             '#act': '#act'}}],
                       ['avg', {'stride': 4}]])

bottleneck_flower = CnnExtModel('bottleneck_flower', fd,
                                ['custom', {'name': 'bottleneck_flower', 'args': {'#act': 'LAB'}}],
                                dump_structure=True)
bottleneck_flower.exec_all(epoch_count=10, report=1)
