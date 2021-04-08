import copy

import numpy as np

from ch8.cnn_reg_model import CnnRegModel


# Inception 모델과 ResNet 모델을 구현하는 클래스이다.
class CnnExtModel(CnnRegModel):
    macros = {}

    # 변수들을 초기화하는 함수이다.
    def __init__(self, name, dataset, hconfigs, show_maps=False, l2_decay=0, l1_decay=0, dump_structure=False):
        self.dump_structure = dump_structure
        self.layer_index = 0
        self.layer_depth = 0
        self.param_count = 0
        super(CnnExtModel, self).__init__(name, dataset, hconfigs, show_maps, l2_decay, l1_decay)

        if self.dump_structure:
            print('Total parameter count: {}'.format(self.param_count))

    # 기존 파라미터 할당 함수에 레이어 구조를 출력하는 기능을 추가하여 오버라이딩했다.
    def alloc_layer_param(self, input_shape, hconfig):
        layer_type = self.get_layer_type(hconfig)
        # 만약 레이어의 타입이 다음중 하나의 경우라면
        if layer_type in ['serial', 'parallel', 'loop', 'add', 'custom']:
            # 만약 범퍼 옵션이 활성화되어있다면
            if self.dump_structure:
                # 레이어의 타입을 별도로 복사한다.
                dump_str = layer_type
                # 만약 커스텀 레이어라면 이름을 별도로 저장하여 추가한다.
                if layer_type == 'custom':
                    name = self.get_conf_param(hconfig, 'name')
                    dump_str += ' ' + name
                print('{:>{width}}{}'.format('', dump_str, width=self.layer_depth * 2))
            # 레이어 깊이를 1 카운트 한다.
            self.layer_depth += 1
        # 파라미터를 원본 메소드를 호출하여 불러온다.
        pm, output_shape = super(CnnExtModel, self).alloc_layer_param(input_shape, hconfig)
        # 만약 레이어 타입이 다음 중 하나의 경우에 속한다면
        if layer_type in ['serial', 'parallel', 'loop', 'add', 'custom']:
            # 레이어 깊이를 1 줄인다.
            self.layer_depth -= 1
        # 특수한 경우가 아니고 범퍼 옵션이 활성화되어있다면
        elif self.dump_structure:
            # 레이어 인덱스에 1을 더해주고
            self.layer_index += 1
            # 파라미터 정보를 입력할 문자열 버퍼를 생성한다
            pm_str = ''
            # 만약 레이어 타입이
            if layer_type == 'full':
                # 가중치 파라미터의 형태를 언팩킹한다.
                ph, pw = pm['w'].shape
                # 파라미터의 총 개수를 더한다.
                pm_count = np.prod(['w'].shpae) + pm['b'].shape[0]
                # 전체 파라미터 개수를 업데이트한다.
                self.param_count += pm_count
                # 파라미터 정보를 버퍼에 입력한다.
                pm_str = ' pm:{}x{}+{}={}'.format(ph, pw, pm['b'].shape[0], pm_count)
            # 만약 레이어의 타입이 합성곱 레이어라면
            elif layer_type == 'conv':
                # 커널 파라미터의 형태를 언팩킹한다.
                kh, kw, xchn, ychn = pm['k'].shape
                # 파라미터의 총 개수를 더한다.
                pm_count = np.prod(pm['k'].shape) + pm['b'].shape[0]
                # 전체 파라미터 개수를 업데이트한다.
                self.param_count += pm_count
                # 파라미터의 정보를 버퍼에 입력한다.
                pm_str = ' pm:{}x{}x{}x{}+{}={}'.format(kh, kw, xchn, ychn, pm['b'].shape[0], pm_count)

            print('{:>{width}}{}: {}, {}=>{}{}'.format('', self.layer_index, layer_type, input_shape, output_shape,
                                                       pm_str, width=self.layer_depth * 2))

        return pm, output_shape

    # 병렬 레이어의 특징은 이전 계층에서 여러가지 합성곱 레이어와 풀링 레이어로 전달한 후 결과를 합하여 다음 레이어로 전달한다는 특징이 있다.

    # 병렬 레이어의 파라미터를 할당해주는 함수이다.
    def alloc_parallel_layer(self, x, input_shape, hconfig):
        pm_hiddens = []
        output_shape = None
        # 만약 hconfig의 두번째 값이 딕셔너리가 아니라면 빈 딕셔너리를 삽입한다.
        if not isinstance(hconfig[1], dict):
            hconfig.insert(1, {})
        # hconfig의 3번째 값부터 반복문을 통해 돌면서 레이어의 정보를 받아오고 리스트에 추가한다.
        for bconfig in hconfig[2:]:
            bpm, bshape = self.alloc_layer_param(input_shape, bconfig)
            pm_hiddens.append(bpm)
            # 만약 출력층이 있다면
            if output_shape:
                # 출력층의 채널수에 각 레이어의 채널 수를 더해준다.
                assert output_shape[0:-1] == bshape[0:-1]
                output_shape[-1] += bshape[-1]
            # 출력층이 없다면 현재 레이어의 형태를 출력층의 형태로 입력한다.
            else:
                output_shape = bshape

        return {'pms': pm_hiddens}, output_shape

    # 병렬 레이어의 순전파를 처리하는 함수이다.
    def forward_parallel_layer(self, x, hconfig, pm):
        bys, bauxes, bchns = [], [], []
        # hconfig의 3번째 값부터 반복문을 돌면서 각 레이어별 순전파를 처리하고 결과를 각각의 리스트에 저장한다.
        for n, bconfig, in enumerate(hconfig[2:]):
            by, baux = self.forward_layer(x, bconfig, pm['pms'][n])
            bys.append(by)
            bauxes.append(baux)
            bchns.append(by.shape[-1])
        y = np.concatenate(bys, axis=-1)

        return y, [bauxes, bchns]

    # 병렬 레이어의 역전파를 처리하는 함수이다.
    def backprop_parallel_layer(self, G_y, hconfig, pm, aux):
        bauxes, bchns = aux
        bcn_from = 0
        G_x = 0
        # 반복문을 통해 레이어를 순차적으로 역전파하며 그 결과를 G_x의 값에 더해준다.
        for n, bconfig in enumerate(hconfig[2:]):
            bcn_to = bcn_from + bchns[n]
            # 슬라이싱을 통해 특정 레이어의 정보들만 역전파에 활용한다.
            G_y_slice = G_y[:, :, :, bcn_from:bcn_to]
            # 순전파시 입력은 모두 동일하기 떄문에 입력값에 대한 손실 기울기는 모든 손실 기울기를 더해주면 된다.
            G_x += self.backprop_layer(G_y_slice, bconfig, pm['pms'][n], bauxes[n])
        return G_x

    # 순차 레이어의 특징은 병렬 레이어와 비슷하지만 각 레이어가 여러개가 이어진 형태로 되어있다는 점이다.

    # 순차 레이어의 파라미터를 할당해주는 함수이다.
    def alloc_serial_layer(self, input_shape, hconfig):
        pm_hiddens = []
        prev_shape = input_shape

        if not isinstance(hconfig[1], dict):
            hconfig.insert(1, {})

        # 반복문을 통해 각 레이어를 순차적으로 돌면서 파라미터를 할당해주고 리스트에 추가한다.
        for sconfig in hconfig[2:]:
            pm_hidden, prev_shape = self.alloc_layer_param(prev_shape, sconfig)
            pm_hiddens.append(pm_hidden)

        return {'pms': pm_hiddens}, prev_shape

    # 순차 레이어의 순전파를 수행하는 함수이다.
    def forward_serial_layer(self, x, hconfig, pm):
        hidden = x
        auxes = []

        # 반복문을 통해서 각 레이어를 순차적으로 돌면서 순전파를 수행하고 그 결과를 리스트에 저장한다.
        for n, sconfig in enumerate(hconfig[2:]):
            hidden, aux = self.forward_layer(hidden, sconfig, pm['pms'][n])
            auxes.append(aux)

        return hidden, auxes

    # 순차 레이어의 역전파를 수행하는 함수이다.
    def backprop_serial_layer(self, G_y, hconfig, pm, aux):
        auxes = aux
        G_hidden = G_y

        # 인덱스를 거꾸로 넣어주면서 뒷 레이어부터 역전파를 수행한다.
        for n in reversed(range(len(hconfig[2:]))):
            sconfig, spm, saux = hconfig[2:][n], pm['pms'][n], auxes[n]
            # 역전파를 레이어수만큼 반복하며 순차적으로 수행하며 파라미터를 업데이트한다.
            G_hidden = self.backprop_layer((G_hidden, sconfig, spm, saux))

        # 맨 첫 레이어의 역전파 값을 반환한다.
        return G_hidden

    # 합산 레이어의 특징은 이어지는 레이어들의 결과에 입력 레이어의 값을 합산해서 최종 결과를 얻어낸다.

    # 합산 레이어의 파라미터를 할당하는 함수이다.
    def alloc_add_layer(self, input_shape, hconfig):
        if not isinstance(hconfig[1], dict):
            hconfig.insert(1, {})

        bpm, output_shape = self.alloc_layer_param(input_shape, hconfig[2])
        pm_hiddens = [bpm]

        for bconfig in hconfig[3:]:
            bpm, bshape = self.alloc_layer_param(input_shape, bconfig)
            pm_hiddens.append(bpm)
            self.check_add_shapes(output_shape, bshape)

        if self.get_conf_param(hconfig, 'x', True):
            self.check_add_shapes(output_shape, input_shape)

        pm = {'pms': pm_hiddens}

        for act in self.get_conf_param(hconfig, 'actions', ''):
            if act == 'B':
                bn_config = ['batch_normal', {'rescale': True}]
                pm['bn'], _ = self.alloc_batch_normal_param(output_shape, bn_config)

        return pm, output_shape

    # 합산 레이어의 순전파를 처리하는 함수이다.
    def forward_add_layer(self, x, hconfig, pm):
        # 처음에는 이전 레이어에 대한 순전파를 처리한다.
        y, baux = self.forward_layer(x, hconfig[2], pm['pms'][0])
        bauxes, bchns, aux_bn = [baux], [y.shape[-1]], []

        # 이전 레이어 이후의 레이어들을 반복문을 통해서 순차적으로 처리한다.
        for n, bconfig in enumerate(hconfig[3:]):
            by, baux = self.forward_layer(x, bconfig, pm['pms'][n + 1])
            # 순전파 결과들을 더해준다. tile_add_result 함수를 통해서 차원을 통일시켜서 차원이 맞지 않는 오류를 해결한다.
            y += self.tile_add_result(by, y.shape[-1], by.shape[-1])
            # 각 레이어의 순전파 결과들을 각 리스트에 추가한다.
            bauxes.append(baux)
            bchns.append(by.shape[-1])

        if self.get_conf_param(hconfig, 'x', True):
            y += self.tile_add_result(x, y.shape[-1], x.shape[-1])

        for act in self.get_conf_param(hconfig, 'actions', ''):
            # 만약 'action'의 키 값이 A라면
            if act == 'A':
                # 활성화 함수를 통과시킨다.
                y = self.activate(y, hconfig)
            # B라면 Batch Normalization을 수행한다.
            if act == 'B':
                y, aux_bn = self.forward_batch_normal_layer(y, None, pm['bn'])

        return y, [y, bauxes, bchns, aux_bn, x.shape]

    # 합산 레이어의 역전파를 수행하는 함수이다.
    def backprop_add_layer(self, G_y, hconfig, pm, aux):
        y, bauxes, bchns, aux_bn, x_shape = aux

        # 뒷 레이어의 actions의 키값에 따라 미분 값을 구한다.
        for act in reversed(self.get_conf_param(hconfig, 'actions', '')):
            if act == 'A':
                G_y = self.activate_derv(G_y, y, hconfig)
            if act == 'B':
                G_y = self.backprop_batch_normal_layer(G_y, None, pm['bn'], aux_bn)

        G_x = np.zeros(x_shape)

        for n, bconfig in enumerate(hconfig[2:]):
            # 순전파에서 각 레이어들의 출력값들을 더해줬기 때문에 역전파에서도 함수를 통해 처리한다.
            G_by = self.merge_add_grad(G_y, G_y.shape[-1], bchns[n])
            # 처리한 값으로 역전파를 수행한다.
            G_x += self.backprop_layer(G_by, hconfig, pm['pms'][n], bauxes[n])

        # x의 키값이 True라면 마찬가지로 함수를 통해 처리한다.
        if self.get_conf_param(hconfig, 'x', True):
            G_x += self.merge_add_grad(G_y, G_y.shape[-1], x_shape[-1])

        return G_x

    # 두 입력값의 가로, 세로 해상도가 각각 일치하는지 확인하는 함수이다.
    def check_add_shapes(self, yshape, bshape):
        assert yshape[:-1] == bshape[:-1]
        assert yshape[-1] % bshape[-1] == 0

    # 두 입력값의 채널 수를 맞춰주는 함수이다.
    def tile_add_result(self, by, ychn, bchn):
        # 만약 두 값의 채널의 수가 같다면
        if ychn == bchn:
            # 그대로 반환해준다.
            return by
        # 채널이 다르다면 ychn에 bchn이 몇번들어가는지 구한다음 그 값만큼 np.tile 함수를 통해 by를 반복하여 채널수를 맞춰준다.
        times = ychn // bchn
        return np.tile(by, times)

    # y의 손실기울기인 G_y를 합산되기 전으로 처리하는 함수이다.
    def merge_add_grad(self, G_y, ychn, bchn):
        # 두 채널이 같다면 그대로 반환한다.
        if ychn == bchn:
            return G_y
        times = ychn // bchn
        # 채널이 불일치한다면 G_y의 채널을 times, bchn으로 대체한다.
        split_shape = G_y.shape[:-1] + tuple([times, bchn])
        # times를 축으로 sum 연산을 통해서 G_y를 다 더해줌으로 손실기울기 값을 구한다.
        return np.sum(G_y.reshape(split_shape), axis=-2)

    # 반복 레이어의 특징은 같은 내용이 반복되는 경우 1개의 레이어만 선언하고 반복을 가능하도록 해준다.

    # 반복 레이어의 파라미터를 할당해주는 함수이다.
    def alloc_loop_layer(self, input_shape, hconfig):
        pm_hiddens = []
        prev_shape = input_shape

        if not isinstance(hconfig[1], dict):
            hconfig.insert(1, {})

        for n in range(self.get_conf_param(hconfig, 'repeat', 1)):
            pm_hidden, prev_shape = self.alloc_layer_param(prev_shape, hconfig[2])
            pm_hiddens.append(pm_hidden)

        return {'pms': pm_hiddens}, prev_shape

    # 반복 레이어의 순전파를 처리하는 함수이다.
    def forward_loop_layer(self, x, hconfig, pm):
        hidden = x
        aux_layers = []
        # report의 값 만큼 레이어를 반복한다.
        for n in range(self.get_conf_param(hconfig, 'repeat', 1)):
            hidden, aux = self.forward_layer(hidden, hconfig[2], pm['pms'][n])
            aux_layers.append(aux)

        return hidden, aux_layers

    # 반복 레이어의 역전파를 처리하는 함수이다.
    def backprop_loop_layer(self, G_y, hconfig, pm, aux):
        G_hidden = G_y
        aux_layers = aux
        # repeat의 값 만큼 레이어를 반복하는데 인덱스를 뒤에서 부터 넣는다.
        for n in reversed(range(self.get_conf_param(hconfig, 'repeat', 1))):
            pm_hidden, aux = pm['pms'][n], aux_layers[n]
            G_hidden = self.backprop_layer(G_hidden, hconfig[2], [pm_hidden, aux])

        return G_hidden

    # 사용자 정의 레이어는 여러가지 레이어들을 묶어서 매크로로 처리하여 간편하게 사용할 수 있는 함수이다. Residual Block을 쉽게 만들고 사용할 수 있다.

    # 사용자 정의 레이어의 파라미터를 할당해주는 함수이다.
    def alloc_custom_layer(self, input_shape, hconfig):
        # 매크로로 지정할 이름을 선언한다.
        name = self.get_conf_param(hconfig, 'name')
        # 매크로에 구성될 레이어들을 선언한다.
        args = self.get_conf_param(hconfig, 'args', {})
        # 이름과 구성 레이어를 통해 매크로를 설정해준다.
        macro = CnnExtModel.get_macro(name, args)

        pm_hidden, output_shape = self.alloc_layer_param(input_shape, macro)

        return {'pm': pm_hidden, 'macro': macro}, output_shape

    # 사용자 정의 레이어의 순전파를 처리하는 함수이다.
    def forward_custom_layer(self, x, hconfig, pm):
        return self.forward_layer(x, pm['macro'], pm['pm'])

    # 사용자 정의 레이어의 역전파를 처리하는 함수이다.
    def backprop_layer(self, G_y, hconfig, pm, aux):
        return self.backprop_layer(G_y, pm['macro'], pm['pm'], aux)

    # 매크로를 등록해주는 함수이다.
    def set_macro(self, name, config):
        # config를 name으로 매크로 등록한다.
        CnnExtModel.macros[name] = config

    # 매크로를 조회해주는 함수이다.
    def get_macro(self, name, args):
        # 매크로를 깊은 복사를 통해 불러온다.
        restored = copy.deepcopy(CnnExtModel.macros[name])
        # 재귀 구조를 통해서 모든 레이어를 조회할 수 있도록 한다.
        self.replace_arg(restored, args)

        return restored

    # 매크로의 내용으로 레이어들을 바꿔주는 함수이다.
    def replace_arg(self, exp, args):
        # 만약 exp의 값이 리스트나 튜플이라면
        if isinstance(exp, (list, tuple)):
            # 반복문을 통해서 모든 레이어를 돌린다.
            for n, term in enumerate(exp):
                # 현재 레이어의 이름이 string이고 첫번째 값이 #이라면
                if isinstance(term, str) and term[0] == '#':
                    # 그리고 두번째도 #이라면 첫번째 값을 삭제한다.
                    if term[1] == '#':
                        exp[n] = term[1:]
                    # 만약 두번째 값이 #이 아니고 레이어의 이름이 args에 있다면
                    elif term in args:
                        # 현재 레이어의 내용은 매크로로 대체한다.
                        exp[n] = args[term]
                # 만약 string이 아니거나 #이 아니라면
                else:
                    # 재귀 함수를 통해 다음 값으로 이동한다.
                    self.replace_arg(term, args)

        # 만약 딕셔너리 형태라면 키 값을 통해서 조회한다.
        elif isinstance(exp, dict):
            for key in exp:
                if isinstance(exp[key], str) and exp[key][0] == '#':
                    if exp[key][1] == '#':
                        exp[key] = exp[key][1:]
                    elif exp[key] in args:
                        exp[key] = args[exp[key]]
                else:
                    self.replace_arg(exp[key], args)

    # 기존 합성곱 연산 파라미터 할당 함수에 몇가지 기능을 추가한 함수이다.
    def alloc_conv_layer(self, input_shape, hconfig):
        pm, output_shape = super(CnnExtModel, self).alloc_conv_layer(input_shape, hconfig)

        pm['actions'] = self.get_conf_param(hconfig, 'actions', 'LA')
        for act in pm['actions']:
            if act == 'L':
                input_shape = output_shape
            # 하이퍼 파라미터를 통해 Batch Normalization을 사용할 수 있다.
            elif act == 'B':
                bn_config = ['batch_normal', {'rescale': False}]
                pm['bn'], _ = self.alloc_batch_normal_layer(input_shape, bn_config)

        xh, xw, xchn = input_shape
        ychn = self.get_conf_param(hconfig, 'chn')
        # Stride 기능을 추가하였다.
        output_shape = self.eval_stride_shape(hconfig, True, xh, xw, ychn)

        return pm, output_shape

    # 기존 합성곱 레이어의 순전파 처리 함수에 몇가지 기능을 추가하였다.
    def forward_conv_layer(self, x, hconfig, pm):
        y = x
        x_flat, k_flat, relu_y, aux_bn = None, None, None, None
        # actions 파라미터의 순서대로 순차적으로 진행한다.
        for act in pm['actions']:
            # 만약 선형 연산 모드라면
            if act == 'L':
                # 레이블의 각 차원의 크기를 언팩킹한다.
                mb_size, xh, xw, xchn = y.shape
                # 커널의 각 차원의 크기를 언팩킹한다.
                kh, kw, _, ychn = pm['k'].shape
                # 차원을 축소하여 행렬곱연산을 하는 Convolution 연산을 수행한다.
                x_flat = self.get_ext_regions_for_conv(y, kh, kw)
                k_flat = pm['k'].reshape([kh * kw * xchn, ychn])
                conv_flat = np.matmul(x_flat, k_flat)
                y = conv_flat.reshape([mb_size, xh, xw, ychn] + pm['b'])
            # 만약 비선형 활성화 함수라면
            elif act == 'A':
                # 활성화 함수를 통과시킨다.
                y = self.activate(y, hconfig)
                relu_y = y
            # 만약 배치 정규화라면
            elif act == 'B':
                # Batch Normaliztion을 수행한다.
                y, aux_bn = self.forward_batch_normal_layer(y, None, pm['bn'])

        # Stride 연산을 수행한다.
        y, aux_stride = self.stride_filter(hconfig, True, y)

        if self.need_maps:
            self.maps.append(y)

        return y, [x_flat, k_flat, relu_y, aux_bn, aux_stride]

    # 합성곱 연산의 역전파를 처리하는 함수에 몇가지 기능을 추가했다.
    def backprop_conv_layer(self, G_y, hconfig, pm, aux):
        x_flat, k_flat, x, relu_y, aux_bn, aux_stride = aux

        G_x = self.stride_filter_derv(hconfig, True, G_y, aux_stride)

        for act in reversed(pm['actions']):
            if act == 'L':
                kh, kw, xchn, ychn = pm['k'].shape
                mb_size, xh, xw, _ = G_x.shape

                G_conv_flat = G_x.reshape(mb_size * xh * xw, ychn)
                g_conv_k_flat = x_flat.transpose()
                g_conv_x_flat = k_flat.transpose()
                G_k_flat = np.matmul(g_conv_k_flat, G_conv_flat)
                G_x_flat = np.matmul(G_conv_flat, g_conv_x_flat)
                G_bias = np.sum(G_conv_flat, axis=0)
                G_kernel = G_k_flat.reshape([kh, kw, xchn, ychn])
                G_x = self.undo_ext_regions_for_conv(G_x_flat, x, kh, kw)

                self.update_param(pm, 'k', G_kernel)
                self.update_param(pm, 'b', G_bias)

            elif act == 'A':
                G_x = self.activate_derv(G_x, relu_y, hconfig)
            elif act == 'B':
                G_x = self.backprop_batch_normal_layer(G_x, None, pm['bn'], aux_bn)

        return G_x

    # Max Pooling 레이어의 파라미터를 할당해주는 함수이다.
    def alloc_max_layer(self, input_shape, hconfig):
        xh, xw, ychn = input_shape
        # Stride를 적용한 크기를 출력값의 크기로 설정한다.
        output_shape = self.eval_stride_shape(hconfig, False, xh, xw, ychn)

        return None, output_shape

    # 기존 Max Pooling 레이어의 순전파 처리 함수에 몇가지 기능을 추가하였다.
    def forward_max_layer(self, x, hconfig, pm):
        mb_size, xh, xw, chn = x.shape
        sh, sw = self.get_conf_param_2d(hconfig, 'stride', [1, 1])
        kh, kw = self.get_conf_param_2d(hconfig, 'ksize', [sh, sw])
        padding = self.get_conf_param(hconfig, 'padding', 'SAME')

        # 만약 sh, sw의 값이 kh, kw가 같고 xh가 sh로 딱 떨어지고 xw가 sw로 딱 떨어지고 SAME 패딩이라면
        if [sh, sw] == [kh, kw] and xh % sh == 0 and xw % sw == 0 and padding == 'SAME':
            # Max Pooling 레이어의 결과값을 반환한다.
            return super(CnnExtModel, self).forward_max_layer(x, hconfig, pm)

        x_flat = self.get_ext_regions(x, kh, kw, -np.inf)
        x_flat = x_flat.transpose([2, 5, 0, 1, 3, 4])
        x_flat = x_flat.reshape(mb_size * chn * xh * xw, kh * kw)
        # 각 출력 픽셀에 대한 최댓값의 위치를 수집한다.
        max_idx = np.argmax(x_flat, axis=1)
        # 최대값 인덱스들의 값을 통해 최대값만 추출한다.
        y = x_flat[np.arange(x_flat.shape[0]), max_idx]
        y = y.reshape([mb_size, chn, xh, xw])
        y = y.transpose([0, 2, 3, 1])
        # Stride를 사용한다.
        y, aux_stride = self.stride_filter(hconfig, False, y)

        if self.need_maps:
            self.maps.append(y)

        return y, [x.shape, kh, kw, sh, sw, padding, max_idx, aux_stride]

    # 기존 Max Pooling 레이어의 역전파를 처리 함수에 몇가지 기능을 추가하였다.
    def backprop_max_layer(self, G_y, hconfig, pm, aux):
        if not isinstance(aux, list):
            return super(CnnExtModel, self).backprop_max_layer(G_y, hconfig, pm, aux)

        x_shape, kh, kw, sh, sw, padding, max_idx, aux_stride = aux
        mb_size, xh, xw, chn, chn = x_shape

        # 순전파에선 마지막으로 Stride를 하기 때문에 역전파에선 제일 먼저 해준다.
        G_y = self.stride_filter_derv(hconfig, False, G_y, aux_stride)

        G_y = G_y.transpose([0, 3, 1, 2])
        G_y = G_y.flatten()

        G_x_flat = np.zeros([mb_size * chn * xh * xw, kh * kw])
        # 최대값이 있는 값들에게만 손실 기울기를 전달해준다.
        G_x_flat[np.arange(G_x_flat.shape[0]), max_idx] = G_y

        G_x_flat = G_x_flat.reshape(mb_size, chn, xh, xw, kh, kw)
        G_x_flat = G_x_flat.transpose([2, 3, 0, 4, 5, 1])
        G_x = self.undo_ext_regions(G_x_flat, kh, kw)

        return G_x

    # Average Pooling 레이어에 새로 기능을 추가하기 위해 파라미터를 추가했다.
    def alloc_avg_layer(self, input_shape, hconfig):
        xh, xw, chn = input_shape
        sh, sw = self.get_conf_param_2d(hconfig, 'stride', [1, 1])
        kh, kw = self.get_conf_param_2d(hconfig, 'ksize', [sh, sw])
        padding = self.get_conf_param(hconfig, 'padding', 'SAME')

        if [sh, sw] == [kh, kw] and xh % sh == 0 and xw % sw == 0 and padding == 'SAME':
            return super(CnnExtModel, self).alloc_avg_layer(input_shape, hconfig)

        one_mask = np.ones([1, xh, xw, chn])

        m_flat = self.get_ext_regions(one_mask, kh, kw, 0)
        m_flat = m_flat.transpose([2, 5, 0, 1, 3, 4])
        m_flat = m_flat.reshape(1 * chn * xh * xw, kh * kw)

        mask = np.sum(m_flat, axis=1)

        output_shape = self.eval_stride_shape(hconfig, False, xh, xw, chn)

        return {'mask': mask}, output_shape

    # 기존 Average Pooling 레이어의 순전파 처리 함수에 몇가지 기능을 추가하였다.
    def forward_avg_layer(self, x, hconfig, pm):
        mb_size, xh, xw, chn = x.shape
        sh, sw = self.get_conf_param_2d(hconfig, 'stride', [1, 1])
        kh, kw = self.get_conf_param_2d(hconfig, 'ksize', [sh, sw])
        padding = self.get_conf_param(hconfig, 'padding', 'SAME')

        if [sh, sw] == [kh, kw] and xh % sh == 0 and xw % sw == 0 and padding == 'SAME':
            return super(CnnExtModel, self).forward_avg_layer(x, hconfig, pm)

        x_flat = self.get_ext_regions(x, kh, kw, 0)
        x_flat = x_flat.transpose([2, 5, 0, 1, 3, 4])
        # 2차원으로 축소한다.
        x_flat = x_flat.reshape(mb_size * chn * xh * xw, kh * kw)
        # 축소한 X의 합을 구한다.
        hap = np.sum(x_flat, axis=1)
        # 합을 구한 X을 mask 파라미터로 나눠서 평균을 구한다.
        y = np.reshape(hap, [mb_size, -1]) / pm['mask']
        y = y.reshape([mb_size, chn, xh, xw])
        y = y.transpose([0, 2, 3, 1])

        y, aux_stride = self.stride_filter(hconfig, False, y)

        if self.need_maps:
            self.maps.append(y)

        return y, [x.shape, kh, kw, sh, sw, padding, aux_stride]

    # 기존 Average Pooling 레이어의 역전파 처리 함수에 몇가지 기능을 추가하였다.
    def backprop_avg_layer(self, G_y, hconfig, pm, aux):
        if not isinstance(aux, list):
            return super(CnnExtModel, self).backprop_avg_layer(G_y, hconfig, pm, aux)

        x_shape, kh, kw, sh, sw, padding, aux_stride = aux
        mb_size, xh, xw, chn = x_shape

        G_y = self.stride_filter_derv(hconfig, False, G_y, aux_stride)

        G_y = G_y.transpose([0, 3, 1, 2])
        G_y = G_y.flatten()

        # Max Pooling과는 다르게 모든 픽셀에 손실기울기를 전달해야한다.
        G_hap = np.reshape(G_y, [mb_size, -1]) / pm['mask']
        G_x_flat = np.tile(G_hap, (kh * kw, 1))

        G_x_flat = G_x_flat.reshape(mb_size, chn, xh, xw, kh, kw)
        G_x_flat = G_x_flat.transpose([2, 3, 0, 4, 5, 1])
        G_x = self.undo_ext_regions(G_x_flat, kh, kw)

        return G_x

    # Stride의 출력 형태를 정리해주는 함수이다.
    def eval_stride_shape(self, hconfig, conv_type, xh, xw, ychn):
        # 커널의 크기와 stride의 길이, padding값을 받는다.
        kh, kw, sh, sw, padding = self.get_shape_apram(hconfig, conv_type)
        # 만약 padding 유형이 VALID라면
        if padding == 'VALID':
            # 커널의 높이와 길이를 각각 뺀다. VALID 패딩 방식의 경우 이미지 범위를 벗어나는 경우 출력 픽셀을 생성하지 않기 떄문이다.
            xh = xh - kh + 1
            xw = xw - kw + 1
        # Stride의 시작점을 중앙에 위치하도록 한다.
        yh = xh // sh
        yw = xw // sw

        return [yh, yw, ychn]

    # Stride 처리를 하는 함수이다.
    def stride_filter(self, hconfig, conv_type, y):
        _, xh, xw, _ = x_shape = y.shape
        nh, nw = xh, xw
        kh, kw, sh, sw, padding = self.get_shape_params(hconfig, conv_type)

        # Padding의 모드가 VALID라면
        if padding == 'VALID':
            # 경계 바깥의 반영되지 않은 부문을 구한 뒤 그 부분만을 y로 설정한다.
            bh, bw = (kh - 1) // 2, (kw - 1) // 2
            nh, nw = xh - kh + 1, xw - kw + 1
            y = y[:, bh:bh + bh, bw:bw + nw:, :]

        # Stride의 보폭의 값이 1이 아니라면 Stride 보폭만큼 연산을 수행한다.
        if sh != 1 or sw != 1:
            bh, bw = (sh - 1) // 2, (sw - 1) // 2
            mh, mw = nh // sh, nw // sw
            y = y[:, bh:bh + mh * sh:sw, bw:bw + mw * sw:sw, :]

        return y, [x_shape, nh, nw]

    # Stride 처리를 하는 함수의 미분 함수이다.
    def stride_filter_defv(self, hconfig, conv_type, G_y, aux):
        x_shape, nh, nw = aux
        mb_size, xh, xw, chn = x_shape
        kh, kw, sh, sw, padding = self.get_shape_param(hconfig, conv_type)

        if sh != 1 or sw != 1:
            bh, bw = (sh - 1) // 2, (sw - 1) // 2
            mh, mw = nh // sh, nw // sw
            G_y_tmp = np.zeros([mb_size, nh, nw, chn])
            G_y_tmp[:, bh:bh + mh * sh:sh, bw:bw + mw * sw:sw, :] = G_y
            G_y = G_y_tmp

        if padding == 'VALID':
            bh, bw = (kh - 1) // 2, (kw - 1) // 2
            nh, nw = xh - kh + 1, xw - kw + 1
            G_y_tmp = np.zeros([mb_size, xh, xw, chn])
            G_y_tmp[:, bh:bh + nh, bw:bw + nw:, :] = G_y
            G_y = G_y_tmp

        return G_y

    #
    def get_shape_params(self, hconfig, conv_type):
        if conv_type:
            kh, kw = self.get_conf_param_2d(hconfig, 'ksize')
            sh, sw = self.get_conf_param_2d(hconfig, 'stride', [1, 1])
        else:
            sh, sw = self.get_conf_param_2d(hconfig, 'stride', [1, 1])
            kh, kw = self.get_conf_param_2d(hconfig, 'ksize', [sh, sw])
        padding = self.get_conf_param(hconfig, 'padding', 'SAME')

        return kh, kw, sh, sw, padding
   