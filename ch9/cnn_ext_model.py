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
            by, baux = self.forward_layer(x, bconfig, pm['pms'][n+1])
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



