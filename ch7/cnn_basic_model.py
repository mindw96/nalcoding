import numpy as np

import mathutil
from ch6.adam_model import AdamModel


# 합성곱 연산을 수행하는 클래스이다.
class CnnBasicModel(AdamModel):
    # 클래스 내 속성들을 초기화하는 함수이다.
    def __init__(self, name, dataset, hconfigs, show_maps=False):
        # hconfigs가 리스트이거나 hconfigs의 첫번째 값이  (list, int)형이 아닐때 hconfigs를 리스트로 감싼다.
        if isinstance(hconfigs, list) and not isinstance(hconfigs[0], (list, int)):
            hconfigs = [hconfigs]

        # 여러 변수들을 초기화해준다.
        self.show_maps = show_maps
        self.need_maps = False
        self.kernels = []
        super(CnnBasicModel, self).__init__(name, dataset, hconfigs)
        self.use_adam = True

    # 초기 파라미터를 할당해주는 함수이다.
    def alloc_layer_param(self, input_shape, hconfig):
        # hconfig의 레이어 정보들을 가져온다
        layer_type = self.get_layer_type(hconfig)
        # alloc_레이어명_layer 형식으로 레이어의 이름을 저장한다.
        m_name = 'alloc_{}_layer'.format(layer_type)
        # 클래스 내 해당 레이어의 속성을 가져온다.
        method = getattr(self, m_name)
        # 해당 레이어의 함수를 호출하여 파라미터와 출력값 형태를 받는다
        pm, output_shape = method(input_shape, hconfig)

        return pm, output_shape

    # 순전파를 처리하는 함수이다.
    def forward_layer(self, x, hconfig, pm):
        # 찾는 레이어의 이름을 저장한다.
        layer_type = self.get_layer_type(hconfig)
        # forward_레이어명_layer 형식으로 저장한다.
        m_name = 'forward_{}_layer'.format(layer_type)
        # m_name과 동일한 함수의 정보를 저장한다.
        method = getattr(self, m_name)
        # m_name과 동일한 함수를 호출하여 순전파를 수행하고 예측값과 입력데이터를 받는다.
        y, aux = method(x, hconfig, pm)

        return y, aux

    # 역전파를 처리하는 함수이다.
    def backprop_layer(self, G_y, hconfig, pm, aux):
        # 해당 레이어의 이름을 저장한다.
        layer_type = self.get_layer_type(hconfig)
        # backprop_레이어명_layer 형식으로 저장한다.
        m_name = 'backprop_{}_layer'.format(layer_type)
        # m_name과 동일한 이름의 함수의 정보를 저장한다.
        method = getattr(self, m_name)
        # m_name과 동일한 이름의 함수를 호출하고 입력값의 손실 기울기를 반환 받는다.
        G_input = method(G_y, hconfig, pm, aux)

        return G_input

    # 완전 연결 레이어인 fully connected layer의 파라미터들을 초기화하는 함수이다.
    def alloc_full_layer(self, input_shape, hconfig):
        # input 값의 원소의 개수를 구한다.
        input_cnt = np.prod(input_shape)
        # output 값의 원소의 개수를 구한다.
        output_cnt = self.get_conf_param(hconfig, 'width', hconfig)
        # 가중치와 bias를 초기화한다.
        weight = np.random(0, self.rand_std, [input_cnt, output_cnt])
        bias = np.zeros([output_cnt])

        return {'w': weight, 'b': bias}, [output_cnt]

    # 합성곱 연산 레이어의 파라미터들을 초기화해주는 함수이다.
    def alloc_conv_layer(self, input_shape, hconfig):
        # 입력 레이어가 3차원인지 확인한다.
        assert len(input_shape) == 3
        # 입력 레이어의 값들을 언팩킹한다.
        xh, xw, xchn = input_shape
        # 커널의 파라미터 정보를 받아온다.
        kh, kw = self.get_conf_param_2d(hconfig, 'chn')
        # 파라미터의 정보를 받아온다.
        ychn = self.get_conf_param(hconfig, 'chn')
        # 커널의 초기값을 설정한다.
        kernel = np.random.normal(0, self.rand_std, [kh, kw, xchn, ychn])
        # bias의 초기값을 설정한다.
        bias = np.zeros([ychn])
        # 시각화 옵션이 있으면 시각화를 위해 별도로 관리한다.
        if self.show_maps:
            self.kernels.append(kernel)

        return {'k': kernel, 'b': bias}, [xh, xw, ychn]

    # pooling 레이어의 파라미터를 할당해주는 함수이다.
    def alloc_pool_layer(self, input_shape, hconfig):
        # 입력 레이어의 차원 수를 확인한다.
        assert len(input_shape) == 3
        # 입력 레이어를 언팩킹한다.
        xh, xw, xchn = input_shape
        # 파라미터의 정보를 얻어온다.
        sh, sw = self.get_conf_param_2d(hconfig, 'stride')

        assert xh % sh == 0
        assert xw % sw == 0

        return {}, [xh // sh, xw // sw, xchn]

    # 레이어의 타입을 반환해주는 함수이다.
    def get_layer_type(self, hconfig):
        # hconfig가 리스트가 아닐경우 'full'로 반환하고 리스트일경우 첫번째 값을 반환한다.
        if not isinstance(hconfig, list):
            return 'full'
        return hconfig[0]

    # 파라미터를 반환해주는 함수이다.
    def get_conf_param(self, hconfig, key, defval=None):
        # 만약 hconfig가 리스트가 아니라면 defval을 반환한다.
        if not isinstance(hconfig, list):
            return defval
        # 만약 hconfig가 1차원이나 그보다 작다면 defval을 반환한다.
        if len(hconfig) <= 1:
            return defval
        # 만약 hconfig의 두번째 값에 key가 없다면 defval을 반환한다.
        if not key in hconfig[1]:
            return defval
        # 위 조건문들에 아무것도 해당되지 않는다면 hconfig의 2행의 key열 값을 반환한다.
        return hconfig[1][key]

    # 2차원 파라미터의 값을 반환해주는 값이다.
    def get_conf_param_2d(self, hconfig, key, defval=None):
        # hconfig가 1차원이거나 그보다 낮다면 defval을 반환한다.
        if len(hconfig) <= 1:
            return defval
        # 만약 hconfig의 2번째 값에 key가 없다면 defval을 반환한다.
        if not key in hconfig[1]:
            return defval
        # val값을 선언한다.
        val = hconfig[1][key]
        # 만약 val이 리스트라면 val을 그대로 반환한다.
        if isinstance(val, list):
            return val
        # val이 리스트가 아니라면 리스트 형태로 반환한다.
        return [val, val]

    # fully connected layer의 순전파 처리를 하는 함수이다.
    def forward_full_layer(self, x, hconfig, pm):
        if pm is None:
            return x, None
        # x의 shape을 따로 복사해둔다.
        x_org_shape = x.shape
        # 만약 x가 2차원이 아니라면 미니배치의 크기를 첫번째 차원으로 처리하고 x를 2차원 형태로 축소한다.
        if len(x.shape) != 2:
            mb_size = x.shape[0]
            x = x.reshape([mb_size, -1])
        # hypothesis를 구하기 위해 y = wx + b라는 선형방정식으로 연산을 진행 한다.
        affine = np.matmul(x, pm['w']) + pm['b']
        # 위에서 구한 affine값에 활성화 함수를 적용시킨다.
        y = self.activate(affine, hconfig)

        return y, [x, y, x_org_shape]

    # fully connected layer의 역전파를 처리하는 함수이다.
    def backprop_full_layer(self, G_y, hconfig, pm, aux):
        if pm is None:
            return G_y

        x, y, x_org_shape = aux
        # y와 G_y의 손실 기울기값을 구한다.
        G_affine = self.activate_derv(G_y, y, hconfig)

        g_affine_weight = x.transpose()
        g_affine_input = pm['w'].transpose()
        # g_affine_weight와 G_affine 사이의 손실기울기를 구한다.
        G_weight = np.matmul(g_affine_weight, G_affine)
        # G_affine의 손실기울기를 구한다.
        G_bias = np.sum(G_affine, axis=0)
        # G_affine과 g_affine_input 사이의 손실기울기를 구한다.
        G_input = np.matmul(G_affine, g_affine_input)
        # weight와 bias를 업데이트한다.
        self.update_param(pm, 'w', G_weight)
        self.update_param(pm, 'b', G_bias)

        return G_input.reshape(x_org_shape)

    # 활성화 함수를 처리하는 함수이다.
    def activate(self, affine, hconfig):
        if hconfig is None:
            return affine
        # hconfig의 정보를 받아온다.
        func = self.get_conf_param(hconfig, 'actfunc', 'relu')
        # 활성화 함수의 종류에 따라 맞는 활성화 함수 처리를 해서 반환해준다.
        if func == 'none':
            return affine
        elif func == 'relu':
            return mathutil.relu(affine)
        elif func == 'sigmoid':
            return mathutil.sigmoid(affine)
        elif func == 'tanh':
            return mathutil.tanh(affine)
        else:
            assert 0

    # 활성화 함수의 미분값을 전환하는 함수이다.
    def activate_derv(self, G_y, y, hconfig):
        if hconfig is None:
            return G_y

        # hconfig의 정보를 받아옴으로 할당된 활성화 함수를 받아온다.
        func = self.get_conf_param(hconfig, 'actfunc', 'relu')
        # 할당된 활성화 함수와 맞는 활성화 함수의 미분값을 반환한다.
        if func == 'none':
            return G_y
        elif func == 'relu':
            return mathutil.relu_derv(y) * G_y
        elif func == 'sigmoid':
            return mathutil.sigmoid_derv(y) * G_y
        elif func == 'tanh':
            return mathutil.tanh_derv(y) * G_y
        else:
            assert 0

    # convolution layer의 순전파 처리를 하는 함수이다.
    def forward_conv_layer_adhoc(self, x, hconfig, pm):
        # x를 언팩킹 해준다.
        mb_size, xh, xw, xchn = x.shape
        # 커널 파라미터를 언팩킹 해준다.
        kh, kw, _, ychn = pm['k'].shape

        # convolution 값을 0으로 초기화한다.
        conv = np.zeros((mb_size, xh, xw, ychn))
        # 7중 반복문을 통해서 데이터의 모든 값들을 순차적으로 찾을 수 있다.
        for n in range(mb_size):
            for r in range(xh):
                for c in range(xw):
                    for ym in range(ychn):
                        for i in range(kh):
                            for j in range(kw):
                                # 커널의 위치를 구한다.
                                rx = r + i - (kh - 1) // 2
                                cx = c + j - (kw - 1) // 2
                                # 커널이 이미지 밖으로 넘어간다면 0으로 처리한다.
                                if rx < 0 or rx >= xh:
                                    continue
                                if cx < 0 or cx >= xw:
                                    continue
                                # 채널별로 연산을 진행한다.
                                for xm in range(xchn):
                                    kval = pm['k'][i][j][xm][ym]
                                    ival = x[n][rx][cx][xm]
                                    conv[n][r][c][ym] += kval * ival
        y = self.activate(conv + pm['b'], hconfig)

        return y, [x, y]

    # adhoc보다 개선된 합성곱 연산 방법으로 처리하는 함수이다. 반복문의 사용을 줄이고자 차원 축소 후 벡터 내적 연산을 통해 계산량을 줄였다.
    def forward_conv_layer_better(self, x, hconfig, pm):
        mb_size, xh, xw, xchn = x.shape
        kh, kw, _, ychn = pm['k'].shape

        conv = np.zeros((mb_size, xh, xw, ychn))
        # 커널의 가운데 위치를 변수로 설정한다.
        bh, bw = (kh - 1) // 2, (kw - 1) // 2
        # 커널이 이미지 밖으로 나가는 최대 길이를 구한다.
        eh, ew = xh + kh - 1, xw + kw - 1
        # 커널이 이미지 범위를 벗어나는 것을 해결하기 위해 0으로 입력된 버퍼를 설정한다.
        x_ext = np.zeros((mb_size, eh, ew, xchn))
        # 버퍼의 중앙부분을 x로 입력해서 x주위에 확장된 곳은 0으로 설정되도록 한다.
        x_ext[:, bh:bh + xh, bw:bw + xw, :] = x
        # 커널을 채널별로 2차원으로 축소한다.
        k_flat = pm['k'].tanspose([3, 0, 1, 2]).reshape([ychn, -1])

        for n in range(mb_size):
            for r in range(xh):
                for c in range(xw):
                    for ym in range(ychn):
                        # 버퍼도 차원을 축소한다.
                        xe_flat = x_ext[n, r:r + kh, c:c + kw, :].flatten()
                        # 커널과 버퍼를 내적하여 convoltuion 연산의 값을 구한다.
                        conv[n, r, c, ym] = (xe_flat * k_flat[ym]).sum()
        # 합성곱 연산의 결과를 활성화 함수를 통과시킨다.
        y = self.activate(conv + pm['b'], hconfig)

        return y, [x, y]

    # 앞 2가지 합성곱 연산의 단점인 느린 속도를 단 한번의 행렬 연산으로 처리하도록하여 속도를 개선하 함수이다.
    def forward_conv_layer(self, x, hconfig, pm):
        mb_size, xh, xw, xchn = x.shape
        kh, kw, _, ychn = pm['k'].shape

        # 커널이 이미지를 벗어나는 범위를 해결하기 위한 버퍼의 크기까지 구한뒤 차원축소한 값을 구한다.
        x_flat = get_ext_regions_for_conv(x, kh, kw)
        # 커널의 차원을 2차원으로 축소한다.
        k_flat = pm['k'].reshape([kh*kw*xchn, ychn])
        # 축소한 버퍼가 포함된 데이터와 축소된 커널을 행렬곱으로 연산하여 convolution 연산 값을 구한다.
        conv_flat = np.matmul(x_flat, k_flat)
        # 합성곱 연산값을 다시 4차원으로 늘려준다.
        conv = conv_flat.reshape([mb_size, xh, xw, ychn])
        # 구한 합성곱값을 활성화 함수를 처리한다.
        y = self.activate(conv + pm['b'], hconfig)

        if self.need_maps:
            self.maps.append(y)

        return y, [x_flat, k_flat, x, y]