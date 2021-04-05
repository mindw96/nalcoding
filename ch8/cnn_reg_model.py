import numpy as np
from matplotlib import pyplot as plt

from ch7.cnn_basic_model import CnnBasicModel


# 여러가지 정규화 기법을 사용하는 클래스이다.
class CnnRegModel(CnnBasicModel):
    # 클래스에서 사용할 변수들을 선언해주는 함수이다.
    def __init__(self, name, dataset, hconfigs, show_maps=False, l2_decay=0, l1_decay=0):
        self.l2_decay = l2_decay
        self.l1_decay = l1_decay
        super(CnnRegModel, self).__init__(name, dataset, hconfigs, show_maps)

    # 전체적인 학습과정을 실행하는 함수로 기존 함수에서 파라미터의 값들의 분포를 히스토그램으로 보여주는 기능만 추가했다.
    def exec_all(self, epoch_count=10, batch_size=10, learning_rate=0.001, report=0, show_cnt=3, show_params=False):
        super(CnnRegModel, self).exec_all(epoch_count, batch_size, learning_rate, report, show_cnt)
        if show_params:
            self.show_param_dist()

    # 파라미터 값의 분포를 보여주는 함수이다.
    def show_param_dist(self):
        # fully connected layer의 파라미터들을 불러온다.
        params = self.collect_params()
        # 파라미터들의 평균을 구한다.
        mu = np.mean(params)
        # 파라미터의 분산을 구하고 제곱근인 표준편차를 구한다.
        sigma = np.sqrt(np.var(params))
        plt.hist(params, 100, density=True, facecolor='g', alpha=0.75)
        plt.axis([-0.2, 0.2, 0, 20.0])
        plt.text(0.08, 15.0, 'mu={:5.3f}'.format(mu))
        plt.text(0.08, 13.0, 'sigma={:5.3f}'.format(sigma))
        plt.grid(True)
        plt.show()

        total_count = len(params)
        near_zero_count = len(list(x for x in params if -1e-5 <= x <= 1e-5))
        print('Near 0 Parameters = {:4.1f}%({}/{})'.format(near_zero_count / total_count * 100, near_zero_count,
                                                           total_count))

    # 파라미터를 리스트로 반환해주는 함수이다.
    def collect_params(self):
        # 최종 레이어인 output layer에서 파라미터들을 리스트로 불러온다.
        params = list(self.pm_output['w'].flatten())
        for pm in self.pm_hiddens:
            if 'w' in pm:
                params += list(pm['w'].flatten())
            if 'k' in pm:
                params += list(pm['k'].flatten())
        return params

    # l1 규제와 l2 규제를 수행하는 함수이다.
    def forward_extra_cost(self, y):
        extra, aux_extra = super(CnnRegModel, self).forward_extra_cost(y)
        if self.l2_decay > 0 or self.l1_decay > 0:
            params = self.collect_params()
            # l2_decay가 0보다 크다면 L2 규제 공식을 통해 extra 값을 업데이트한다.
            if self.l2_decay > 0:
                extra += np.sum(np.square(params)) / 2
            # l1_decay이 값이 0보다 크다면 L1 규제 공식을 통해 extra 값을 업데이트한다.
            if self.l1_decay > 0:
                extra += np.sum(np.abs(params))
        return extra, aux_extra

    # 파라미터를 업데이트하는 함수이다.
    def update_param(self, pm, key, delta):
        if self.use_adam:
            delta = self.eval_adam_delta(pm, key, delta)
        # key가 w나 k라면
        if key in ['w', 'k']:
            # l2_decay가 0보다 크다면
            if self.l2_decay > 0:
                # delta에서 l2_decay값에 파라미터를 곱한 값을 뺀다.
                delta += self.l2_decay * pm[key]
            # l1_decay가 0보다 크다면
            if self.l1_decay > 0:
                # delta에서 l1_decay값에 파라미터의 부호에 따른 값(양수면 1, 음수면 -1, 0이면 0)을 곱한 후 뺀다
                delta += self.l1_decay * np.sign(pm[key])
        # 파라미터 값에서 학습률과 delta를 곱한 값을 뺀다.
        pm[key] -= self.learning_rate * delta

    # 정규화 기법 중 하나인 drop out을 할당 해주는 함수이다.
    def alloc_dropout_layer(self, input_shape, hconfig):
        # 사전에 입력받은 dropout 비율이 0에서 1사이에 있는지 확인 후 딕셔너리 형태로 반환한다.
        keep_prob = self.get_conf_param(hconfig, 'keep_prob', 1.0)
        assert 0 < keep_prob <= 1

        return {'keep_prob': keep_prob}, input_shape

    # dropout 레이어의 순전파를 처리하는 함수이다.
    def forward_dropout_layer(self, x, hconfig, pm):
        # 학습 모드라면
        if self.is_training:
            # 1이 나올 확률이 keep_prob 만큼인 랜덤한 값을 x의 형태만큼 만든다.
            # 즉 0과 1로 이루어진 x크기의 배열을 만드는데 1의 값은 keep_prob만큼이다.
            dmask = np.random.binomial(1, pm['keep_prob'], x.shape)
            # 원본인 x에 dmask을 곱하여 일부 원소들을 0으로 만둘어서 학습 과정에서 다음 레이어로 전달되지 않도록 한다.
            dropped = x * dmask / pm['keep_prob']
            return dropped, dmask
        # 학습 모드가 아니라면 dropout을 사용하지 않는다.
        else:
            return x, None

    # dropout의 역전파를 수행하는 함수이다.
    def backprop_dropout_layer(self, G_y, hconfig, pm, aux):
        dmask = aux
        # 학습 과정에서 배제된 값들을 다시 베재시킨다.
        G_hidden = G_y * dmask / pm['keep_prob']
        return G_hidden

    # 정규화 기법 중 하나인 noise의 여러 파라미터들을 할당해주는 함수이다.
    def alloc_noise_layer(self, input_shape, hconfig):
        noise_type = self.get_conf_param(hconfig, 'type', 'normal')
        mean = self.get_conf_param(hconfig, 'mean', 0)
        std = self.get_conf_param(hconfig, 'std', 1.0)
        ratio = self.get_conf_param(hconfig, 'ratio', 1.0)

        assert noise_type == 'normal'

        return {'mean': mean, 'std': std, 'ratio': ratio}, input_shape

    # noise layer의 순전파를 수행하는 함수이다.
    def forward_noise_layer(self, x, hconfig, pm):
        # 학습 모드이고 랜덤값이 노이즈 비율보다 낮다면
        if self.is_training and np.random.random() < pm['ratio']:
            # mean과 std 파라미터로 x 형태의 랜덤 값들을 생성해서 noise를 정의한다.
            noise = np.random.normal(pm['mean'], pm['std'], x.shape)
            # x에 noise를 추가해서 반환한다.
            return x + noise, None
        # 학습 모드가 아니거나 랜덤값이 노이즈 비율보다 높다면 원본을 반환한다.
        else:
            return x, None

    # 역전파는 계산할 필요가 없다.
    def backprop_noise_layer(self, G_y, hconfig, pm, aux):
        return G_y

    # 정규화 기법 중 하나인 batch normalization의 파라미터들을 할당해주는 함수이다.
    def alloc_batch_normal_layer(self, input_shape, hconfig):
        pm = {}
        rescale = self.get_conf_param(hconfig, 'rescale', True)
        pm['epsilon'] = self.get_conf_param(hconfig, 'epsilon', 1e-10)
        pm['exp_ratio'] = self.get_conf_param(hconfig, 'exp_ratio', 0.001)

        bn_dim = input_shape[-1]
        # 미니 배치의 평균과 분산의 이동 평균을 저장할 버퍼를 생성한다.
        pm['mavg'] = np.zeros(bn_dim)
        pm['mvar'] = np.ones(bn_dim)

        if rescale:
            # rescale 플래그가 True라면 scale과 shift를 저장할 버퍼를 생성한다.
            pm['scale'] = np.ones(bn_dim)
            pm['shift'] = np.zeros(bn_dim)

        return pm, input_shape

    # batch normalization의 순전파를 수행하는 함수이다.
    def forward_batch_normal_layer(self, x, hconfig, pm):
        # 학습모드라면
        if self.is_training:
            # x를 2차원으로 축소한다.
            x_flat = x.reshape([-1, x.shape[-1]])
            # 축소한 x의 평균을 구한다.
            avg = np.mean(x_flat, axis=0)
            # 축소한 x의 분산을 구한다.
            var = np.var(x_flat, axis=0)
            # 미니 배치의 평균과 분산을 구하고 이 값들의 이동 평균을 구하여 업데이트한다.
            pm['mavg'] += pm['exp_ratio'] * (avg - pm['mavg'])
            pm['mvar'] += pm['exp_ratio'] * (var - pm['mvar'])
        # 학습모드가 아니라면 평균과 분산을 업데이트하지 않고 그대로 사용한다
        else:
            avg = pm['mavg']
            var = pm['mvar']
        # 표준 편차를 구한다. 분산이 0일때를 대비하여 엡실론 값을 더한다.
        std = np.sqrt(var + pm['epsilon'])
        # x에서 평균을 빼고 표준 편차로 나누는 정규화 과정을 거친다.
        y = norm_x = (x - avg) / std
        # 만약 rescale이 True라면
        if 'scale' in pm:
            # 새롭게 선형 공식을 통해 y를 구한다.
            y = pm['scale'] * norm_x + pm['shift']
        return y, [norm_x, std]

    # batch normalization의 역전파를 수행하는 함수이다.
    def backprop_batch_normal_layer(self, G_y, hconfig, pm, aux):
        norm_x, std = aux
        # 만약 rescale이 True라면
        if 'scale' in pm:
            # G_y가 2차원이라면 axis는 0으로 아니라면 (0,1,2)로 설정한다.
            if len(G_y.shape) == 2:
                axis = 0
            else:
                axis = (0, 1, 2)
            G_scale = np.sum(G_y * norm_x, axis=axis)
            G_shift = np.sum(G_y, axis=axis)
            G_y = G_y * pm['scale']
            # scale과 shift 파라미터를 업데이트한다.
            pm['scale'] -= self.learning_rate * G_scale
            pm['shift'] -= self.learning_rate * G_shift
        # 순전파와 마찬가지로 표준편차로 나누어준다.
        G_input = G_y / std

        return G_input
