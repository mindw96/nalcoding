import numpy as np

import mathutil
from mlp_model import MlpModel


# 최적화 함수로 경사하강법 대신 Adam을 사용하는 모델을 만든다.
class AdamModel(MlpModel):
    # 객체 초기화를 해주는 기본 함수이다.
    def __init__(self, name, dataset, hconfigs):
        self.use_adam = False
        super(AdamModel, self).__init__(name, dataset, hconfigs)

    # Adam 알고리즘을 사용하기 위한 역전파 함수이다.
    def backprop_layer(self, G_y, hconfig, pm, aux):
        x, y = aux

        if hconfig is not None:
            G_y = mathutil.relu_derv(y) * G_y

        g_y_weight = x.transpose()
        g_y_input = pm['w'].transpose()

        G_weight = np.matmul(g_y_weight, G_y)
        G_bias = np.sum(G_y, axis=0)
        G_input = np.matmul(G_y, g_y_input)

        # 기존에는 역전파 함수에서 직접 가중치와 bias를 업데이트 했지만 이제 업데이트 함수를 사용한다.
        self.update_param(pm, 'w', G_weight)
        self.update_param(pm, 'b', G_bias)

        return G_input

    # 파라미터를 업데이트 해주는 함수이다.
    def update_param(self, pm, key, delta):
        # Adam을 사용한다면 파라미터의 손실 기울기인 delta를 수정한다.
        if self.use_adam:
            delta = self.eval_adam_delta(pm, key, delta)
        # 파라미터의 값을 업데이트한다.
        pm[key] -= self.learning_rate * delta

    # 파라미터의 손실 기울기인 delta를 수정하는 함수이다.
    def eval_adam_delta(self, pm, key, delta):
        # 이동 평균 계산시 기존 값의 비율
        ro_1 = 0.9
        ro_2 = 0.999
        # 분모가 0이 되는것을 방지
        epsilon = 1.0e-8

        # 모멘텀, 2차 모멘텀, 처리 횟수를 관리 할 수 있도록 별도의 변수로 저장
        skey, tkey, step = 's' + key, 't' + key, 'n' + key
        # 1차 모멘텀과 2차 모멘텀 정보를 저장할 버퍼 공간을 할당
        if skey not in pm:
            pm[skey] = np.zeros(pm[key].shape)
            pm[tkey] = np.zeros(pm[key].shape)
            pm[step] = 0

        # 기존 값과 새 값의 가중평균인 이동평균을 계산하여 모멘텀값과 2차 모멘텀 값을 구한다.
        s = pm[skey] = ro_1 * pm[skey] + (1 - ro_1) * delta
        t = pm[tkey] = ro_2 * pm[tkey] + (1 - ro_2) * (delta * delta)

        # 처리횟수를 1 증가 시킨다.
        pm[step] += 1
        # 버퍼 할당시 초기값을 0으로 했기 때문에 step만큼 제곱으로 나누어서 step이 작을수록 변화가 커지도록 한다.
        s = s / (1 - np.power(ro_1, pm[step]))
        t = t / (1 - np.power(ro_2, pm[step]))

        # 모멘텀 값을 2차 모멘텀 값의 제곱근으로 나눈 값을 반환한다.
        return s / (np.sqrt(t) + epsilon)
