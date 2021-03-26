# 여러 모델들의 기본 베이스가 될 모델 클래스를 선언한다.
import time

import numpy as np


class Model(object):
    # 변수들을 초기화해주는 함수이다.
    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset
        self.is_training = False
        # 만약 rand_std라는 속성이 없다면 0.03으로 초기화해준다.
        if not hasattr(self, 'rand_std'):
            self.rand_std = 0.03

    # 모델의 이름과 데이터셋 str으로 반환해주는 함수이다.
    def __str__(self):
        return '{}/{}'.format(self.name, self.dataset)

    # 모델의 함수들을 실행하는 메인 함수이다.
    def exec_all(self, epoch_count=10, batch_size=10, learning_rate=0.001, report=0, show_cnt=3):
        self.train(epoch_count, batch_size, learning_rate, report)
        self.test()
        # show_cnt가 0이 아니라면 시각화를 해준다.
        if show_cnt > 0:
            self.visualize(show_cnt)


# 다층 신경망 모델의 클래스를 선언한다.
class MlpModel(Model):
    # 클래스에서 사용할 변수 및 속성들을 선언하는 함수이다.
    def __init__(self, name, dataset, hconfigs):
        # 부모 클래스의 객체 초기화 함수를 호출해서 name과 dataset의 값을 저장한다.
        super(MlpModel, self).__init__(name, dataset)
        # 은닉층의 계층 구성을 해주는 함수를 선언한다.
        self.init_parameters(hconfigs)

    # 다층 신경망의 파라미터들을 초기화하는 함수이다.
    def init_parameters(self, hconfigs):
        self.hconfigs = hconfigs
        self.pm_hiddens = []

        prev_shape = self.dataset.input_shape
        # 레이어 별 파라미터들을 초기화한다.
        for hconfig in hconfigs:
            pm_hidden, prev_shape = self.alloc_layer_param(prev_shape, hconfig)
            self.pm_hiddens.append(pm_hidden)
        # 출력층의 파라미터를 초기화한다.
        output_cnt = int(np.prod(self.dataset.output_shape))
        self.pm_output, _ = self.alloc_layer_param(prev_shape, output_cnt)

    # 레이어 별 파라미터를 할당해주는 함수이다.
    def alloc_layer_param(self, input_shape, hconfig):
        input_cnt = np.prod(input_shape)
        output_cnt = hconfig
        # 가중치와 bias를 한 쌍으로 초기화해준다.
        weight, bias = self.alloc_param_pair([input_cnt, output_cnt])

        return {'w': weight, 'b': bias}, output_cnt

    # 파라미터들을 쌍으로 할당해주는 함수이다.
    def alloc_param_pair(self, shape):
        # 가중치의 초기값을 랜덤으로 초기화해준다.
        weight = np.random.normal(0, self.rand_std, shape)
        # bias의 초기값을 0으로 초기화해준다.
        bias = np.zeros([shape[-1]])

        return weight, bias

    # 모델의 학습을 진행하는 함수이다.
    def model_train(self, epoch_count=10, batch_size=10, learning_rate=0.001, report=0):
        self.learning_rate = learning_rate
        # 지정된 학습 횟수를 배치 사이즈로 나눠서 총 진행할 미니배치의 크기를 정한다.
        batch_count = int(self.dataset.train_count / batch_size)
        time1 = time2 = int(time.time())
        # report가 0이 아니라면 학습 시작을 report한다
        if report != 0:
            print('Model {} train started: '.format(self.name))
        # 지정된 학습 횟수만큼 학습을 진행한다.
        for epoch in range(epoch_count):
            costs = []
            accs = []
            # 배치 크기를 배치 횟수만큼 곱해서 전체 학습할 데이터를 랜덤하게 선언한다.
            self.dataset.shuffle_train_data(batch_size * batch_count)
            # 미니배치 학습을 수행한다.
            for n in range(batch_count):
                # 학습 데이터와 레이블을 불러온다.
                train_x, train_y = self.dataset.get_train_data(batch_size, n)
                # 학습 후 얻은 cost와 정확도를 리스트에 보낸다.
                cost, acc = self.train_step(train_x, train_y)
                costs.append(cost)
                accs.append(acc)
            # report 옵션이 있고 일정 epoch마다 report한다.
            if report > 0 and (epoch + 1) % report == 0:
                # validation 데이터와 레이블을 불러온다.
                val_x, val_y = self.dataset.get_validate_data(100)

                acc = self.eval_accuracy(val_x, val_y)
                time3 = int(time.time())
                self.dataset.train_prt_result(epoch + 1, costs, accs, acc, time3 - time2, time3 - time1)
                time2 = time3

        tm_total = int(time.time()) - time1
        print('Model {} train ended in {} secs: '.format(self.name, tm_total))

    # test를 수행하는 함수를 선언한다.
    def model_test(self):
        # test데이터와 레이블을 불러온다.
        test_x, test_y = self.dataset.get_test_data()
        time1 = int(time.time())
        # 정확도를 측정한다.
        acc = self.eval_accuracy(test_x, test_y)
        time2 = int(time.time())
        self.dataset.test_ptr_result(self.name, acc, time2 - time1)

    # 모델을 시각화해준다.
    def model_visualize(self, num):
        print('Model {} Visualization'.format(self.name))
        # 시각화에 사용할 데이터를 불러온다.
        deX, deY = self.dataset.get_visualize_data(num)
        # deX의 추정치를 불러온다.
        est = self.get_estimate(deX)
        # 시각화를 한다.
        self.dataset.visualizer(deX, est, deY)

    # 미니배치 단위 학습을 진행하는 함수이다.
    def train_step(self, x, y):
        # train이 시작될 때 True로 설정함으로써 validation 등과 겹치지 않도록 방지한다.
        self.is_training = True
        # 순전파를 진행한다.
        output, aux_nn = self.forward_neuralnet(x)
        # 순전파를 통해서 loss 값을 구한다.
        loss, aux_pp = self.forward_postproc(output, y)
        # 정확도를 측정한다.
        accuracy = self.eval_accuracy(x, y, output)
        # 초기 loss의 편미분을 1로 설정한다.
        G_loss = 1.0
        # 역전파 처리를 통해서 G_loss와 원본 사이의 손실 기울기 값을 구한다.
        G_output = self.backprop_postproc(G_loss, aux_pp)
        # 역전파를 통해 가중치와 bias를 업데이트한다.
        self.backprop_neuralnet(G_output, aux_nn)
        # 학습 과정이 종료되었으니 training flag를 False로 꺼준다.
        self.is_training = False

        return loss, accuracy

    # 순전파를 수행하는 함수이다.
    def forward_neuralnet(self, x):
        hidden = x
        aux_layers = []
        # 은닉층의 수 만큼 순전파를 진행한다.
        for n, hconfig in enumerate(self.hconfigs):
            hidden, aux = self.forward_layer(hidden, hconfig, self.pm_hiddens[n])
            aux_layers.append(aux)
        # 출력층의 순전파를 수행한다.
        output, aux_out = self.forward_layer(hidden, None, self.pm_output)

        return output, [aux_out, aux_layers]

    # 역전파를 수행하는 함수이다.
    def backprop_neuralnet(self, G_output, aux):
        aux_out, aux_layers = aux
        # 은닉층의 편미분값을 구한다.
        G_hidden = self.backprop_layer(G_output, None, self.pm_output, aux_out)
        # 은칙층의 수 만큼 역전파를 진행한다.
        for n in reversed(range(len(self.hconfigs))):
            hconfig, pm, aux = self.hconfigs[n], self.pm_hiddens[n], aux_layers[n]
            G_hidden = self.backprop_layer(G_hidden, hconfig, pm, aux)

        return G_hidden

    # 은닉층의 레이어 별 순전파를 진행하는 함수이다.
    def forward_layer(self, x, hconfig, pm):
        # y = wx + b 식의 Hypothesis를 구한다.
        y = np.matmul(x, pm['w']) + pm['b']
        # hconfig의 값이 있다면 활성화 함수를 통과시킨다.
        if hconfig is not None:
            y = relu(y)

        return y, [x, y]

    # 레이어 별 역전파를 수행하는 함수이다.
    def backprop_layer(self, G_y, hconfig, pm, aux):
        x, y = aux
        # hconfig의 값이 있다면 Relu의 미분함수를 통과시킨다.
        if hconfig is not None:
            G_y = relu_derv(y) * G_y
        # 가중치 편미분 연산을 위해서 원본 x을 전치한다.
        g_y_weight = x.transpose()
        # 입력값의 연산을 위해서 가중치를 전치한다.
        g_y_input = pm['w'].transpose()
        # 전치한 x와 은닉층의 값을 곱해서 가중치이 편미분값을 구한다.
        G_weight = np.matmul(g_y_weight, G_y)
        # 은닉층의 값을 다 더한다.
        G_bias = np.sum(G_y, axis=0)
        # 전치한 가중치와 은닉층을 곱한다.
        G_input = np.matmul(G_y, g_y_input)
        # 가중치를 업데이트한다.
        pm['w'] -= self.learning_rate * G_weight
        # bias를 업데이트한다.
        pm['b'] -= self.learning_rate * G_bias

        return G_input

    # 순전파의 후처리를 하는 함수이다.
    def forward_postproc(self, output, y):
        # 데이터셋에서 loss를 계산하도록 호출한다.
        loss, aux_loss = self.dataset.forward_postproc(output, y)
        # 추가적인 loss 값을 구한다.
        extra, aux_extra = self.forward_extra_cost(y)

        return loss + extra, [aux_loss, aux_extra]

    # 순저파 과정의 추가적인 loss 값을 구한다.
    def forward_extra_cost(self, y):
        return 0, None

    # 역전파의 후처리를 하는 함수이다.
    def backprop_postproc(self, G_loss, aux):
        aux_loss, aux_extra = aux
        # 역전파 과정의 추가적인 loss를 구한다.
        self.backprop_extra_cost(G_loss, aux_extra)
        # 데이터셋 클래스에서 역전파 후처리를 통해서 출력값의 손실 기울기를 구한다.
        G_output = self.dataset.backprop_postproc(G_loss, aux_loss)

        return G_output

    # 역전파의 추가적인 loss 값을 구한다.
    def backprop_extra_cost(self, G_loss, aux):
        pass

    # 정확도를 측정해주는 함수이다.
    def eval_accuracy(self, x, y, output=None):
        # outputdml 값이 있다면
        if output is None:
            # 순전파를 통해 모델의 예측값을 구한다.
            output, _ = self.forward_neuralnet(x)
        # 데이터셋 클래스의 정확도 측정 메소드를 호출해서 정확도를 구하도록 한다.
        accuracy = self.dataset.eval_accuracy(x, y, output)

        return accuracy

    # 시각화 함수에 전달할 추정 결과를 산출하는 함수.
    def get_estimate(self, x):
        output, _ = self.forward_neuralnet(x)
        estimate = self.dataset.get_estimate(output)

        return estimate

