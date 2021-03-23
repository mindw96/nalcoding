import _csv

import numpy as np

from ch1.abalone_class import Abalone


class Pulsar(Abalone):
    def __init__(self, epoch_count, mb_size, report):
        self._epoch_count = epoch_count
        self._mb_size = mb_size
        self._report = report

    # 메인 함수, 여러 함수들을 초기화 해준다.
    def pulsar_exec(self):
        self.load_dataset()
        self.init_model()
        self.train_and_test(self._epoch_count, self._mb_size, self._report)

    # pulsar 데이터셋을 불러오는 함수
    def load_dataset(self):
        # 데이터셋 파일만 다를뿐 나머지는 1장 함수와 동일하다.
        with open('pulsar_stars.csv') as csvfile:
            csvreader = _csv.reader(csvfile)
            next(csvreader, None)
            rows = []
            for row in csvreader:
                rows.append(row)

        global data, input_cnt, output_cnt
        input_cnt, output_cnt = 8, 1
        data = np.asarray(rows, dtype='float32')

    # 순전파의 후처리를 해주는 함수
    def forward_postproc(self, output, y):
        # 예측값과 레이블을 통해 sigmoid cross entropy 값을 얻는다
        entropy = self.sigmoid_cross_entropy_with_logits(y, output)
        # loss는 엔트로피의 평균으로 구한다.
        loss = np.mean(entropy)
        return loss, [y, output, entropy]

    # 역전파의 후처리를 해주는 함수
    def backprop_postproc(self, G_loss, aux):
        y, output, entropy = aux

        g_loss_entropy = 1.0 / np.prod(entropy.shape)
        g_entropy_output = self.sigmoid_cross_entropy_with_logits_derv(y, output)

        G_entropy = g_loss_entropy * G_loss
        G_output = g_entropy_output * G_entropy

        return G_output

    # 정확도를 계산해주는 함수
    def eval_accuracy(self, output, y):
        # 예측값이 0보다 큰 양수라면 True, 0보다 작거나 같다면 False로 처리한다.
        estimate = np.greater(output, 0)
        # 레이블값이 0.5보다 큰 값이라면 True, 0.5보다 작거나 같다면 False로 처리한다.
        answer = np.greater(y, 0.5)
        # True False로 처리한 예측값과 정답값을 비교하여 정답인 개수를 구한다.
        correct = np.equal(estimate, answer)

        return np.mean(correct)

    # 활성화 함수인 Relu를 정의하는 함수
    def relu(self, x):
        # 0보다 크다면 x값 그대로 반환되고 0보다 작거나 같다면 0으로 반환된다.
        return np.maximum(x, 0)

    # 활성화 함수인 sigmoid를 정의하는 함수
    def sigmoid(self, x):
        return np.exp(-self.relu(-x)) / (1.0 + np.exp(-np.abs(x)))

    # sigmoid를 편미분한 함수이다.
    def sigmoid_derv(self, x, y):
        return y * (1 - y)

    # sigmoid cross entropy를 정의하는 함수이다.
    def sigmoid_cross_entropy_with_logits(self, z, x):
        self.z = np.reshape(z, (10, 1))
        return self.relu(x) - x * z + np.log(1 + np.exp(-np.abs(x)))

    # sigmoid cross entropy의 편미분를 정의하는 함수이다.
    def sigmoid_cross_entropy_with_logits_derv(self, z, x):
        self.z = np.reshape(z, (10, 1))
        return -self.z + self.sigmoid(x)


main = Pulsar(10, 10, 1)
main.pulsar_exec()
