import numpy as np

# 데이터셋을 만들고 여러 기능을 사용할 수 있게 해줄 클레스를 선언한다.
class Dataset(object):
    # 매개 변수로 전달된 데이터셋의 이름과 모드값을 선언한다.
    def __init__(self, name, mode):
        self.name = name
        self.mode = mode

    # 데이터셋의 정보들을 string 문자열로 반환한다.
    def __str__(self):
        return '{}({}, {}{}{})'.format(self.name, self.mode, len(self.tr_xs), len(self.te_xs))

    # 메소드가 아닌 속성으로 취급할 수 있도록 데코레이터를 붙여준다.
    @property
    def train_count(self):
        return len(self.tr_xs)

    # 학습에 사용할 데이터를 설정하는 함수이다.
    def get_train_data(self, batch_size, nth):
        # 시작 인덱스를 설정한다.
        from_idx = nth * batch_size
        # 종료 인덱스를 설정한다.
        to_idx = (nth + 1) * batch_size
        # 인덱스를 통해 데이터셋의 범위를 통해 슬라이싱한다.
        tr_X = self.tr_sx[self.indices[from_idx:to_idx]]
        tr_Y = self.tr_ys[self.indices[from_idx:to_idx]]

        return tr_X, tr_Y

    # 학습 데이터를 섞어주는 함ㅁ수이다.
    def shuffle_train_data(self, size):
        self.indices = np.arange(size)
        np.random.shuffle(self.indices)

    # 학습에 사용할 검증 데이터셋을 설정하는 함수이다.
    def get_validate_data(self, count):
        # 인덱스를 설정하고 렌덤하게 섞습니다.
        self.va_indices = np.arange(len(self.va_xs))
        np.random.shuffle(self.va_indices)
        # 설정한 인덱스를 통해 데이터셋을 슬라이싱 합니다.
        va_X = self.va_xs[self.va_indices[0:count]]
        va_Y = self.va_ys[self.va_indices[0:count]]

        return va_X, va_Y

    # 데이터를 섞어주는 함수이다.
    def shuffle_data(self, xs, ys, tr_ratio=0.8, va_ratio=0.05):
        # 데이터의 총 카운트를 xs의 개수로 설정합니다.
        data_count = len(xs)

        # 학습, 검증, 테스트의 카운트를 설정합니다.
        tr_cnt = int(data_count * tr_ratio / 10) * 10
        va_cnt = int(data_count * va_ratio)
        te_cnt = data_count - (tr_cnt + va_cnt)

        # 인덱스를 설정해줍니다.
        tr_from, tr_to = 0, tr_cnt
        va_from, va_to = tr_cnt, tr_cnt + va_cnt
        te_from, te_to = tr_cnt + va_cnt, data_count

        indices = np.arange(data_count)
        np.random.shuffle(indices)
        # 설정한 인덱스를 통해 슬라이싱을 해서 학습, 검증, 테스트 데이터셋을 설정합니다.
        self.tr_xs = xs[indices[tr_from:tr_to]]
        self.tr_ys = xs[indices[tr_from:tr_to]]
        self.va_xs = xs[indices[va_from:va_to]]
        self.va_ys = xs[indices[va_from:va_to]]
        self.te_xs = xs[indices[te_from:te_to]]
        self.te_ys = xs[indices[te_from:te_to]]

        self.input_shape = xs[0].shape
        self.output_shape = ys[0].shape

        return indices[tr_from:tr_to], indices[va_from:va_to], indices[tr_from:te_to]

    # 순전파의 후처리를 진행하는 함수이다.
    def forward_postproc(self, output, y, mode=None):
        # 모드의 초기값을 설정한다.
        if mode is None:
            mode = self.mode

        # 모드가 회귀라면 오차제곱을 이용하여 loss를 구한다.
        if mode == 'regression':
            diff = output - y
            square = np.square(diff)
            loss = np.mean(square)
            aux = diff
        # 만약 이진 문제라면 sigmoid를 이용하여 loss를 구한다.
        elif mode == 'binary':
            entropy = sigmoid_cross_entropy_with_logits(y, output)
            loss = np.mean(entropy)
            aux = [output, y, entropy]
        # 만약 다중 문제라면 softmax를 이용하여 loss를 구한다.
        elif mode == 'select':
            entropy = softmax_cross_entropy_with_logits(y, output)
            loss = np.mean(entropy)
            aux = [output, y, entropy]

        return loss, aux

    # 역전파의 후처리를 처리하는 함수이다.
    def backprop_postproc(self, G_loss, aux, mode=None):
        # 모드를 전달받은 값으로 초기화해주는 함수이다.
        if mode is None:
            mode = self.mode

        # 만약 모드가 회귀라면 오차제곱법의 역순으로 각기의 편미분값을 통해서 미분값을 구한다.
        if mode == 'regression':
            diff = aux
            shape = diff.shape

            g_loss_square = np.ones(shape) / np.prod(shape)
            g_square_diff = 2 * diff
            g_diff_output = 1

            G_square = g_loss_square * G_loss
            G_diff = g_square_diff * G_square
            G_output = g_diff_output * G_diff

        # 만약 모드가 이진 모드라면 sigmoid의 편미분 함수를 통해서 미분 값을 구한다.
        elif mode == 'binary':
            y, output = aux
            shape = output.shape

            g_loss_entropy = np.ones(shape) / np.prod(shpae)
            g_entropy_output = sigmoid_cross_entropy_with_logits_derv(y, output)

            G_entropy = g_loss_entropy * G_loss
            G_output = g_entropy_output * G_entropy

        # 만약 모드가 다중 레이블이라면 sigmoid의 편미분 함수를 통해서 미분값을 구한다.
        elif mode == 'select':
            output, y, entropy = aux

            g_loss_entropy = 1.0 / np.prod(entropy.shape)
            g_entropy_output = softmax_cross_entropy_with_logits_derv(y, output)

            G_entropy = g_loss_entropy * G_loss
            G_output = g_entropy_output * G_entropy

        return G_output

    # 정확도를 측정해주는 함수이다.
    def eval_accuracy(self, x, y, output, mode=None):
        # 모드를 전달 받은 값으로 초기화해주는 함수이다.
        if mode is None:
            mode = self.mode
        # 모드가 회귀라면 오차제곱을 통해 나온 예측값들과 레이블의 평균을 활용하여 정확도를 구한다.
        if mode == 'regression':
            mse = np.mean(np.square(output - y))
            accuracy = 1 - np.sqrt(mse / np.mean(y))

        # 만약 모드가 이진 문제라면 예측값 중 1인 값들과 레이블 중 1인 개수를 통해서 정확도를 구한다.
        elif mode == 'binary':
            estimate = np.greater(output, 0)
            answer = np.equal(y, 1.0)
            correct = np.equal(estimate, answer)
            accuracy = np.mean(correct)
        # 만약 모드가 다중 클래스 문제라면 예측값과 레이블을 비교하여 정확도를 구한다.
        elif mode == 'select':
            estimate = np.argmax(y, axis=1)
            answer = np.argmax(y, axis=1)
            correct = np.equal(estimate, answer)
            accuracy = np.mean(correct)

        return accuracy

    # 모델의 예측값을 구하는 함수이다.
    def get_estimate(self, output, mode=None):
        # 모드를 전달 받은 모드로 초기화해주는 함수이다.
        if mode is None:
            mode = self.mode

        # 회귀 문제일경우 output을 그대로 예측값으로 전달한다.
        if mode == 'regression':
            estimate = output

        # 만약 모드가 이진 문제일경우 output을 sigmoid에 통과시킨 후 전달한다.
        elif mode == 'binary':
            estimate = sigmoid(output)

        # 만약 모드가 다중 문제일 경우 output을 softmax에 통과시킨 후 전달한다.
        elif mode == 'select':
            estimate = softmax(output)

        return estimate

    # 학습 과정 중 정확도를 출력해주는 함수이다.
    def train_prt_result(self, epoch, costs, accs, acc, time1, time2):
        print('Epoch P{: cost = {:5.3f}, accuracy = {:5.3f}/{:5.3f} ({}/{} secs)'.format(epoch, np.mean(costs),
                                                                                         np.mean(accs), acc, time1,
                                                                                         time2))

    # 테스트 결과를 출력해주는 함수이다.
    def test_prt_result(self, name, acc, time):
        print('Model {} test report: accuracy = {:5.3f}, ({} secs)\n'.format(name, acc, time))