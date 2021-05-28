import _csv

import numpy as np

# adjust_ration 옵션을 추가하여 균형잡힌 데이터셋을 사용가능하게 설정
def pulsar_exec(epoch_count=10, mb_size=10, report=1, adjust_ratio=False):
    load_pulsar_dataset(adjust_ratio)
    init_model()
    train_and_test(epoch_count, mb_size, report)

# 기존에 편향된 데이터셋을 균형잡힌 데이터셋으로 전환
def load_pulsar_dataset(adjust_ratio):
    pulsars, stars = [], []
    with open('pulsar_stars.csv') as csvfile:
        csvreader = _csv.reader(csvfile)
        next(csvreader, None)
        for row in csvreader:
            # 레이블이 1이면 pulsar로 아니라면 star로 따로 처리
            if row[8] == '1':
                pulsars.append(row)
            else:
                stars.append(row)

    global data, input_cnt, output_cnt
    input_cnt, output_cnt = 8, 1

    star_cnt, pulsar_cnt = len(stars), len(pulsars)
    # 균형잡힌 데이터셋을 원한다면 stars의 수만큼만 pulsars를 넣어서 같은 길이의 데이터셋을 구축한다.
    if adjust_ratio:
        data = np.zeros([2 * star_cnt, 9])
        data[0:star_cnt, :] = np.asarray(stars, dtype='float32')
        for n in range(star_cnt):
            data[star_cnt + n] = np.asarray(pulsars[n % pulsar_cnt], dtype='float32')
    else:
        data = np.zeros([star_cnt + pulsar_cnt, 9])
        data[0:star_cnt, :] = np.asarray(stars, dtype='float32')
        data[star_cnt:, :] = np.asarray(pulsars, dtype='float32')


# 정확도 외에도 precision, recall, f1 등의 평가지표를 정의한다.
def eval_accuracy(output, y):
    est_yes = np.greater(output, 0)
    ans_yes = np.greater(y, 0.5)
    est_no = np.logical_not(est_yes)
    ans_no = np.logical_not(ans_yes)

    tp = np.sum(np.logical_and(est_yes, ans_yes))
    fp = np.sum(np.logical_and(est_yes, ans_no))
    fn = np.sum(np.logical_and(est_no, ans_yes))
    tn = np.sum(np.logical_and(est_no, ans_no))

    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = 2 * safe_div(recall * precision, recall + precision)

    return [accuracy, precision, recall, f1]


def safe_div(p, q):
    p, q = float(p), float(q)
    if np.abs(q) < 1.0e-20:
        return np.sign(p)
    return p / q


def train_and_test(epoch_count, mb_size, report):
    step_count = arrange_data(mb_size)
    test_x, test_y = get_test_data()

    for epoch in range(epoch_count):
        losses = []

        for n in range(step_count):
            train_x, train_y = get_train_data(mb_size, n)
            loss, _ = run_train(train_x, train_y)
            losses.append(loss)

        if report > 0 and (epoch + 1) % report == 0:
            acc = run_test(test_x, test_y)
            acc_str = ','.join(['%5.3f'] * 4) % tuple(acc)
            print('Epoch {}: loss={:5.3f}, result={}'.format(epoch + 1, np.mean(losses), acc_str))

    acc = run_test(test_x, test_y)
    acc_str = ','.join(['%5.3f'] * 4) % tuple(acc)
    print('\nFinal Test: final result = {}'.format(acc_str))

# pulsar의 함수 그대로 사용

# 순전파의 후처리를 해주는 함수
def forward_postproc(output, y):
    # 예측값과 레이블을 통해 sigmoid cross entropy 값을 얻는다
    entropy = sigmoid_cross_entropy_with_logits(y, output)
    # loss는 엔트로피의 평균으로 구한다.
    loss = np.mean(entropy)
    return loss, [y, output, entropy]


# 역전파의 후처리를 해주는 함수
def backprop_postproc(G_loss, aux):
    y, output, entropy = aux

    g_loss_entropy = 1.0 / np.prod(entropy.shape)
    g_entropy_output = sigmoid_cross_entropy_with_logits_derv(y, output)

    G_entropy = g_loss_entropy * G_loss
    G_output = g_entropy_output * G_entropy

    return G_output


# 활성화 함수인 Relu를 정의하는 함수
def relu(x):
    # 0보다 크다면 x값 그대로 반환되고 0보다 작거나 같다면 0으로 반환된다.
    return np.maximum(x, 0)


# 활성화 함수인 sigmoid를 정의하는 함수
def sigmoid(x):
    return np.exp(-relu(-x)) / (1.0 + np.exp(-np.abs(x)))


# sigmoid를 편미분한 함수이다.
def sigmoid_derv(x, y):
    return y * (1 - y)


# sigmoid cross entropy를 정의하는 함수이다.
def sigmoid_cross_entropy_with_logits(z, x):
    z = np.reshape(z, (10, 1))
    return relu(x) - x * z + np.log(1 + np.exp(-np.abs(x)))


# sigmoid cross entropy의 편미분를 정의하는 함수이다.
def sigmoid_cross_entropy_with_logits_derv(z, x):
    z = np.reshape(z, (10, 1))
    return -z + sigmoid(x)


# 1장에서 사용했던 함수들 재사용

# 모델의 가중치를 초기화하는 함수
def init_model():
    global weight, bias, input_cnt, output_cnt
    # 초기 가중치를 정규분포를 갖는 난수로 초기화
    weight = np.random.normal(RND_MEAN, RND_STD, [input_cnt, output_cnt])
    # bias는 0으로 초기화
    bias = np.zeros([output_cnt])


# 미니배치를 설정해주는 함수
def arrange_data(mb_size):
    global data, shuffle_map, test_begin_idx
    # 데이터의 수 만큼 랜덤한 번호들 생성 ( 전역변수라 다른 함수에서도 사용)
    shuffle_map = np.arange(data.shape[0])
    # 랜덤한 번호들을 섞어준다.
    np.random.shuffle(shuffle_map)
    # 학습에 사용할 전체 데이터의 80%를 다시 하이퍼파라미터인 미니배치 사이즈로 나누어서 학습에 사용할 미니배치 처리 양을 정한다.
    step_count = int(data.shape[0] * 0.8) // mb_size
    # 전체 데이터 뒷부분 20%만큼을 테스트 데이터로 쓰기위해 인덱스 설정
    test_begin_idx = step_count * mb_size
    return step_count


# 테스트 데이터를 설정해주는 함수
def get_test_data():
    global data, shuffle_map, test_begin_idx, output_cnt
    # 테스트에 사용할 전체 데이터의 20% 설정
    test_data = data[shuffle_map[test_begin_idx:]]
    # output_cnt를 기준으로 앞을 데이터, 뒤를 레이블로 반환한다.
    return test_data[:, :-output_cnt], test_data[:, -output_cnt:]


# 학습 데이터를 설정해주는 함수
def get_train_data(mb_size, nth):
    global data, shuffle_map, test_begin_idx, output_cnt
    # 미니배치가 0일 때 즉 각 에폭의 첫 시작일 때 값들을 섞어줌으로 에폭마다 학습에 사용되는 데이터의 순서를 모두 다르게 해준다.
    if nth == 0:
        np.random.shuffle(shuffle_map[:test_begin_idx])
    # 전체 데이터의 80%를 학습 데이터로 사용
    train_data = data[shuffle_map[mb_size * nth:mb_size * (nth + 1)]]
    return train_data[:, :-output_cnt], train_data[:, -output_cnt]


# 미니배치 단위 학습을 해주는 함수
def run_train(x, y):
    # 순전파를 통한 output을 구한다.
    output, aux_nn = forward_neuralnet(x)
    # output과 레이블로 순전파 후처리 함수를 통해 loss를 구한다.
    loss, aux_pp = forward_postproc(output, y)
    # output과 레이블을 통해 정확도를 구한다.
    accuracy = eval_accuracy(output, y)
    # 초기 손실 기울기 초기화
    G_loss = 1.0
    # 역전파 후처리 함수를 통해 G_output을 구한다
    G_output = backprop_postproc(G_loss, aux_pp)
    # 파라미터 업데이트를 진행할 역전파 함수를 실행한다.
    # 책에서는 G_x를 구해야하는데 고정값이고 역전파도 수행되지 않으니 반환받지 않는다고 한다, 이해가 잘 안간다. 추가적인 검색이 필요하다.
    backprop_neuralnet(G_output, aux_nn)

    return loss, accuracy


# 테스트를 실행해주는 함수
def run_test(x, y):
    # 값을 넣었을 때 모델의 output 값을 얻는 함수
    output, _ = forward_neuralnet(x)
    # output과 레이블을 비교하여 정확도를 얻는 함수
    accuracy = eval_accuracy(output, y)
    return accuracy


# 순전파를 처리하는 함수
def forward_neuralnet(x):
    global weight, bias
    # y = wx + b 형식의 간단한 연산
    output = np.matmul(x, weight) + bias
    return output, x


# 역전파를 처리하는 함수
def backprop_neuralnet(G_output, x):
    global weight, bias
    # x는 순전파에서 사용된 입력값이다
    g_output_w = x.transpose()
    # 손실 기울기와 x를 연산하여 weight의 손실 기울기를 구한다. (잘 이해가 안간다 역전파의 수식적 내용을 다시 읽어봐야겠다)
    G_w = np.matmul(g_output_w, G_output)
    # G_output값들을 다 더해서 bias의 손실 기울기 구한다
    G_b = np.sum(G_output, axis=0)
    # 파라미터인 weight와 bias를 업데이트한다.
    weight -= learningrate * G_w
    bias -= learningrate * G_b


np.random.seed(1234)

RND_MEAN = 0
RND_STD = 0.003

learningrate = 0.001

pulsar_exec(adjust_ratio=True)
