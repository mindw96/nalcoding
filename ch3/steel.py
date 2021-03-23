import _csv

import numpy as np


# 여러 함수들을 초기화해주는 메인 함수이다.
def steel_exec(epoch_count=10, mb_size=10, report=1):
    load_steel_dataset()
    init_model()
    train_and_test(epoch_count, mb_size, report)


# csv파일을 읽어서 데이터셋을 구축하는 함수이다.
def load_steel_dataset():
    with open('faults.csv') as csvfile:
        csvreader = _csv.reader(csvfile)
        next(csvreader, None)
        rows = []
        for row in csvreader:
            rows.append(row)

    global data, input_cnt, output_cnt
    input_cnt, output_cnt = 27, 7
    data = np.asarray(rows, dtype='float32')


# 순전파의 후처리를 해주는 함수이다.
def forward_postproc(output, y):
    # softmax 뒤에 cross entropy를 붙인 함수를 활용해 entropy를 구한다.
    entropy = softmax_cross_entropy_with_logits(y, output)
    # entropy의 평균을 통해 loss를 구한다.
    loss = np.mean(entropy)

    return loss, [y, output, entropy]


# 역전파의 후처리를 해주는 함수이다.
def backprop_postproc(G_loss, aux):
    y, output, entropy = aux

    g_loss_entropy = 1.0 / np.prod(entropy.shape)
    # softmax_cross_entropy 함수를 미분한 함수를 활용해 레이블과 output 사이의 편미분 값을 구한다.
    g_entropy_output = softmax_cross_entropy_with_logits_derv(y, output)

    G_entropy = g_loss_entropy * G_loss
    G_output = g_entropy_output * G_entropy

    return G_output


# 정확도를 측정하는 함수이다.
def eval_accuracy(output, y):
    # y축 기준으로 output의 값 중 최대값의 인덱스를 반환한다.
    estimate = np.argmax(output, axis=1)
    # y축을 기준으로 레이블의 값 중 최대값의 인덱스를 반환한다.
    answer = np.argmax(y, axis=1)
    # 예측값의 최대값들의 인덱스와 레이블의 최대값의 인덱스 중 같은 개수를 구한다.
    correct = np.equal(estimate, answer)

    return np.mean(correct)

# 활성화 함수인 softmax를 구현하는 함수이다.
def softmax(x):
    # x값 중 y축 기준으로 가장 큰 값을 반환한다. 각 행에서 최대값을 구하는 작업
    max_elem = np.max(x, axis=1)
    # 오차를 구해야 하는데 x는 행렬이고 max_elem은 벡터이기 때문에 연산을 위해 x를 전치하고 연산 후 다시 전치하여 원상복귀 시켜준다.
    diff = (x.transpose() - max_elem).transpose()
    # 오차에 자연로그를 취해준다.
    exp = np.exp(diff)
    # 자연로그를 취한 오차들의 합을 구한다.
    sum_exp = np.sum(exp, axis=1)
    # 마찬가지로 exp를 전치하고 연산 후 다시 전치하여 원상복귀 시켜서 softmax 출력 벡터를 구한다.
    probs = (exp.transpose() / sum_exp).transpose()

    return probs


# softmax의 편미분을 구하는 함수이다.
def softmax_derv(x, y):
    mb_size, nom_size = x.shape
    derv = np.ndarray([mb_size, nom_size, nom_size])
    # 미니배치 데이터에 대한 야코비 행렬을 구한ㄴ다.
    for n in range(mb_size):
        for i in range(nom_size):
            for j in range(nom_size):
                derv[n, i, j] = -y[n, i] * y[n, j]
            # i = j 일때의 값들을 더해준다.
            derv[n, i, j] += y[n, i]

    return derv


# 다중 클래스 문제에 주로 사용되는 Softmax Cross Entropy를 구현하는 함수이다.
def softmax_cross_entropy_with_logits(labels, logits):
    labels = np.reshape(labels, (10, 1))
    probs = softmax(logits)
    # 1.0e-10을 더해줌으로 probs가 0이 되어도 log값을 구할 수 있도록 해준다.
    return -np.sum(labels * np.log(probs + 1.0e-10), axis=1)


# Softmax Cross Entropy의 편미분을 구하는 함수이다.
def softmax_cross_entropy_with_logits_derv(labels, logits):
    labels = np.reshape(labels, (10, 1))
    return softmax(logits) - labels


# 앞에서 사용했던 함수를 재사용

# 모델의 가중치를 초기화하는 함수
def init_model():
    global weight, bias, input_cnt, output_cnt
    # 초기 가중치를 정규분포를 갖는 난수로 초기화
    weight = np.random.normal(RND_MEAN, RND_STD, [input_cnt, output_cnt])
    # bias는 0으로 초기화
    bias = np.zeros([output_cnt])


# 학습을 진행하는 함수
def train_and_test(epoch_count, mb_size, report):
    # 한번에 처리할 미니배치의 양 설정
    step_count = arrange_data(mb_size)
    # 테스트셋 설정
    test_x, test_y = get_test_data()
    # 정해진 에폭만큼 학습 진행
    for epoch in range(epoch_count):
        losses, accs = [], []

        for n in range(step_count):
            # 학습 데이터 불러오기
            train_x, train_y = get_train_data(mb_size, n)
            # 학습 진행
            loss, acc = run_train(train_x, train_y)
            losses.append(loss)
            accs.append(acc)
        # 일정 구간마다 학습 결과를 출력하도록 설정
        if report > 0 and (epoch + 1) % report == 0:
            acc = run_test(test_x, test_y)
            print('Epoch {}: loss={:5.3f}, accuracy={:5.3f}/{:5.3f}'.format(epoch + 1, np.mean(losses), np.mean(accs),
                                                                            acc))
    # 테스트 함수를 통해 얻은 최종 정확도 출력
    final_acc = run_test(test_x, test_y)
    print('\nFinal Test: final accuracy = {:5.3f}'.format(final_acc))


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
    y = np.reshape(y, (10, 1))
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

steel_exec()