import numpy as np
import csv
import time

np.random.seed(1234)

RND_MEAN = 0
RND_STD = 0.003

learningrate = 0.001


class Abalone:
    # 하이퍼파라미터 값들을 매개변수로 받아서 학습에 사용되는 함수로 전달해주는 전체적인 함수
    def abalone_exec(self, epoch_count=50, mb_size=150, report=1):
        self.load_abalone_dataset()
        self.init_model()
        self.train_and_test(epoch_count, mb_size, report)

    # 데이터셋을 불러오는 함수
    def load_abalone_dataset(self):
        # csv 파일을 csv라이브러리를 활용하여 불러온다
        with open('../ch1/abalone.csv') as csvfile:
            csvreader = csv.reader(csvfile)
            # 첫째줄은 헤더정보들이 있기 떄문에 생략
            next(csvreader, None)
            rows = []
            # 1줄씩 읽어서 rows 리스트에 전달
            for row in csvreader:
                rows.append(row)
        global data, input_cnt, output_cnt
        input_cnt, output_cnt = 10, 1
        data = np.zeros([len(rows), input_cnt + output_cnt])
        # 첫번째 열의 값을 보고 클래스인 성별을 원핫인코딩으로 변환
        for n, row in enumerate(rows):
            if row[0] == 'I':
                data[n, 0] = 1
            if row[0] == 'M':
                data[n, 1] = 1
            if row[0] == 'F':
                data[n, 2] = 1
            data[n, 3:] = row[1:]

    # 모델의 가중치를 초기화하는 함수
    def init_model(self):
        global weight, bias, input_cnt, output_cnt
        # 초기 가중치를 정규분포를 갖는 난수로 초기화
        weight = np.random.normal(RND_MEAN, RND_STD, [input_cnt, output_cnt])
        # bias는 0으로 초기화
        bias = np.zeros([output_cnt])

    # 학습을 진행하는 함수
    def train_and_test(self, epoch_count, mb_size, report):
        # 한번에 처리할 미니배치의 양 설정
        step_count = self.arrange_data(mb_size)
        # 테스트셋 설정
        test_x, test_y = self.get_test_data()
        # 정해진 에폭만큼 학습 진행
        for epoch in range(epoch_count):
            losses, accs = [], []

            for n in range(step_count):
                # 학습 데이터 불러오기
                train_x, train_y = self.get_train_data(mb_size, n)
                # 학습 진행
                loss, acc = self.run_train(train_x, train_y)
                losses.append(loss)
                accs.append(acc)
            # 일정 구간마다 학습 결과를 출력하도록 설정
            if report > 0 and (epoch + 1) % report == 0:
                acc = self.run_test(test_x, test_y)
                print(
                    'Epoch {}: loss={:5.3f}, accuracy={:5.3f}/{:5.3f}'.format(epoch + 1, np.mean(losses), np.mean(accs),
                                                                              acc))
        # 테스트 함수를 통해 얻은 최종 정확도 출력
        final_acc = self.run_test(test_x, test_y)
        print('\nFinal Test: final accuracy = {:5.3f}'.format(final_acc))

    # 미니배치를 설정해주는 함수
    def arrange_data(self, mb_size):
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
    def get_test_data(self):
        global data, shuffle_map, test_begin_idx, output_cnt
        # 테스트에 사용할 전체 데이터의 20% 설정
        test_data = data[shuffle_map[test_begin_idx:]]
        # output_cnt를 기준으로 앞을 데이터, 뒤를 레이블로 반환한다.
        return test_data[:, :-output_cnt], test_data[:, -output_cnt:]

    # 학습 데이터를 설정해주는 함수
    def get_train_data(self, mb_size, nth):
        global data, shuffle_map, test_begin_idx, output_cnt
        # 미니배치가 0일 때 즉 각 에폭의 첫 시작일 때 값들을 섞어줌으로 에폭마다 학습에 사용되는 데이터의 순서를 모두 다르게 해준다.
        if nth == 0:
            np.random.shuffle(shuffle_map[:test_begin_idx])
        # 전체 데이터의 80%를 학습 데이터로 사용
        train_data = data[shuffle_map[mb_size * nth:mb_size * (nth + 1)]]
        return train_data[:, :-output_cnt], train_data[:, -output_cnt]

    # 미니배치 단위 학습을 해주는 함수
    def run_train(self, x, y):
        # 순전파를 통한 output을 구한다.
        output, aux_nn = self.forward_neuralnet(x)
        # output과 레이블로 순전파 후처리 함수를 통해 loss를 구한다.
        loss, aux_pp = self.forward_postproc(output, y)
        # output과 레이블을 통해 정확도를 구한다.
        accuracy = self.eval_accuracy(output, y)
        # 초기 손실 기울기 초기화
        G_loss = 1.0
        # 역전파 후처리 함수를 통해 G_output을 구한다
        G_output = self.backprop_postproc(G_loss, aux_pp)
        # 파라미터 업데이트를 진행할 역전파 함수를 실행한다.
        # 책에서는 G_x를 구해야하는데 고정값이고 역전파도 수행되지 않으니 반환받지 않는다고 한다, 이해가 잘 안간다. 추가적인 검색이 필요하다.
        self.backprop_neuralnet(G_output, aux_nn)

        return loss, accuracy

    # 테스트를 실행해주는 함수
    def run_test(self, x, y):
        # 값을 넣었을 때 모델의 output 값을 얻는 함수
        output, _ = self.forward_neuralnet(x)
        # output과 레이블을 비교하여 정확도를 얻는 함수
        accuracy = self.eval_accuracy(output, y)
        return accuracy

    # 순전파를 처리하는 함수
    def forward_neuralnet(self, x):
        global weight, bias
        # y = wx + b 형식의 간단한 연산
        output = np.matmul(x, weight) + bias
        return output, x

    # 역전파를 처리하는 함수
    def backprop_neuralnet(self, G_output, x):
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

    # 순전파의 후처리를 하는 함수
    def forward_postproc(self, output, y):
        # output과 레이블의 차이인 오차를 구한다.
        # 책에서는 (10,1)인 output과 (10,)인 y의 '-' 연산의 결과가 (10,1)이 나온다 하지만 브로드캐스팅으로 (10,10)이 나온다.
        # 따라서 output을 (10,)으로 바꿔주고 연산 후 (10,)인 diff의 형태를 reshape함수를 통해 (10,1)로 바꿔주었다.
        output = output.flatten()
        diff = output - y
        diff = np.reshape(diff, (-1, 1))
        # 오차를 제곱한다.
        square = np.square(diff)
        # 오차 제곱의 평균을 통해 loss를 구한다. (평균오차제곱)
        loss = np.mean(square)
        return loss, diff

    # 역전파의 후처리를 하는 함수
    def backprop_postproc(self, G_loss, diff):
        # 오차의 형태를 갖는다.
        shape = diff.shape
        # loss와 오차 제곱 사이의 부분 기울기는 1/MN이다
        g_loss_square = np.ones(shape) / np.prod(shape)
        # 오차 제곱과 오차 사이의 부분 기울기는 2 * 오차이다.
        g_square_diff = 2 * diff
        # 오차와 output 사이의 부분 기울기는 1이다.
        g_diff_output = 1
        # G_loss는 1.0이기 때문에 G_square은 g_loss_square이다.
        G_square = g_loss_square * G_loss
        # G_diff는 diff * 2 * g_loss_square이다.
        G_diff = g_square_diff * G_square
        # 결과적으로 G_output은 2 * diff / np.prod(shape)으로 나타낼 수 있다.
        G_output = g_diff_output * G_diff
        return G_output

    # 챕터1은 회귀 분석이기 때문에 그에 맞는 정확도를 계산해주는 함수이다.
    def eval_accuracy(self, output, y):
        # 오차의 비율의 평균을 구한다.
        mdiff = np.mean(np.abs((output - y) / y))  # 원래는 y인데 len(y)같다...
        # 1에서 오차율 평균을 뺀 값을 정확도로 정의한다.
        return 1 - mdiff


main = Abalone()
main.abalone_exec()
print(weight)
print(bias)
