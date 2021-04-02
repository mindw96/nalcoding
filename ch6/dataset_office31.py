import os

import numpy as np

import mathutil
from dataset import Dataset


# Office31 데이터셋을 불러와서 구축하는 클래스
class Office31Dataset(Dataset):
    # self.Office31Dataset은 너무 길기 떄문에 self.base로 간단히 호출 가능하도록 설정
    @property
    def base(self):
        return super(Office31Dataset, self)

    # 생성자를 선언하는 함수이다.
    def __init__(self, resolution=[100, 100], input_shape=[-1]):
        # 데이터셋의 이름은 office31이고 모드는 dual select로 설정한다.
        self.base.__init__('office31', 'dual_select')

        # 데이터셋의 경로를 설정한다.
        path = '../ch6/office31'
        # 경로에 있는 폴더들의 이름을 통해서 도메인들을 설정한다.
        domain_names = mathutil.list_dir(path)

        images = []
        domain_indexs, object_indexs = [], []

        # 각 도메인 별 세부 폴더의 이미지들을 불러와서 데이터셋을 만든다.
        for domain_index, dname in enumerate(domain_names):
            domainpath = os.path.join(path, dname, 'images')
            # 하위 폴더들의 이름을 통해서 클래스를 설정한다.
            object_names = mathutil.list_dir(domainpath)
            # 각 클래스의 파일들을 불러와서 데이터셋을 구축한다.
            for object_index, oname in enumerate(object_names):
                objectpath = os.path.join(domainpath, oname)
                filenames = mathutil.list_dir(objectpath)
                for filename in filenames:
                    if filename[-4:] != '.jpg':
                        continue
                    imagepath = os.path.join(objectpath, filename)
                    pixels = mathutil.load_image_pixels(imagepath, resolution, input_shape)
                    images.append(pixels)
                    domain_indexs.append(domain_index)
                    object_indexs.append(object_index)

        self.image_shape = resolution + [3]

        # 이미지 데이터셋을 float32로 이루어진 넘파이 배열로 변환한다.
        xs = np.asarray(images, np.float32)

        # 도메인을 원핫인코딩 처리한다.
        ys0 = mathutil.onehot(domain_indexs, len(domain_names))
        # 클래스를 원핫인코딩 처리한다.
        ys1 = mathutil.onehot(object_indexs, len(object_names))
        # 둘을 리스트 형식으로 합친다.
        ys = np.hstack([ys0, ys1])

        # 데이터셋을 섞어준다.
        self.shuffle_data(xs, ys, 0.8)
        # (도메인, 클래스) 형식의 레이블을 만든다.
        self.target_names = [domain_names, object_names]
        # 도메인의 개수를 구한다.
        self.cnts = [len(domain_names)]

    # 순전파를 구하는 함수를 새로 정의한다.
    def forward_postproc(self, output, y):
        # output은 (도메인 예측값, 클래스 예측값)으로 되어있다. 따라서 도메인과 클래스를 분리하여 계산해야 한다.
        outputs, ys = np.hsplit(output, self.cnts), np.hsplit(y, self.cnts)

        # 도메인만 분리하여 순전파를 통해 loss를 계산한다.
        loss0, aux0 = self.base.forward_postproc(outputs[0], ys[0], 'select')
        # 클래스만 분리하여 순전파를 통해 loss를 계산한다.
        loss1, aux1 = self.base.forward_postproc(outputs[1], ys[1], 'select')

        # 각 각 구한 Loss를 더하여 loss로 반환한다.
        return loss0 + loss1, [aux0, aux1]

    # 역전파의 후처리를 하는 함수를 새로 정의한다.
    def backprop_postproc(self, G_loss, aux):
        aux0, aux1 = aux

        # 도메인에 대한 예측값의 손실 기울기 값을 구한다.
        G_output0 = self.base.backprop_postproc(G_loss, aux0, 'select')
        # 클래스에 대한 예측값의 손실 기울기를 구한다.
        G_output1 = self.base.backprop_postproc(G_loss, aux1, 'select')

        return np.hstack([G_output0, G_output1])

    # 정확도를 측정해주는 함수를 새로 정의한다.
    def eval_accuracy(self, x, y, output):
        # output과 레이블을 도메인과 클래스로 분리한다.
        outputs, ys = np.hsplit(output, self.cnts), np.hsplit(y, self.cnts)

        # 도메인과 클래스 따로 따로 정확도를 측정한다.
        acc0 = self.base.eval_accuracy(x, ys[0], outputs[0], 'select')
        acc1 = self.base.eval_accuracy(x, ys[1], outputs[1], 'select')

        return [acc0, acc1]

    # 학습의 결과를 출력해주는 함수를 새로 정의한다.
    def train_prt_result(self, epoch, costs, accs, acc, time1, time2):
        acc_pair = np.mean(accs, axis=0)
        print(
            'Epoch {}: cost={:5.3f}, accuracy={:5.3f}+{:5.3f}/{:5.3f}+{:5.3f} ({}/{} secs)'.format(epoch, np.mean(costs),
                                                                                                  acc_pair[0],
                                                                                                  acc_pair[1], acc[0],
                                                                                                  acc[1], time1, time2))
    # 테스트의 결과를 출력해주는 함수를 새로 정의한다.
    def test_prt_result(self, name, acc, time):
        print('Model {} test report: accuracy = {:5.3f}+{:5.3f}, ({} secs)\n'.format(name, acc[0], acc[1], time))

    # 예측값을 구하는 함수이다.
    def get_estimate(self, output):
        # 도메인과 클래스로 분리해서 각각 예측값을 구한다.
        outputs = np.hsplit(output, self.cnts)

        estimate0 = self.base.get_estimate(outputs[0], 'select')
        estimate1 = self.base.get_estimate(outputs[1], 'select')

        return np.hstack([estimate0, estimate1])

    # 시각화를 해주는 함수이다.
    def visualize(self, xs, estimates, answers):
        # plot을 그려주는 함수이다.
        mathutil.draw_images_horz(xs, self.image_shape)

        ests, anss = np.hsplit(estimates, self.cnts), np.hsplit(answers, self.cnts)

        captions = ['도메인', '상품']

        for m in range(2):
            print('[ {} 추정결과 ]'.format(captions[m]))
            mathutil.show_select_results(ests[m], anss[m], self.target_names[m], 8)
