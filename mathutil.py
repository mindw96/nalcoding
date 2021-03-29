import _csv
import os

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np


# 수학적인 계산이 필요한 함수들을 모아둔 파일이다.

def relu(x):
    return np.maximum(x, 0)


def relu_derv(y):
    return np.sign(y)


def sigmoid(x):
    return np.exp(-relu(-x)) / (1.0 + np.exp(-np.abs(x)))


def sigmoid_derv(y):
    return y * (1 - y)


def sigmoid_cross_entropy_with_logits(z, x):
    return relu(x) - x * z + np.log(1 + np.exp(-np.abs(x)))


def sigmoid_cross_entropy_with_logits_derv(z, x):
    return -z + sigmoid(x)


def tanh(x):
    return 2 * sigmoid(2 * x) - 1


def tanh_derv(y):
    return (1.0 + y) * (1.0 - y)


def softmax(x):
    max_elem = np.max(x, axis=1)
    diff = (x.transpose() - max_elem).transpose()
    exp = np.exp(diff)
    sum_exp = np.sum(exp, axis=1)
    probs = (exp.transpose() / sum_exp).transpose()

    return probs


def softmax_cross_entropy_with_logits(labels, logits):
    probs = softmax(logits)
    return -np.sum(labels * np.log(probs + 1.0e-10), axis=1)


def softmax_cross_entropy_with_logits_derv(labels, logits):
    return softmax(logits) - labels


# csv 파일을 한 줄씩 읽어와서 리스트 형식으로 반환해주는 함수이다.
def load_csv(path, skip_header=True):
    with open(path) as csvfile:
        csvreader = _csv.reader(csvfile)
        headers = None
        if skip_header:
            headers = next(csvreader, None)
        rows = []
        for row in csvreader:
            rows.append(row)

    return rows, headers


'''
원핫인코딩이란 0,1외의 다른 숫자나 문자가 레이블로 주어졌을 때 이름 0과 1로 인코딩해주는 것을 말한다.
예를 들어 1,2,3이 주어진다면 [1,0,0],[0,1,0],[0,0,1] 등으로 바꿔서 표현해줄 수 있다.
'''


# xs 텐서의 값들을 cnt크기의 원-핫 벡터로 변환해주는 함수이다.
def onehot(xs, cnt):
    # cnt 크기의 단위 행렬을 선언하고 각 값을 int로 해줌으로써 나중에 발생할지 모르는 충동을 방지한다.
    return np.eye(cnt)[np.array(xs).astype(int)]


# 벡터를 문자열로 변환해주는 함수이다.
def vector_to_str(x, fmt='%.2f', max_cnt=0):
    # max_cnt가 0이거나 x의 길이가 max_cnt보다 작다면 벡터 그대로 문자열로 반환해준다.
    if max_cnt == 0 or len(x) <= max_cnt:
        return '[' + ','.join([fmt] * len(x)) % tuple(x) + ']'
    # 아니라면 과도하게 긴 문자열의 생성을 막는다.
    v = x[0:max_cnt]
    return '[' + ','.join([fmt] * len(v)) % tuple(v) + ',...]'


# 이미지를 numpy 배열로 변환해주는 함수이다.
def load_image_pixels(imagepath, resolution, input_shape):
    # PIL 라이브러리를 활용해서 이미지 파일을 열어준다.
    img = PIL.Image.open(imagepath)
    # 이미지의 크기를 전달받은 해상도로 변환한다.
    resized = img.resize(resolution)
    # 전달받은 형태로 shape을 변환하고 numpy 배열로 변경한다.
    return np.array(resized).reshape(input_shape)


# 이미지를 화면에 출력해주는 함수이다.
def draw_images_horz(xs, image_shape=None):
    # 출력할 xs의 개수를 구한다.
    show_cnt = len(xs)
    # 이미지를 출력할 plot의 크기를 설정한다.

    # xs의 개수만큼 반복문을 통해 출력할 이미지를 설정한다.
    for n in range(show_cnt):
        img = xs[n]
        if image_shape:
            x3d = img.reshape(image_shape)
            img = PIL.Image.fromarray(np.uint8(x3d))
        axes = plt.subplot(1, show_cnt, n + 1)
        axes.imshow(img)
        axes.axis('off')
    # 이미지들을 화면에 출력한다.
    plt.draw()
    plt.show()


# 선택 분류의 결과를 출력해주는 함수이다.
def show_select_results(est, ans, target_names, max_cnt=0):
    for n in range(len(est)):
        pstr = vector_to_str(100 * est[n], '%2.0f', max_cnt)
        estr = target_names[np.argmax(est[n])]
        astr = target_names[np.argmax(ans[n])]
        rstr = '0'
        if estr != astr:
            rstr = 'X'
        print('추정 확률 분포 {} => 추정 {} : 정답 {} => {}'.format(pstr, estr, astr, rstr))


# 특정 폴더에 있는 파일이나 서브 폴더의 이름을 모아 정렬된 형태의 리스트로 반환해주는 함수이다.
def list_dir(path):
    # 파이썬 내장 라이브러리인 os를 활용하여 전달 받은 경로에서 파일들의 목록을 리스트로 불러온다.
    filenames = os.listdir(path)
    # 불러온 리스트를 정렬한다.
    filenames.sort()
    return filenames
