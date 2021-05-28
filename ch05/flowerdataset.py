import os.path

import numpy as np

import mathutil
from dataset import Dataset


# 기본이 되는 Dataset 클래스를 상속하면서 꽃 이미지를 갖고있는 데이터셋 클래스를 생성한다.
class FlowersDataset(Dataset):
    # 여러 변수들을 선언해주는 함수이다.
    def __init__(self, resolution=[100, 100], input_shape=[-1]):
        # 부모 클래스에서 데이터셋의 이름은 flowers, 모드는 select로 선언한다.
        super(FlowersDataset, self).__init__('flowers', 'select')

        path = '/flowers'
        # 데이터셋의 폴더명을 통해 레이블을 추출한다.
        self.target_names = mathutil.list_dir(path)
        images = []
        idxs = []
        # enumerate를 활용하여 레이블 별 폴더를 접근하고 이미지를 추출한다.
        for dx, dname in enumerate(self.target_names):
            # 접근할 데이터셋 폴더를 선택한다.
            subpath = path + '/' + dname
            # 불러올 파일 이름들을 선택한다.
            filenames = mathutil.list_dir(subpath)
            # 파일 이름들 중 한개씩 고른다.
            for fname in filenames:
                # jpg파일 형식만 추출한다.
                if fname[-4:] != '.jpg':
                    continue
                # 이미지 파일의 경로를 만든다.
                imagepath = os.path.join(subpath, fname)
                # 이미지를 불러오는 메소드를 통해 이미지를 불러온다.
                pixels = mathutil.load_image_pixels(imagepath, resolution, input_shape)
                # 이미지 리스트로 이미지를 추가한다.
                images.append(pixels)
                # 인덱스 리스트에 인덱스를 추가한다.
                idxs.append(dx)
        # 이미지를 3차원으로 만들어준다.
        self.image_shape = resolution + [3]
        # 이미지들을 넘파이 행렬로 변환한다.
        xs = np.asarray(images, np.float32)
        # 레이블을 원핫인코딩을 통해서 변환 시킨다
        ys = mathutil.onehot(idxs, len(self.target_names))
        # 데이터를 섞는다.
        self.shuffle_data(xs, ys, 0.8)

    # 시각화를 해주는 함수이다.
    def visualize(self, xs, estimates, answers):
        mathutil.draw_images_horz(xs, self.image_shape)
        mathutil.show_select_results(estimates, answers, self.target_names)
