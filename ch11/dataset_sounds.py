import os.path
import wave

import numpy as np
from IPython.core.display import HTML

import mathutil
from dataset import Dataset


# 10가지 소음에 대해 데이터 음원 파일들에 담긴 주파수 분포 정보를 분석하여 시계열 데이터 형태로 제공하는 다중 클래스 분류 데이터셋이다.
class UrbanSoundDataset(Dataset):
    def __init__(self, interval, window):
        super(UrbanSoundDataset, self).__init__('urbansound', 'select')

        x_train, y_train, paths, self.target_names = self.load_urban_files('train.csv', 'Train', interval, window)
        yy_train = np.eye(len(self.target_names))[y_train]

        _, va_indices, _ = self.shuffle_data(x_train, yy_train, 0.8)
        self.va_paths = [paths[k] for k in va_indices]

        # x_test, y_test, paths, self.test_target_names = self.load_urban_files('test.csv', 'Test', interval, window)
        # yy_test = np.eye(len(self.test_target_names))[y_test]

    # 파일들을 불러와 데이터셋으로 반환해주는 함수이다.
    def load_urban_files(self, csv_filename, wav_foldername, interval, window):
        cache_path = '../ch11/urban-sound/{}.{}-{}.cache'.format(wav_foldername, interval, window)

        if os.path.isfile(cache_path):
            fc = np.load(cache_path)
            print('loaded from cache')

            return fc['arr_0'], fc['arr_1'], fc['arr_2'], fc['arr_3']
        # 경로에서 csv 파일을 읽어온다.
        rows, _ = mathutil.load_csv('../ch11/urban-sound/' + csv_filename)

        xs, ys, lengs, paths, targets, n = [], [], [], [], [], 0
        for row in rows:
            if row[1] not in targets:
                targets.append(row[1])
            cat_idx = targets.index(row[1])

            # 경로에서 wav 파일들을 불러온다.
            wav_path = '../ch11/urban-sound/{}/{}.wav'.format(wav_foldername, row[0])
            # 음악의 주파수 스펙트럼을 분석한다.
            chunk_cnt, chunk_dat = self.wav_to_fft(wav_path, interval, window)

            if chunk_cnt <= 0:
                continue

            xs.append(chunk_dat)
            lengs.append(chunk_cnt)
            ys.append(cat_idx)
            paths.append(wav_path)
            n += 1
            if n % 100 == 0:
                print('{} files are processed'.format(n))

        xmax_size = np.max(lengs)
        xxs = np.zeros([n, xmax_size + 1, xs[0].shape[1]])

        for n, x in enumerate(xs):
            xxs[n, 0, 0] = lengs[n]
            xxs[n, 1:lengs[n] + 1, :] = x

        fid = open(cache_path, 'wb')
        # 스펙트럼 정보들을 넘파이로 저장하여 데이터셋을 구축한다.
        np.savez(fid, xxs, ys, paths, targets)
        fid.close()

        return xxs, ys, paths, targets

    def wav_to_fft(self, fname, interval, window):
        try:
            f = wave.open(fname, 'rb')
            params = f.getparams()
            nchannels, sampwidth, framerate, nframes = params[:4]

            if nchannels != 1 and nchannels != 2:
                return 0, 1
            if sampwidth != 2:
                return 0, 2
            if framerate != 44100:
                return 0, 3

            str_data = f.readframes(nframes)
            f.close()

        except:
            return 0, 4

        wave_data = np.frombuffer(str_data, dtype=np.short)
        wave_data.shape = -1, nchannels
        wave_data = wave_data.T

        if wave_data.shape[1] < framerate * 1:
            return 0, 5

        chunk_size = framerate * window // 1000
        chunk_interval = framerate * interval // 1000
        chunk_cnt = (wave_data.shape[1] - chunk_size) // chunk_interval + 1

        wave_buf = np.zeros([chunk_cnt, chunk_size])

        for n in range(chunk_cnt):
            pos = n * chunk_interval
            wave_buf[n, :] = wave_data[0, pos:pos + chunk_size]

        fft_dat = np.fft.fft(wave_buf)

        n = fft_dat.shape[-1] // 200
        fft_dat = np.abs(fft_dat) / (chunk_size // 2)
        fft_dat = fft_dat[:, -n * 200:-n * 100].reshape([-1, 100, n])
        fft_dat = np.average(fft_dat, 2)

        return chunk_cnt, np.asarray(fft_dat)

    def visualize(self, xs, estimates, answers):
        for n in range(len(xs)):
            path = self.va_paths[self.va_indices[n]]
            self.wavPlayer(path[17:])
            est = np.argmax(estimates[n])
            ans = np.argmax(answers[n])

            if est == ans:
                print('{}: correct'.format(self.target_names[est]))
            else:
                print("{}: wrong({})".format(self.target_names[est], self.target_names[ans]))

    def wavPlayer(self, filepath):
        src = """
        <head>
        <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
        <title>Audio Player</title>
        </head>
        
        <body>
        <audio controls="controls" style="width:600px" >
          <source src="files/%s" type="audio/wav" />
          Your browser does not support the audio element.
        </audio>
        </body>
        """ % filepath
        HTML(src)
