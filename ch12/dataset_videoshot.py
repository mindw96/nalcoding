import os

import numpy as np

from dataset import Dataset

import cv2

class VideoShotDataset(Dataset):
    def __init__(self, filenames, timesteps=5):
        super(VideoShotDataset, self).__init__('Videoshot', 'binary')

        self.video_create_cache(filenames)

        self.frames, self.marks = self.video_load_cache(filenames)

        self.set_timesteps(timesteps)

    def set_timesteps(self, timesteps):
        self.timesteps = timesteps
        self.input_shape = [timesteps + 1, 90, 120, 3]
        self.output_shape = [timesteps + 1, 1]

    @property
    def train_count(self):
        return 2000

    def __str__(self):
        return '{}({}, {} frames, {} shots, {} train_data)'.format(self.name, self.mode, len(self.frames),
                                                                   np.sum(self.shots), self.train_count)

    def video_create_cache(self, filenames):
        movie_path = '../ch12/movie'
        cache_path = '../ch12/cache'

        if not os.path.exists(cache_path):
            os.mkdir(cache_path)

        for filename in filenames:
            movie_fname = movie_path + filename
            cache_fname = cache_path + filename + '.npy'

            if os.path.exists(cache_fname):
                print('{}: cache  file is found => use cache'.format(filename))
                continue

            if not os.path.exists(movie_fname):
                print('{}: file is not found => ERROR'.format(filename))
                assert 0

            print('{}: creating cache file...'.format(filename))

            cap = cv2.VideoCapture(movie_fname)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

            shot_idxs = list(np.sort(np.random.randint(0, frame_count-400,100)))
            thumbs = np.zeros([100, 4, 90, 120,3])
            sn = 0

            for fn in range(frame_count-400):
                ret = cap.grab()
                if fn == shot_idxs[sn]:
                    for k in range(4):
                        _, frame = cap.retrieve(0)
                        cap.grab()
                        thumbs[sn, k] = cv2.resize(frame, (120, 90))
                    sn += 1
                    if sn >= 100:
                        break

            cap.close()
            np.save(cache_fname, thumbs)

        print('Creating thumbnail cache is done')

