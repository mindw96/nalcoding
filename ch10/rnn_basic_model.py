import numpy as np

from ch9.cnn_ext_model import CnnExtModel


# RNN의 기본 틀이 되는 클래스이다.
class RnnBasicModel(CnnExtModel):
    # rnn 레이어의 파라미터들을 할당해주는 함수이다.
    def alloc_rnn_layer(self, input_shape, hconfig):
        inseq = self.get_conf_param(hconfig, 'inseq', True)
        outseq = self.get_conf_param(hconfig, 'outseq', True)
        # 입력값이 시계열 데이터인지 확인한다. 만약 시계열 데이터라면
        if inseq:
            # 입력 레이어의 형태를 timesteps와 timefeats로 언팩킹한다.
            timesteps1, timefeats = input_shape
        # 시계열 데이터가 아니라면
        else:
            # hconfig의 timesteps를 불어와 1을 더한 값을 timesteps1로 설정한다.
            timesteps1 = self.get_conf_param(hconfig, 'timesteps') + 1
            # 입력 레이어의 차원을 다 곱해서 timefeats로 설정한다.
            timefeats = np.prod(input_shape)

        # hconfig에서 recur_size를 불러와서 변수로 할당한다.
        recur_size = self.get_conf_param(hconfig, 'recur_size')
        # timefeats와 recur_size를 더해서 확장된 입력 차원을 설정한다.
        ex_inp_dim = timefeats + recur_size
        # 가중치와 Bias를 할당한다.
        weight, bias = self.alloc_param_pair([ex_inp_dim, recur_size])
        # 출력값이 시계열 데이터라면
        if outseq:
            # 출력의 형태를 [timesteps1, reucr_size] 형태의 리스트로 설정한다.
            output_shape = [timesteps1, recur_size]
        # 시계열 데이터가 아니라면
        else:
            # 출력의 형태를 [recur_size] 형태로 설정한다.
            output_shape = [recur_size]

        # Rnn의 정보들을 리스트 형식으로 패킹해서 전달한다.
        rnn_info = [inseq, outseq, timesteps1, timefeats, recur_size]

        return {'w': weight, 'b': bias, 'info': rnn_info}, output_shape

    # RNN 레이어의 순전파를 수행하는 함수이다.
    def forward_rnn_layer(self, x, hconfig, pm):
        inseq, outseq, timesteps1, timefeats, recur_size = pm['info']
        mb_size = x.shape[0]

        if inseq:
            x_slices = x[:, 1:, :].transpose([1, 0, 2])
            lengths = x[:, 0, 0].astype(np.int32)
            timesteps = np.max(lengths)
        else:
            x_slice = x
            timesteps = timesteps1 - 1
            lengths = [timesteps] * mb_size


        recurrent = np.zeros([mb_size, recur_size])
        outputs, aux_steps = [], []

        for n in range(timesteps):
            if inseq:
                x_slice = x_slices[n]
                ex_inp = np.hstack([x_slice, recurrent])
                affine = np.matmul(ex_inp, pm['w'], pm['b'])
                recurrent = self.activate(affine, hconfig)

                outputs.append(recurrent)
                aux_steps.append(ex_inp)

            if outseq:
                output = np.zeros([mb_size, timesteps1, recur_size])
                output[:, 0, 0] = lengths
                output[:, 1: :] = np.asarray(outputs).transpose([1, 0, 2])
            else:
                output = np.zeros([mb_size, recur_size])
                for i in range(mb_size):
                    output[i] = outputs[lengths[i]-1][i]

        return output, [x, lengths, timesteps, outputs, aux_steps]



