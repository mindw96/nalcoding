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
        # 파라미터들을 불러온다.
        inseq, outseq, timesteps1, timefeats, recur_size = pm['info']
        mb_size = x.shape[0]
        # 만약 입력 레이어가 시계열 데이터라면
        if inseq:
            # 입력 데이터의 슬라이스 정보
            x_slices = x[:, 1:, :].transpose([1, 0, 2])
            # 입력 데이터의 길이 정보
            lengths = x[:, 0, 0].astype(np.int32)
            # 반복 횟수를 설정한다.
            timesteps = np.max(lengths)
        # 시계열 데이터가 아니라면
        else:
            x_slice = x
            timesteps = timesteps1 - 1
            lengths = [timesteps] * mb_size
        # 순환벡터를 0으로 초기화하여 선언한다.
        recurrent = np.zeros([mb_size, recur_size])
        # 출력물과 역전파를 위한 정보를 전달하는 리스트를 선언한다.
        outputs, aux_steps = [], []

        # timesteps만큼 반복문을 통해 순전파를 진행한다.
        for n in range(timesteps):
            # 만약 입력데이터가 시계열 데이터라면
            if inseq:
                # x_slices에서 순차적으로 x_slice를 사용한다.
                x_slice = x_slices[n]
            # x_slice를 순환벡터에 합쳐서 확장 입력 변수를 선언한다.
            ex_inp = np.hstack([x_slice, recurrent])
            # 확장 입력값과 파라미터들을 행렬곱 연산을 통해 affine 값을 계산한다.
            affine = np.matmul(ex_inp, pm['w']) +  pm['b']
            # affine 값을 활성화 함수를 통과시켜 얻은 값을 순확벡터로 다시 선언한다.
            recurrent = self.activate(affine, hconfig)
            # 순환벡터 값을 결과물 리스트에 추가한다.
            outputs.append(recurrent)
            # 확장 입력 값을 리스트에 추가한다.
            aux_steps.append(ex_inp)
            # 만약 입력데이터가 시계열 데이터라면
        if outseq:
            # 0으로 출력값 버퍼를 초기화하여 선언한다.
            output = np.zeros([mb_size, timesteps1, recur_size])
            # 출력값 버퍼에서 길이를 추출한다.
            output[:, 0, 0] = lengths
            # 순환벡터들이 저장된 출력값 리스트를 시간대 순으로 전치 하여 출력값 버퍼에 대입한다.
            output[:, 1:, :] = np.asarray(outputs).transpose([1, 0, 2])
            # 만약 아니라면
        else:
            # 0으로 초기화된 출력값 버퍼를 선언한다.
            output = np.zeros([mb_size, recur_size])
            # 미니배치 크기만큼 반복문을 돌린다.
            for n in range(mb_size):
                # 출력값을 순서대로 순환벡터들 중 마지막 위치에서 찾는다.
                output[n] = outputs[lengths[n] - 1][n]
        return output, [x, lengths, timesteps, outputs, aux_steps]

    # 역전파를 처리하는 함수이다.
    def backprop_rnn_layer(self, G_y, hconfig, pm, aux):
        inseq, outseq, timesteps1, timefeats, recur_size = pm['info']
        x, lengths, timesteps, outputs, aux_steps = aux
        mb_size = x.shape[0]

        G_weight = np.zeros_like(pm['w'])
        G_bias = np.zeros_like(pm['b'])
        G_x = np.zeros(x.shape)
        G_recurrent = np.zeros([mb_size, recur_size])

        # 만약 입력 데이터가 시계열 함수라면
        if inseq:
            # 손길 기울기의 값도 시계열 함수이므로 lengths의 정보를 G_x 앞부분에 입력한다
            G_x[:, 0, 0, ] = lengths

        # 만약 출력 데이터가 시계열 함수라면
        if outseq:
            # G_outputs는 G_y를 슬라이싱 후 전치하여 사용한다.
            G_outputs = G_y[:, 1:, :].transpose([1, 0, 2])
        else:
            # 버퍼를 생성하고 반복문을 통해 G_outputs의 값들의 뒤에서부터 G_y의 값을 입력한다.
            G_outputs = np.zeros([timesteps, mb_size, recur_size])
            for n in range(mb_size):
                G_outputs[lengths[n]-1, n, :] = G_y[n]

        for n in reversed(range(0, timesteps)):
            G_recurrent += G_outputs[n]

            ex_inp = aux_steps[n]

            G_affine = self.activate_derv(G_recurrent, outputs[n], hconfig)

            g_affine_weight = ex_inp.transpose()
            g_affine_input = pm['w'].transpose()

            G_weight += np.matmul(g_affine_weight, G_affine)
            G_bias += np.sum(G_affine, axis=0)
            G_ex_inp = np.matmul(G_affine, g_affine_input)

            if inseq:
                G_x[:, n+1, :] = G_ex_inp[:, :timefeats]
            else:
                G_x[:,:] += G_ex_inp[:, timefeats:]

            G_recurrent = G_ex_inp[:, timefeats:]

        self.update_param(pm, 'w', G_weight)
        self.update_param(pm, 'b', G_bias)

        return G_x
