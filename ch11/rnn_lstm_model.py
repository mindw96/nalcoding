import numpy as np

import mathutil
from ch10.rnn_basic_model import RnnBasicModel


class RnnLstmModel(RnnBasicModel):
    # LSTM 레이어에서 사용되는 파라미터들을 할당해주는 함수이다. LSTM에는 별도 장치 4개가 들어가기 때문에 가중치와 bias의 크기도 더 크게 할당해야한다
    def alloc_lstm_layer(self, input_shape, hconfig):
        inseq = self.get_conf_param(hconfig, 'inseq', True)
        outseq = self.get_conf_param(hconfig, 'outseq', True)
        use_state = self.get_conf_param(hconfig, 'use_state', False)

        if inseq:
            timesteps1, timefeats = input_shape
        else:
            timesteps1 = self.get_conf_param(hconfig, 'timesteps') + 1
            timefeats = np.prod(input_shape)

        recur_size = self.get_conf_param(hconfig, 'recur_size')

        ex_inp_dim = timefeats + recur_size
        weight, bias = self.alloc_param_pair([ex_inp_dim, 4 * recur_size])
        bias[0 * recur_size:1 * recur_size] = 1.0

        if outseq:
            output_shape = [timesteps1, recur_size]
        else:
            output_shape = [recur_size]

        rnn_info = [inseq, outseq, timesteps1, timefeats, recur_size, use_state]

        return {'w': weight, 'b': bias, 'info': rnn_info}, output_shape

    # LSTM 레이어의 순전파를 수행하는 함수이다.
    def forward_lstm_layer(self, x, hconfig, pm):
        inseq, outseq, timesteps1, timefeats, recur_size, use_state = pm['info']
        mb_size = x.shape[0]

        if inseq:
            x_slices = x[:, 1:, :].transpose([1, 0, 2])
            lengths = x[:, 0, 0].astype(np.int32)
            max_length = np.max(lengths)
        else:
            x_slice = x
            max_length = timesteps1 - 1
            lengths = [max_length] * mb_size

        recurrent = np.zeros([mb_size, recur_size])
        state = np.zeros([mb_size, recur_size])
        outputs, aux_steps = [], []

        for n in range(max_length):
            if inseq:
                x_slice = x_slices[n]

            ex_inp = np.hstack([x_slice, recurrent])
            affine = np.matmul(ex_inp, pm['w']) + pm['b']

            # LSTM의 특징은 3개의 게이트와 1개의 입력 장치가 있다는 점이다. 할당 함수에서 별도로 할당한 부분은 사용한다.
            # 망각 게이트는 이전 값을 삭제할지 말지를 결정하는 역할이다. 시그모이드를 통해서 1이면 이전 정보들을 보존하고 0이면 제거한다.
            forget_gate = mathutil.sigmoid(affine[:, 0 * recur_size:1 * recur_size])
            # 입력 게이트는 입력값을 업데이트할지 말지 결정한다. 시그모이드를 통해서 1이면 업데이트하고 0이면 업데이트하지않고 그대로 제거한다.
            input_gate = mathutil.sigmoid(affine[:, 1 * recur_size:2 * recur_size])
            # 출력 게이트는 출력할 정보를 결정한다. 시그모이드를 통해서 1이면 내보내고 0이면 출력하지 않고 제거한다.
            output_gate = mathutil.sigmoid(affine[:, 2 * recur_size:3 * recur_size])
            # input gate에 어떠한 정보를 입력할지를 결정한다. tanh 함수를 통해서 -1에서 1 사이의 값으로 나온다.
            block_input = mathutil.tanh(affine[:, 3 * recur_size:4 * recur_size])

            state_tmp = state
            # 현재 상태에 forget_gate를 곱하여 이전 상태를 보존할지 제거할지 계산한 후
            # tanh를 적용한 block_input 값과 input_gate의 결과를 곱하여 cell state에 추가할지 말지를 계산한다.
            state = state_tmp * forget_gate + block_input * input_gate

            # state에 tanh를 적용하고 출력 게이트를 곱하여 정보를 출력할지 말지를 정한다.
            recur_tmp = mathutil.tanh(state)
            recurrent = recur_tmp * output_gate

            # use_state의 파라미터 값에 따라서 순환벡터인지 상태 벡터인지 결정 후 결과값 리스트에 추가한다.
            if use_state:
                outputs.append(state)
            else:
                outputs.append(recurrent)

            aux_step = [ex_inp, state_tmp, block_input, input_gate, forget_gate, output_gate, recur_tmp]
            aux_steps.append(aux_step)

        if outseq:
            output = np.zeros([mb_size, timesteps1, recur_size])
            output[:, 0, 0] = lengths
            output[:, 1:, :] = np.asarray(outputs).transpose([1, 0, 2])
        else:
            output = np.zeros([mb_size, recur_size])
            for n in range(mb_size):
                output[n] = outputs[lengths[n] - 1][n]

        return output, [x, lengths, max_length, outputs, aux_steps]

    # LSTM 레이어의 역전파를 수행하는 함수이다.
    def backprop_lstm_layer(self, G_y, hconfig, pm, aux):
        inseq, outseq, timesteps1, timefeats, recur_size, use_state = pm['info']
        x, lengths, max_length, outputs, aux_steps = aux
        mb_size = x.shape[0]

        G_weight = np.zeros_like(pm['w'])
        G_bias = np.zeros_like(pm['b'])
        G_x = np.zeros(x.shape)
        G_recurrent =np.zeros([mb_size, recur_size])
        G_state = np.zeros([mb_size, recur_size])

        if inseq:
            G_x[:, 0, 0 ] = lengths

        if outseq:
            G_outputs = G_y[:, 1:, :].transpose([1, 0, 2])
        else:
            G_outputs = np.zeros([max_length, mb_size, recur_size])
            for n in range(mb_size):
                G_outputs[lengths[n]-1, n, :] = G_y[n]

        for n in reversed(range(0, max_length)):
            if use_state:
                G_state += G_outputs[n]
            else:
                G_recurrent += G_outputs[n]

            ex_inp, state_tmp, block_input, input_gate, forget_gate, output_gate, recur_tmp = aux_steps[n]

            G_recur_tmp = G_recurrent * output_gate
            G_output_gate = G_recurrent * recur_tmp

            G_state += mathutil.tanh_derv(recur_tmp) * G_recur_tmp

            G_input_gate = G_state * block_input
            G_block_input = G_state * input_gate

            G_forget_gate = G_state * state_tmp
            G_state = G_state * forget_gate

            G_affine = np.zeros([mb_size, 4*recur_size])

            G_affine[:, 0*recur_size:1*recur_size] = mathutil.sigmoid_derv(forget_gate) * G_forget_gate
            G_affine[:, 1*recur_size:2*recur_size] = mathutil.sigmoid_derv(input_gate)* G_input_gate
            G_affine[:, 2*recur_size:3*recur_size] = mathutil.sigmoid_derv(output_gate) * G_output_gate
            G_affine[:, 3*recur_size:4*recur_size] = mathutil.tanh_derv(block_input) * G_block_input

            g_affine_weight = ex_inp.transpose()
            g_affine_input = pm['w'].transpose()

            G_weight += np.matmul(g_affine_weight, G_affine)
            G_bias += np.sum(G_affine, axis=0)
            G_ex_inp = np.matmul(G_affine, g_affine_input)

            if inseq:
                G_x[:, n+1, :] = G_ex_inp[:, :timefeats]
            else:
                G_x[:,:] += G_ex_inp[:,:timefeats]

            G_recurrent = G_ex_inp[:, timefeats:]

        self.update_param(pm, 'w', G_weight)
        self.update_param(pm, 'b', G_bias)

        return G_x
