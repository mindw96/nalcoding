import numpy as np

from dataset import Dataset

# 문맥자유문법으로 표현된 문법 규칙에 따라 수식에 해당하는 문장을 생성하고 검사하는 데데 필요한 데이터셋을 생성한다.

# 시계열 데이터의 최소와 최대 길이를 설정한다.
MIN_LENGTH = 10
MAX_LENGTH = 40
# 모든 영어 소문자와 십진수 숫자들을 리스트로 정의한다.
ALPHA = [chr(n) for n in range(ord('a'), ord('z') + 1)]
DIGIT = [chr(n) for n in range(ord('0'), ord('9') + 1)]
# 여러가지 부호와 기호등을 정의한다.
EOS = ['$']
ADDOP = ['+', '-']
MULTOP = ['*', '/']
LPAREN = ['(']
RPAREN = [')']
SYMBOLS = EOS + ADDOP + MULTOP + LPAREN + RPAREN
ALPHANUM = ALPHA + DIGIT
ALPHABET = SYMBOLS + ALPHANUM

# 문법 기호들을 상수로 임베딩한다.
S = 0  # sent
E = 1  # exp
T = 2  # term
F = 3  # factor
V = 4  # variable
N = 5  # number
V2 = 6  # var_tail

# 문법 규칙을 정의한다.
RULES = {
    S: [[E]],
    E: [[T], [E, ADDOP, T]],
    T: [[F], [T, MULTOP, F]],
    F: [[V], [N], [LPAREN, E, RPAREN]],
    V: [[ALPHA], [ALPHA, V2]],
    V2: [[ALPHANUM], [ALPHANUM, V2]],
    N: [[DIGIT], [DIGIT, N]]
}

# ACTION 테이블에서 사용할 기호들을 정의한다.
E_NEXT = EOS + RPAREN + ADDOP
T_NEXT = E_NEXT + MULTOP
F_NEXT = T_NEXT
V_NEXT = F_NEXT
N_NEXT = F_NEXT

# LALR 파서에서 사용하는 ACTION 테이블을 스택 구조로 정의한다.
# 0은 수락, 양수는 이동, 음수는 축소이며 오류는 따로 지정하지 않는다. ACTION 테이블 결과에서 항목이 발견되지 않으면 오류이다.
action_table = {
    0: [[ALPHA, 6], [DIGIT, 7], [LPAREN, 8]],
    1: [[ADDOP, 9], [EOS, 0]],
    2: [[MULTOP, 10], [E_NEXT, -1, E]],
    3: [[T_NEXT, -1, T]],
    4: [[F_NEXT, -1, F]],
    5: [[F_NEXT, -1, F]],
    6: [[ALPHANUM, 6], [V_NEXT, -1, V]],
    7: [[DIGIT, 7], [N_NEXT, -1, N]],
    8: [[ALPHA, 6], [DIGIT, 7], [LPAREN, 8]],
    9: [[ALPHA, 6], [DIGIT, 7], [LPAREN, 8]],
    10: [[ALPHA, 6], [DIGIT, 7], [LPAREN, 8]],
    11: [[V_NEXT, -2, V]],
    12: [[N_NEXT, -2, N]],
    13: [[RPAREN, 16], [ADDOP, 9]],
    14: [[MULTOP, 10], [T_NEXT, -3, T]],
    15: [[F_NEXT, -3, F]],
    16: [[F_NEXT, -3, F]],
}
# 우변의 길이를 축소하고 이동할 상태를 결정하는 값들을 정의한다.
goto_table = {
    0: {E: 1, T: 2, F: 3, V: 4, N: 5},
    6: {V: 11},
    7: {N: 12},
    8: {E: 13, T: 2, F: 3, V: 4, N: 5},
    9: {T: 14, F: 3, V: 4, N: 5},
    10: {F: 15, V: 4, N: 5},
}


# 어떤 패턴이 오토마타로부터 생성되었는지를 판정하는 패턴 검사기에 사용될 데이터셋 클래스다.
class AutomataDataset(Dataset):
    def __init__(self):
        super(AutomataDataset, self).__init__('automata', 'binary')
        self.input_shape = [MAX_LENGTH + 1, len(ALPHABET)]
        self.output_shape = [1]

    @property
    # epoch를 설정한다.
    def train_count(self):
        return 10000

    def get_train_data(self, batch_size, nth):
        return self.automata_generate_data(batch_size)

    def get_validate_data(self, count):
        return self.automata_generate_data(count)

    def get_test_data(self):
        return self.automata_generate_data(1000)

    def get_visualize_data(self, count):
        return self.automata_generate_data(count)

    def automata_generate_data(self, count):
        # 0으로 초기화된 버퍼를 생성한다.
        xs = np.zeros([count, MAX_LENGTH, len(ALPHABET)])
        ys = np.zeros([count, 1])

        #
        for n in range(count):
            # is_correct의 값을 0과 1이 순서대로 반복하도록 설정
            is_correct = n % 2

            # 만약 옳은 문장이라면
            if is_correct:
                # 생성 메소드를 통해 생성한다.
                sent = self.automata_generate_sent()
            # 틀린 문장이라면
            else:
                # 무한 반복문을 통해서 올바르게 생성한 문장의 일부분은 랜덤으로 다른 알파벳으로 대치한다.
                while True:
                    sent = self.automata_generate_sent()
                    touch = np.random.randint(1, len(sent) // 5)
                    for k in range(touch):
                        sent_pos = np.random.randint(len(sent))
                        char_pos = np.random.randint(len(ALPHABET) - 1)
                        sent = sent[:sent_pos] + ALPHABET[char_pos] + sent[sent_pos + 1:]
                    if not self.automata_is_correct_sent(sent):
                        break
            ords = [ALPHABET.index(ch) for ch in sent]
            xs[n, 0, 0] = len(sent)
            # 원핫벡터화 하여 저장한다.
            xs[n, 1:len(sent) + 1, :] = np.eye(len(ALPHABET))[ords]
            # 옳은 문장인지 아닌지 결과를 저장한다.
            ys[n, 0] = is_correct

        return xs, ys

    # 옳은 문장을 생성할 떄 까지 무한 반복하는 함수이다.
    def automata_generate_sent(self):
        while True:
            try:
                # 문장을 생성한다.
                sent = self.automata_gen_mode(S, 0)
                # 만약 길이 제한을 벗어난다면 다시 생성한다.
                if len(sent) >= MAX_LENGTH:
                    continue
                if len(sent) <= MIN_LENGTH:
                    continue

                return sent

            except Exception:
                continue

    # 재귀적으로 문장을 생성하는 함수이다.
    def automata_gen_mode(self, node, depth):
        # 재귀 반복의 제한을 설정한다.
        if depth > 30:
            raise Exception
        # node가 규칙에 없다면 0으로 처리한다.
        if node not in RULES:
            assert 0

        # 규칙에서 node의 값을 가져온다.
        rules = RULES[node]
        # 규칙에서 node의 길이만큼 랜덤한 길이의 정수를 생성한다.
        nth = np.random.randint(len(rules))
        sent = ''
        # 반복문을 통해 각 항에 해당하는 문자열을 생성하여 sent에 덧붙이며 문장을 생성한다.
        for term in rules[nth]:
            if isinstance(term, list):
                pos = np.random.randint(len(term))
                sent += term[pos]
            # 만약 term이 리스트가 아니라면 문법기호이기 때문에 재귀호출하여 그 다음 문자열을 생성한다.
            else:
                sent += self.automata_gen_mode(term, depth + 1)

        return sent

    # 문장이 옳바른 수식 표현인지 검사하는 함수이다.
    @classmethod
    def automata_is_correct_sent(self, sent):
        sent = sent + '$'
        states, pos, nextch = [0], 0, sent[0]

        while True:
            actions = action_table[states[-1]]
            found = False
            # 반복문을 통해 ACTION을 차례로 수행한다.
            for pair in actions:
                if nextch not in pair[0]:
                    continue
                found = True
                # 만약 action이 0이라면 성공을 반환하는 수락
                if pair[1] == 0:
                    return True
                # 만약 action이 양수라면 새로운 상태를 스택에 추가하고 다음 위치의 입력 문자를 읽어들이는 이동
                elif pair[1] > 0:
                    states.append(pair[1])
                    pos += 1
                    nextch = sent[pos]
                    break
                # 둘 다 아니라면 문법 규칙 우변의 길이만큼 스택을 줄인 후 새로운 스택 꼭대기 상태와 처리된 문법 기호에 따라 새로운 상태로 이동하는 축소
                else:
                    states = states[:pair[1]]
                    goto = goto_table[states[-1]]
                    goto_state = goto[pair[2]]
                    states.append(goto_state)
                    break
            # 분석 실패를 반환하는 오류
            if not found:
                return False

    # 시각화를 지원하는 함수이다.
    def visualize(self, xs, est, ans):
        for n in range(len(xs)):
            length = int(xs[n, 0, 0])
            sent = np.argmax(xs[n, 1:length + 1], axis=1)
            text = "".join([ALPHABET[letter] for letter in sent])

            answer, guess, result = '잘못된 패턴', '탈락추정', 'X'

            if ans[n][0] > 0.5:
                answer = '올바른 패턴'
            if est[n][0] > 0.5:
                guess = '합격추정'
            if ans[n][0] > 0.5 and est[n][0] > 0.5:
                result = 'O'
            if ans[n][0] < 0.5 and est[n][0] < 0.5:
                result = 'O'

            print('{}: {} => {}({:4.2f}) : {}'.format(text, answer, guess, est[n][0], result))
