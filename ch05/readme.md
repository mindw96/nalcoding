# 5장. 다층 퍼셉트론 모델 구조 : 꽃 이미지 분류 신경망
5장은 앞으로 사용할 mlp 모델과 데이터셋 클래스, 수학 유틸 클래스를 만들었다.  
  
mlp_model은 학습과 테스트가 진행되도록 설계됐다. 학습은 회귀, 이진분류, 다중 분류로 나눠지기 때문에  
데이터셋에 따라 학습을 달리 해야하기 때문에 데이터셋 클래스에서 처리 후 결과만 반환받는다.
  
dataset 파일은 데이터셋의 이름과 모드를 입력받아서 각 모드에 맞는 오차 함수를 사용해서 학습을 진행해주는 역할을 한다.  
  
mathutil 파일에서는 활성화 함수나 오차 함수 등 수학적 계산식이 필요한 함수들을 따로 적어서 간편히 호출해서 사용할 수 있도록 했다.
  
5장은 꽃 이미지를 통해 꽃의 종류를 분류하는 문제를 해결하기 위해 별도의 데이터셋 파일을 만들었다.  
이 파일은 먼저 만든 데이터셋을 상속 받으면서 시각화 기능만 추가했다.  
클래스가 5개이기 때문에 softmax cross entropy를 사용했다.

# 수정사항
matplotlib를 활용해서 그림을 그릴때 subplot의 문법이 바뀌어 반복문마다 subplot을 그리도록 설정했다.


## 캐글 사이트에서 꽃 분류 데이터셋 다운받기
꽃 분류 데이터셋 접근 경로는 https://www.kaggle.com/alxmamaev/flowers-recognition 입니다.<br/>

