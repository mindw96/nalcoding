# 7장. 간단한 합성곱 모델 : 꽃 이미지 분류 신경망
7장은 드디어 CNN인 Convolution Layer를 구현한다.  
Convolution 연산은 3종류가 있다. 단순하게 7중 반복문으로 구현하는 방법,   
커널의 이동 반복 부분을 벡터 내적으로 구현하여 5중 반복문으로 줄인방법,  
2차원으로 줄여서 행렬곱 연산을 통해서 2중 반복문 만으로 해결하도록 줄인방법이다.  
그동안 간단하게 api로만 사용하여서 이론적인 부분만 알고 코드로 구현하는건 몰랐는데  
직접 넘파이로 구현을 해보니 여간 복잡하고 어려운게 아닐 수 없었다.  
가장 조심하며 신경써야할 부분은 커널이 이동하며 이미지를 벗어나는 부분이 생길 수 있는데  
이를 버퍼를 통해서 미리 처리하는 부분이다. 이 부분을 신경써서 구현해야한다.  


