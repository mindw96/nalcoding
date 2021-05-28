# 9장. 인셉션 모델과 레스넷 모델 : 꽃 이미지 분류 신경망
9장에서는 Inception 모델과 Resnet 모델을 직접 구현했다.  
이 모델들의 특징은 Convolution 레이어를 깊게 쌓으면 정확도가 높아진다는 특징이 있지만  
모델의 깊이가 깊어지면 깊어질수록 역전파시 Vanishing Gradient가 발생하는데 이를 해결하기 위한   
Inception 구조와 Skip Connection 구조를 사용한것이 특징이다.  
Inception 구조를 구현하기 위해 Add, Serial, Parallel 구조를 사용했다.  
이러한 구조는 간단하게 매크로 구조를 통해서 구현하였으며 Forward와 Backprop 함수를 수정하는것이 중요하다.  
