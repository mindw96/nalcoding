# 8장. 다섯 가지 정규화 확장 : 꽃 이미지 분류 신경망
8장에서는 여러가지 정규화 기법을 활용해서 학습의 정확도를 높힌다.  
L1, L2 정규화, DropOut, Batch Normalization, Noise Injection 기법이다.  
L1, L2 정규화의 경우 수식만 넘파이로 구현하면 되기 때무에 쉽게 구현이 가능하다.  
DropOut의 경우 랜덤 확률로 구한 mask를 곱합으로써 간단하게 구현했다.  
Bacth Normalization의 경우는 이동 평균을 구하는 과정을 구현하는 부분이 다소 복잡하다. 저장할 버퍼를 미리 선언 후 학습해야한다.  
주의할 점은 정규화 과정에서 표준편차로 나누는데 이때 분산이 0이면 표준편차도 0이 되기 때문에 엡실론 값을 더해줘야한다.  
Noise 주입은 임의의 Noise를 구해서 + 연산으로 원래 값에 조금의 값을 더하는 것으로 구현한다.  
  
파라미터를 시각해보면 각 정규화의 효과를 쉽게 확인 할 수 있다.  
L1의 경우 기존 종모양에서 비틀린 삼각형에 가깝게 변화하고 0에 가까운 값들이 증가해 가운데 선이 생기는 모습이다.  
L2의 경우 기존 종모양을 양 옆에서 누른 모양으로 최대값과 최소값 양 끝쪽의 값들이 줄어들고 0에 가까운 값들이 늘어났다.  
DropOut의 경우 기존 종모양보다 더 정규분포 형태의 모양을 보였다.
Batch Normalization의 경우는 부분적으로 수직화됐다. 직각의 형태를 보인다.  
Noise 주입 기법은 큰 차이가 없다. 

<img src="/ch8/img/mlp_og.png" width="40%" height="30%" title="원본"></img>  
<img src="/ch8/img/mlp_l1.png" width="40%" height="30%" title="L1 규제"></img>  
<img src="/ch8/img/mlp_l2.png" width="40%" height="30%" title="L2 규제"></img>  
<img src="/ch8/img/mlp_dropout.png" width="40%" height="30%" title="DopOut"></img>  
<img src="/ch8/img/mlp_batch_normalization.png" width="40%" height="30%" title="Batch_Normalization"></img>  
<img src="/ch8/img/mlp_noise.png" width="40%" height="30%" title="Noise Injection"></img>  

