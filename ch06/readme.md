# 6장. 복합 출력의 처리 방법 : 오피스31 다차원 분류 신경망
6장은 기존에 사용했던 경사하강법을 통해서 최적화를 하지않고 Adam이라는 알고리즘을 사용해서 최적화를 구한다.  
이동평균 방식을 통해서 모멘텀값과 2차 모멘텀 값을 구하는데 이러한 방식을 사용함으로써 얻는 장점은 각 파라미터별로 모멘텀 정보를 구해서 계산하기 때문에  
학습률로 부터 강인해진다는 점이다. 하지만 매번 연산마다 파라미터의 값 외에도 모멘텀과 2차 모멘텀 값도 업데이트해야하기 때문에 연산량이 3배로 증가한다는 단점이 있다.  
  
# 느낀것
구현은 생각보다 쉬웠다. 파라미터 업데이트 과정에서 모멘텀 값과 2차 모멘텀 값을 구하고 처리횟수 등을 추가적으로 업데이트하고 관리해주면 된다.  
delta 반환 과정에서 2차 모멘텀의 제곱근값으로 나눌 때 2차 모멘텀이 0이 나오면 오류가 발생하기 때문에 epsilon을 더해주어야하는 점을 주의해야한다.  

## 보스턴 대학교 사이트에서 꽃 분류 데이터셋 다운받기
오피스31 데이터셋에 대한 다운로드 링크는 https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view 입니다.