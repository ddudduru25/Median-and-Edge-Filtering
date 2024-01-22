# Median and Edge Filtering

## Digital Image Process - week6


### 기능

1. Median Filter
  * 커널 윈도우 안에 있는 화소 값의 중간값을 선택하여 화소 값이 너무 크거나 작은 노이즈를 제거
  * 비선형 필터
    
2. Bilateral Filter
  * gaussian filter와 range 가중치를 결합하여 edge를 보존하며 smoothing 수행

3. Edge Detection
  * Gradient 계산을 통해 커널을 구함
  * Gradient 계산 특성상 잡음에 취약하므로 smoothing 후 적용하는 것이 일반적

3-1. Canny Edge Detection
  * 불 분명한 edge 중 실제 edge를 찾아 분명히 함
  * threshold와의 비교를 통해 분명한 edge를 구분
    * OpenCV의 Canny함수에서 threshold값을 크게할 수록 처리시간은 증가하지만 더 많은 edge가 검출됨을 확인함


