# MNIST Classification Project

Overview
This project aims to build a classification model using the MNIST dataset to recognize handwritten digits. The MNIST dataset is a large database of handwritten digits that is commonly used for training various image processing systems. The project utilizes neural network models to learn from the dataset and predict the digit presented in an image.

Usage
Dataset Download and Preprocessing
Download the dataset and extract it.
Use the dataset.py file to load and preprocess the dataset.
Model Selection and Training
Select the LeNet-5 or custom MLP model from the model.py file.
Use the main.py file to train and evaluate the model.
Project Execution
Run the main.py file to execute the project.
Statistical Graphs Creation
The training process of the models is evaluated by visualizing the average loss values and accuracy for both the training and testing datasets.


The graphs show that the LeNet-5 model's training loss decreases progressively, while the test loss shows some volatility over epochs. In terms of accuracy, the training accuracy continues to improve, reducing the gap with the test accuracy. This indicates a robust learning process, albeit with room for improvement in model generalization on unseen data.

![LeNet Loss over Epochs](https://github.com/mungmung53/seoultech/assets/161430566/c89502af-833c-4c78-893d-816c671383f2)

Model Performance Comparison
The LeNet-5 model consistently shows lower loss values and achieves higher accuracy compared to the custom MLP model. The LeNet-5 model reaches an accuracy of approximately 89% on the test dataset, similar to known benchmarks. The custom MLP model has a slightly lower test accuracy of about 85%. This suggests that LeNet-5 has a more suitable architecture for image recognition tasks within this context.

Applying Regularization Techniques
To improve the LeNet-5 model's performance, regularization techniques such as dropout and data augmentation were applied. Dropout helped prevent overfitting and improve the model's generalization capabilities. Data augmentation, primarily applied to the training data, allowed the model to learn from a more varied dataset. The application of these techniques led to a performance enhancement, with the test dataset accuracy improving by approximately 3%. This underlines the effectiveness of regularization techniques in model performance optimization.

Prerequisites
Python 3.x
PyTorch
OpenCV
Additional libraries listed in requirements.txt
Contact Information
For any questions or suggestions, please send an email to: shinhyelee0503@gmail.com

_

이 프로젝트는 MNIST 데이터셋을 사용하여 손으로 쓴 숫자를 인식하는 분류 모델을 구축하는 것을 목표로 합니다.

## 사용 방법

1. **데이터셋 다운로드 및 전처리**
   - 먼저 데이터셋을 다운로드하고 압축을 해제합니다.
   - `dataset.py` 파일을 사용하여 데이터셋을 불러오고 전처리합니다.

2. **모델 선택과 훈련**
   - `model.py` 파일에서 LeNet-5 및 사용자 정의 MLP 모델을 선택합니다.
   - `main.py` 파일을 사용하여 모델을 훈련하고 평가합니다.

3. **프로젝트 실행**
   - `main.py` 파일을 실행하여 프로젝트를 실행합니다.

4. 통계 그래프 작성
- 모델의 학습 과정을 평가하기 위해 LeNet-5와 사용자 정의 MLP 모델에 대해 각각의 학습 및 테스트 데이터셋에 대한 평균 손실값과 정확도를 시각화했습니다.
- 그래프에서 LeNet-5 모델은 학습 손실이 점진적으로 감소하는 반면, 테스트 손실은 에포크를 거듭하면서 약간의 변동성을 보였습니다.
- 정확도 면에서는 학습 정확도가 지속적으로 향상되어 테스트 정확도와의 격차가 줄어드는 추세를 나타냈습니다.
- 사용자 정의 MLP 모델은 학습 손실이 LeNet-5보다 다소 높게 유지되었으나, 정확도는 비슷한 수준으로 나타났습니다.
- 이러한 결과는 두 모델의 학습 곡선을 통해 명확히 확인할 수 있습니다.

![LeNet Loss over Epochs](https://github.com/mungmung53/seoultech/assets/161430566/c89502af-833c-4c78-893d-816c671383f2)


5. 모델 성능 비교
- LeNet-5 모델과 사용자 정의 MLP 모델의 성능을 비교한 결과, LeNet-5가 일관되게 더 낮은 손실값을 보이며 높은 정확도를 달성했습니다.
- 테스트 데이터셋에서 LeNet-5의 정확도는 약 89%로, 기존에 알려진 성능과 유사한 결과를 보였습니다.
- 반면, 사용자 정의 MLP 모델은 테스트 정확도가 약 85%로 조금 더 낮게 나타났습니다.
- 이는 LeNet-5가 이미지 인식에 더 적합한 구조를 가지고 있음을 시사합니다.
- 추가적으로, 두 모델의 손실 및 정확도 추이를 분석하여, 학습 과정에서의 과적합 여부와 학습 효율성을 평가할 수 있습니다.

6. 정규화 기법 적용
- LeNet-5 모델의 성능을 향상시키기 위해 드롭아웃과 데이터 증강 기법을 정규화 기법으로 적용하였습니다.
- 드롭아웃은 과적합을 방지하고 모델의 일반화 능력을 향상시키는 데 도움이 되었습니다.
- 데이터 증강은 주로 학습 데이터에 변형을 가하여 더 다양한 데이터셋에서 모델이 학습할 수 있도록 하였습니다.
- 이 두 기법을 적용한 결과, 테스트 데이터셋에서의 정확도가 초기 모델 대비 약 3% 향상되었습니다.
- 이러한 결과는 정규화 기법이 모델의 성능을 개선하는 데 효과적임을 보여줍니다.
- 또한, 이러한 기법들이 특히 훈련 데이터에만 적용되었기 때문에, 테스트 데이터셋에서의 성능 향상은 모델의 일반화 능력이 개선되었음을 의미합니다.


## 필수 구성 요소

- Python 3.x
- PyTorch
- OpenCV
- 기타 필요한 라이브러리 (requirements.txt 참조)



## 연락처 정보

질문이나 제안이 있으시면 이메일을 보내주세요: shinhyelee0503@gmail.com
