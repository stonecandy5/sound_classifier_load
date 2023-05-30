import tensorflow as tf
import numpy as np

model_path = 'C:\\Users\\spong\\myvenv\\dl_model\\sound_classifier_model'

model = tf.saved_model.load(model_path)

# 입력 데이터 준비
input_data = 'C:\\Users\\spong\\myvenv\\dl_model\\101415-3-0-2.npy'  # 적절한 입력 데이터를 준비하세요
input_data = np.load(input_data)  # 입력 데이터를 npy 파일에서 로드

# 입력 데이터 형태 변경
input_data = np.expand_dims(input_data, axis=-1)  # 마지막 차원 추가
input_data = np.expand_dims(input_data, axis=0)   # 배치 차원 추가

# 모델 예측
predictions = model(input_data)
class_index = tf.argmax(predictions, axis = 1 )
# 예측 결과 확인
temp = str(class_index)

pre = temp.split(" ")
# 예측 결과 확인
print(predictions)
print('case :',pre[0][11])

