import tensorflow as tf
import numpy as np

# TensorFlow 모델을 TensorFlow Lite 모델로 변환
converter = tf.lite.TFLiteConverter.from_saved_model('C:\\Users\\spong\\myvenv\\dl_model\\sound_classifier_model')
tflite_model = converter.convert()

# TensorFlow Lite 모델 저장
tflite_model_path = 'C:\\Users\\spong\\myvenv\\dl_model\\sound_classifier_model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

# TensorFlow Lite Interpreter 로드
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# 입력 데이터 준비
input_data = 'C:\\Users\\spong\\myvenv\\dl_model\\102305-6-0-0.npy'  # 적절한 입력 데이터를 준비하세요
input_data = np.load(input_data)  # 입력 데이터를 npy 파일에서 로드

# 입력 데이터 형태 변경
input_data = np.expand_dims(input_data, axis=-1)  # 마지막 차원 추가
input_data = np.expand_dims(input_data, axis=0)   # 배치 차원 추가

# TensorFlow Lite 모델 실행
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# 예측 결과 확인
predictions = interpreter.get_tensor(output_details[0]['index'])
class_index = np.argmax(predictions)
print('Predicted class index:', class_index)
