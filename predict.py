import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

import vggmodel
from scipy import misc
from keras.preprocessing import image as image_utils

def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
        # Zero-center by mean pixel
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
    return x

#17 is start
path = '/home/adwin5/testing/CNN-example/dataset/val/16/M01_words_06_09result.jpg'
#input = misc.imread('/home/adwin5/testing/CNN-example/dataset/val/16/M01_words_06_09result.jpg')
#input = input.astype(np.float32)
#input = np.reshape(input, (1, 3, 175, 175))

print("[INFO] loading and preprocessing image...")
image = image_utils.load_img(path, target_size=(175, 175))
image = image_utils.img_to_array(image)
input_image = np.expand_dims(image, axis=0)
#input_image = preprocess_input(image, 'th')
print(input_image)

img_width, img_height = 175, 175
model = vggmodel.create_model(img_width, img_height)
#weights_path="vgg-finetune-model.h5"
weights_path="model/weights-VggFinetune-17-0.65.f5"
model.load_weights(weights_path)
## import val data to play with
#validation_data_dir = 'dataset-small/val'
#img_width, img_height = 175, 175
#test_datagen = ImageDataGenerator(rescale=1./255)
#validation_generator = test_datagen.flow_from_directory(
#        validation_data_dir,
#        target_size=(img_height, img_width),
#        batch_size=32,
#        shuffle=False,
#        class_mode='categorical')

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9),
              metrics=['accuracy'])
prediction = model.predict(input_image)
prediction_class = np.argmax(prediction, axis=1)
class_names=["Stop navigation", "Excuse me", "I am sorry", "Thank you", "Good bye", "I love this grace", "Nice to meet you", "You are welcome", "How are you", "Have a good time", "Begin", "Choose", "Connection", "Navigation", "Next", "Previous", "Start", "Stop", "Hello", "Web"]
#print(prediction[2])
#print(prediction[3])
print(class_names[prediction_class[0]-1])
print(prediction_class[0])

