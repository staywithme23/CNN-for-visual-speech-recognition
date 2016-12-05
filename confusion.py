import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

import vggmodel
img_width, img_height = 175, 175
model = vggmodel.create_model(img_width, img_height)
#weights_path="vgg-finetune-model.h5"
weights_path="model/weights-VggFinetune-17-0.65.f5"
model.load_weights(weights_path)
# import val data to play with
validation_data_dir = 'dataset-small/val'
img_width, img_height = 175, 175
test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        shuffle=False,
        class_mode='categorical')
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9),
              metrics=['accuracy'])
prediction = model.predict_generator(validation_generator, 400)
prediction_class = np.argmax(prediction, axis=1)
ans_class = np.array([0] * 20 + [1] * 20 + [2] * 20 + [3] * 20 + [4] * 20 + [5] * 20 + [6] * 20 + [7] * 20 + [8] * 20 + [9] * 20 + [10] * 20 + [11] * 20 + [12] * 20 + [13] * 20 + [14] * 20 + [15] * 20 + [16] * 20 + [17] * 20 + [18] * 20 + [19] * 20)
class_names=["Stop navigation", "Excuse me", "I am sorry", "Thank you", "Good bye", "I love this grace", "Nice to meet you", "You are welcome", "How are you", "Have a good time", "Begin", "Choose", "Connection", "Navigation", "Next", "Previous", "Start", "Stop", "Hello", "Web"]
#print(prediction[2])
#print(prediction[3])
print(prediction_class)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=270)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

#    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        plt.text(j, i, cm[i, j],
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(ans_class, prediction_class)
np.set_printoptions(precision=2)

## Plot non-normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names,
#                      title='Confusion matrix, without normalization')
#
## Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
