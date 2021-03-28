import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
import sklearn.metrics 
from sklearn.metrics import classification_report, confusion_matrix
import os
from google.colab import drive
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,\
                         Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import optimizers
import math
from keras import callbacks
from datetime import datetime
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

%load_ext tensorboard
import tensorboard
tensorboard.__version__

drive.mount('/content/drive')

datadir = '/content/drive/MyDrive/ObjectClassificationRandomSplit'
trainingset = datadir+'/randtrain/'
testset = datadir + '/randtest/'

lr = 0.01
regl2 = 0.001
batch_size = 128

train_datagen = ImageDataGenerator(
    rescale = 1. / 255,\
    zoom_range=0.1,\
    rotation_range=10,\
)
train_generator = train_datagen.flow_from_directory(
    directory=trainingset,
    target_size=(118, 118),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True)

test_datagen = ImageDataGenerator(
    rescale = 1. / 255)

test_generator = test_datagen.flow_from_directory(
    directory=testset,
    target_size=(118, 118),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

num_samples = train_generator.n
num_classes = train_generator.num_classes
input_shape = train_generator.image_shape

classnames = [k for k,v in train_generator.class_indices.items()]

print("Image input %s" %str(input_shape))
print("Classes: %r" %classnames)

print('Loaded %d training samples from %d classes.' %(num_samples,num_classes))
print('Loaded %d test samples from %d classes.' %(test_generator.n,test_generator.num_classes))


def OwnCNN(input_shape, num_classes, regl2 = 0.001, lr=0.001):

    model = Sequential()
    #first Convolutional
    model.add(Conv2D(filters = 32, input_shape = input_shape, kernel_size = (5,5), strides=(1,1), padding ='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (5,5), strides = (3,3), padding = 'same'))
    model.add(BatchNormalization())

    #second Convolutional
    model.add(Conv2D(filters = 128, input_shape = input_shape, kernel_size = (5,5), strides=(1,1), padding ='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (6,6), strides = (4,4), padding = 'same'))
    model.add(BatchNormalization())

    #third Convolutional
    model.add(Conv2D(filters = 128, input_shape = input_shape, kernel_size = (5,5), strides=(1,1), padding ='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (10,10), strides = (10,10), padding = 'same'))
    model.add(BatchNormalization())

    model.add(Flatten())
    shape = (input_shape[0]*input_shape[1]*input_shape[2],)

    # D1 Dense Layer
    model.add(Dense(512, input_shape=shape, kernel_regularizer=regularizers.l2(regl2)))
    model.add(Activation('relu'))
    # Dropout
    model.add(Dropout(0.6))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
   
    opt = optimizers.Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer = opt, metrics=['accuracy'])
    return model

model = OwnCNN(input_shape,num_classes, regl2, lr)
model.summary()

def lr_scheduler(epoch):
  initial_lrate = lr
  drop = 0.5
  epochs_drop = 7.0
  lrate = initial_lrate * math.pow(drop,  
          math.floor((1+epoch)/epochs_drop))
  return lrate
   

steps_per_epoch=train_generator.n//train_generator.batch_size
val_steps=test_generator.n//test_generator.batch_size+1
epochs = 100
early_stop = callbacks.EarlyStopping(monitor='val_accuracy',patience=3, restore_best_weights = True)
learning_rate = callbacks.LearningRateScheduler(lr_scheduler)
logdir = datadir + "/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

try:
    history = model.fit(train_generator, epochs=epochs, verbose=1,callbacks=[early_stop, learning_rate, tensorboard_callback],\
                    steps_per_epoch=steps_per_epoch,\
                    validation_data=test_generator,\
                    validation_steps=val_steps)
    %tensorboard --logdir /content/drive/MyDrive/ObjectClassificationRandomSplit/logs
except KeyboardInterrupt:
    pass

#Save model
models_dir = datadir + '/models/'

def savemodel(model,problem):
    filename = os.path.join(models_dir, '%s.h5' %problem)
    model.save(filename)
    print("\nModel saved successfully on file %s\n" %filename)

savemodel(model,'OwnCNN-0.7637')

#Test accuracy and Loss
val_steps=test_generator.n//test_generator.batch_size+1
loss, acc = model.evaluate(test_generator,verbose=1,steps=val_steps)
print('Test loss: %f' %loss)
print('Test accuracy: %f' %acc)

preds = model.predict(test_generator,verbose=1,steps=val_steps)

Ypred = np.argmax(preds, axis=1)
Ytest = test_generator.classes  # shuffle=False in test_generator

#Classification report
print(classification_report(Ytest, Ypred, labels=None, target_names=classnames, digits=3))

#Confusion matrix plot
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=80)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cm = confusion_matrix(Ytest, Ypred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(15*2, 4*2))
plot_confusion_matrix(cm, classes=classnames,
                      title='Confusion matrix')
                      
#Different Layout of confusion Matrix
conf = []
for i in range(0,cm.shape[0]):
  for j in range(0,cm.shape[1]):
    if (i!=j and cm[i][j]>0):
      conf.append([i,j,cm[i][j]])

col=2
conf = np.array(conf)
conf = conf[np.argsort(-conf[:,col])]

print('%-25s     %-25s  \t%s \t%s ' %('True','Predicted','errors','err %'))
print('------------------------------------------------------------------')
for k in conf:
  print('%-25s ->  %-25s  \t%d \t%.2f %% ' %(classnames[k[0]],classnames[k[1]],k[2],k[2]*100.0/test_generator.n))
  
  
#Accuracy history
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#Loss history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Learning rate history
plt.plot(history.history['lr'])
plt.legend('learning rate', loc='upper left')
plt.xlabel('epoch')
plt.ylabel('learning rate')
    
    
