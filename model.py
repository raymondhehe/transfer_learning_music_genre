from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
    
def MLP(x, num_classes, keep_prob=0.75, optimizer=None, summary=True):
    model = Sequential()
    model.add(Dense(x.shape[1], activation='relu', input_shape=(x.shape[1],)))
    model.add(Dropout(keep_prob))
    model.add(Dense(x.shape[1], activation='relu'))
    model.add(Dropout(keep_prob))
    model.add(Dense(num_classes, activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    if summary:
        model.summary()
    return model

def DNN(x, num_classes, keep_prob=0.5, optimizer=None, summary=True):
    model = Sequential()
    model.add(Dense(x.shape[1], activation='relu', input_shape=(x.shape[1],)))
    model.add(Dropout(keep_prob))
    model.add(Dense(x.shape[1], activation='relu'))
    model.add(Dropout(keep_prob))
    model.add(Dense(x.shape[1]//2, activation='relu'))
    model.add(Dropout(keep_prob))
    model.add(Dense(x.shape[1]//4, activation='relu'))
    model.add(Dropout(keep_prob))
    model.add(Dense(num_classes, activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    if summary:
        model.summary()
    return model