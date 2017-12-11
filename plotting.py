import matplotlib.pyplot as plt
import numpy as np

def plot_distribution(labels, plot=False):
    """ plotting the distribution """
    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(unique, counts)) # record the label distribution
    if plot:
        objects = dist.keys()
        y_pos = np.arange(len(objects))
        x = dist.values()
        plt.bar(y_pos, x, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.title('Distribution')
        plt.show()
    return dist

def plot_gridsearch(name, train_acc, test_acc, param):
    """ plotting the grid search result """
    plt.clf()   
    plt.plot(train_acc.keys(), train_acc.values(), '-', label='Training acc')
    plt.plot(test_acc.keys(), test_acc.values(), '--', label='Testing acc')
    plt.title('({}) Training and Testing accuracy'.format(name))
    plt.xlabel(param)
    plt.ylabel('Acc')
    plt.legend()
    plt.show()

def plot_report(name, clf, expected, predicted):
    """ plotting the result from the model """
    """ clf: classifier
        name: name of the classifier
        expected: expected labels
        predicted: predicted labels
    """
    from sklearn import metrics
    print("#### {} model####".format(name))
    print("Testing Accuracy: {:.2%}\n".format(metrics.accuracy_score(expected, predicted)))
    print("Classification report for classifier %s:\n%s\n"
          % (clf, metrics.classification_report(expected, predicted)))
    cm = metrics.confusion_matrix(expected, predicted)
    plot_confusion_matrix(cm, np.unique(predicted))

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """ plotting the confusion matrix """
    """ cm: confusion matrix from sklearn.metrics.confusion_matrix
        classes: list of classes 
    """
    import itertools
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('model.png')

def plot_keras_history(history):
    """ plotting the result from keras model """
    """ history: keras result form model.fit """
    plt.clf()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()