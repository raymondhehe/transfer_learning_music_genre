from keras import backend as K
from keras.models import Model
from keras.layers import GlobalAveragePooling2D as GAP2D
from keras.layers import concatenate as concat
import keras
import kapre

def model_load():
    model = keras.models.load_model('model_best.hdf5', 
                                custom_objects={'Melspectrogram':kapre.time_frequency.Melspectrogram,
                                                'Normalization2D':kapre.utils.Normalization2D})
    feat_layer1 = GAP2D()(model.get_layer('elu_1').output)
    feat_layer2 = GAP2D()(model.get_layer('elu_2').output)
    feat_layer3 = GAP2D()(model.get_layer('elu_3').output)
    feat_layer4 = GAP2D()(model.get_layer('elu_4').output)
    feat_layer5 = GAP2D()(model.get_layer('elu_5').output)
    feat_all = concat([feat_layer1, feat_layer2, feat_layer3, feat_layer4, feat_layer5])
    feat_extractor = Model(inputs=model.input, outputs=feat_all)
    feat_extractor.summary(line_length=90)
    return feat_extractor