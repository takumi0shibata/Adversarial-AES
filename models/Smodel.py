from tensorflow.keras import layers, models
from custom_layers.attention import Attention
from custom_layers.zeromasking import ZeroMaskedEntries
from custom_layers.gradient_reversal import GradientReversalLayer
import tensorflow as tf

class Smodel(models.Model):
    def __init__(self, configs, maxnum, maxlen, vocab_size, embed_table, with_features=False):
        super().__init__()
        self.with_features = with_features
        embedding_dim = configs.EMBEDDING_DIM
        dropout_prob = 0.2
        cnn_filters = configs.CNN_FILTERS
        cnn_kernel_size = configs.CNN_KERNEL_SIZE
        lstm_units = configs.LSTM_UNITS

        #Feature Extractor
        self.feature_extractor_layer0 = layers.Embedding(output_dim=embedding_dim, input_dim=vocab_size, input_length=maxnum*maxlen,
                                                         weights=embed_table, mask_zero=True)
        self.feature_extractor_layer1 = ZeroMaskedEntries(name='x_maskedout')
        self.feature_extractor_layer2 = layers.Dropout(dropout_prob, name='drop_x')
        self.feature_extractor_layer3 = layers.Reshape((maxnum, maxlen, embedding_dim), name='resh_W')
        self.feature_extractor_layer4 = layers.TimeDistributed(layers.Conv1D(cnn_filters, cnn_kernel_size, padding='valid'), name='zcnn')
        self.feature_extractor_layer5 = layers.TimeDistributed(Attention(), name='avg_zcnn')
        self.feature_extractor_layer6 = layers.LSTM(lstm_units, return_sequences=True, name='hz_lstm')
        self.feature_extractor_layer7 = Attention(name='avg_hz_lstm')
        
        #Label Predictor
        self.label_predictor_layer0 = layers.Dense(1, activation='sigmoid')
        self.label_predictor_layer1 = layers.Concatenate(axis=-1)
        
        #Domain Predictor
        self.domain_predictor_layer0 = GradientReversalLayer()
        self.domain_predictor_layer1 = layers.Dense(100, activation='relu')
        self.domain_predictor_layer2 = layers.Dense(2, activation=None)
        
    def call(self, x, x_lin, x_read, train=False, source_train=True, lamda=1.0):
        #Feature Extractor
        x = self.feature_extractor_layer0(x)
        x = self.feature_extractor_layer1(x)
        x = self.feature_extractor_layer2(x, training=train)
        x = self.feature_extractor_layer3(x)
        x = self.feature_extractor_layer4(x)
        x = self.feature_extractor_layer5(x)
        x = self.feature_extractor_layer6(x)
        feature = self.feature_extractor_layer7(x)
        
        
        #Label Predictor
        if source_train is True:
            if self.with_features:
                feature_slice = self.label_predictor_layer1([feature, x_lin, x_read])
            else:
                feature_slice = feature
        else:
            feature_slice = tf.slice(feature, [0, 0], [feature.shape[0] // 2, -1])
            feature_slice = self.label_predictor_layer1([feature_slice, x_lin, x_read])
            
        
        l_score = self.label_predictor_layer0(feature_slice)
        
        #Domain Predictor
        if source_train is True:
            return l_score
        else:
            dp_x = self.domain_predictor_layer0(feature, lamda)    #GradientReversalLayer
            dp_x = self.domain_predictor_layer1(dp_x)
            d_logits = self.domain_predictor_layer2(dp_x)
            
            return l_score, d_logits