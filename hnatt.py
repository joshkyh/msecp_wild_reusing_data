import pickle

from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import CustomObjectScope
from keras.utils.vis_utils import plot_model
from sklearn.utils import compute_class_weight

from util.evaluate_results import LossHistory
from util.glove import load_glove_embedding
from util.text_util import normalize

TOKENIZER_STATE_PATH = 'saved_models/tokenizer.p'
GLOVE_EMBEDDING_PATH = 'saved_models/glove.6B.300d.txt'


class Attention(Layer):
    def __init__(self, regularizer=None, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.regularizer = regularizer
        self.supports_masking = True

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.context = self.add_weight(name='context',
                                       shape=(input_shape[-1], 1),
                                       initializer=initializers.RandomNormal(
                                           mean=0.0, stddev=0.05, seed=1),
                                       regularizer=self.regularizer,
                                       trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        attention_in = K.exp(K.squeeze(K.dot(x, self.context), axis=-1))
        attention = attention_in / K.expand_dims(K.sum(attention_in, axis=-1), -1)

        if mask is not None:
            # use only the inputs specified by the mask
            # import pdb; pdb.set_trace()
            attention = attention * K.cast(mask, 'float32')

        weighted_sum = K.batch_dot(K.permute_dimensions(x, [0, 2, 1]), attention)
        return weighted_sum

    def compute_output_shape(self, input_shape):
        print(input_shape)
        return (input_shape[0], input_shape[-1])


class HNATT():
    def __init__(self):
        self.model = None
        self.MAX_SENTENCE_LENGTH = 0
        self.MAX_SENTENCE_COUNT = 0
        self.VOCABULARY_SIZE = 0
        self.word_embedding = None
        self.model = None
        self.word_attention_model = None
        self.tokenizer = None
        self.class_count = None
        self.lossweights_list = None

    def _generate_embedding(self, path, dim):
        return load_glove_embedding(path, dim, self.tokenizer.word_index)

    def _build_model_flat_han(self, hyperparams={}, embeddings_path=False):
        l2_reg = regularizers.l2(hyperparams['l2_regularization'])
        # embedding_weights = np.random.normal(0, 1, (len(self.tokenizer.word_index) + 1, embedding_dim))
        # embedding_weights = np.zeros((len(self.tokenizer.word_index) + 1, embedding_dim))
        embedding_dim = hyperparams['WV_Dim']
        np.random.seed(hyperparams['train_seed'])
        embedding_weights = np.random.normal(0, 1, (len(self.tokenizer.word_index) + 1, embedding_dim))
        if embeddings_path:
            embedding_weights = self._generate_embedding(embeddings_path, embedding_dim)

        # Generate word-attention-weighted sentence scores
        sentence_in = Input(shape=(self.MAX_SENTENCE_LENGTH,), dtype='int32')
        embedded_word_seq = Embedding(
            self.VOCABULARY_SIZE,
            embedding_dim,
            weights=[embedding_weights],
            input_length=self.MAX_SENTENCE_LENGTH,
            trainable=True,
            mask_zero=True,
            name='word_embeddings', )(sentence_in)
        word_encoder = Bidirectional(
            GRU(hyperparams['Num_GRU'],
                return_sequences=True,
                dropout=hyperparams['GRU_dropout'],
                recurrent_dropout=hyperparams['GRU_recurrent_dropout'],
                kernel_regularizer=l2_reg))(embedded_word_seq)
        dense_transform_w = Dense(
            hyperparams['Num_GRU'] * 2,
            activation='relu',
            name='dense_transform_w',
            kernel_regularizer=l2_reg)(word_encoder)
        attention_weighted_sentence = Model(
            sentence_in, Attention(name='word_attention', regularizer=l2_reg)(dense_transform_w))
        self.word_attention_model = attention_weighted_sentence
        attention_weighted_sentence.summary()
        if hyperparams['code_development'] == 'y':
            plot_model(attention_weighted_sentence, to_file='./flat_attention_weighted_sentence.png', show_shapes=True, show_layer_names=True)

        # Generate sentence-attention-weighted document scores
        texts_in = Input(shape=(self.MAX_SENTENCE_COUNT, self.MAX_SENTENCE_LENGTH), dtype='int32')
        attention_weighted_sentences = TimeDistributed(attention_weighted_sentence)(texts_in)
        sentence_encoder = Bidirectional(
            GRU(hyperparams['Num_GRU'],
                return_sequences=True,
                dropout=hyperparams['GRU_dropout'],
                recurrent_dropout=hyperparams['GRU_recurrent_dropout'],
                kernel_regularizer=l2_reg))(attention_weighted_sentences)
        dense_transform_s = Dense(
            hyperparams['Num_GRU'] * 2,
            activation='relu',
            name='dense_transform_s',
            kernel_regularizer=l2_reg)(sentence_encoder)
        attention_weighted_text = Attention(name='sentence_attention', regularizer=l2_reg)(dense_transform_s)

        if hyperparams['y_config'] == 'c':
            prediction = Dense(self.class_count, activation='softmax', name='primary')(
                attention_weighted_text)  # change from classification to regression
            loss_list = ['sparse_categorical_crossentropy'] + ['mean_absolute_error'] * 16

        else:
            prediction = Dense(1, activation="linear", name='primary')(attention_weighted_text)
            loss_list = ['mean_absolute_error'] * 17


        # Aux Outputs
        aux_happiness = Dense(1, activation="linear", name='aux_happiness')(attention_weighted_text)
        aux_sadness = Dense(1, activation="linear", name='aux_sadness')(attention_weighted_text)
        aux_anger = Dense(1, activation="linear", name='aux_anger')(attention_weighted_text)
        aux_fear = Dense(1, activation="linear", name='aux_fear')(attention_weighted_text)

        au05 = Dense(1, activation="linear", name='aux_au05')(attention_weighted_text)
        au17 = Dense(1, activation="linear", name='aux_au17')(attention_weighted_text)
        au20 = Dense(1, activation="linear", name='aux_au20')(attention_weighted_text)
        au25 = Dense(1, activation="linear", name='aux_au25')(attention_weighted_text)

        # Lead/lag aux labels can be softmax or regression

        if hyperparams['y_config'] == 'c':
            # class count +1 because i group the other emotions into others
            aux_lag1 = Dense(self.class_count+1, activation='softmax', name='aux_lag1')(attention_weighted_text)
            aux_lag2 = Dense(self.class_count+1, activation='softmax', name='aux_lag2')(attention_weighted_text)
            aux_lag3 = Dense(self.class_count+1, activation='softmax', name='aux_lag3')(attention_weighted_text)
            aux_lag4 = Dense(self.class_count+1, activation='softmax', name='aux_lag4')(attention_weighted_text)

            aux_lead1 = Dense(self.class_count+1, activation='softmax', name='aux_lead1')(attention_weighted_text)
            aux_lead2 = Dense(self.class_count+1, activation='softmax', name='aux_lead2')(attention_weighted_text)
            aux_lead3 = Dense(self.class_count+1, activation='softmax', name='aux_lead3')(attention_weighted_text)
            aux_lead4 = Dense(self.class_count+1, activation='softmax', name='aux_lead4')(attention_weighted_text)
        else:
            aux_lag1 = Dense(1, activation="linear", name='aux_lag1')(attention_weighted_text)
            aux_lag2 = Dense(1, activation="linear", name='aux_lag2')(attention_weighted_text)
            aux_lag3 = Dense(1, activation="linear", name='aux_lag3')(attention_weighted_text)
            aux_lag4 = Dense(1, activation="linear", name='aux_lag4')(attention_weighted_text)

            aux_lead1 = Dense(1, activation="linear", name='aux_lead1')(attention_weighted_text)
            aux_lead2 = Dense(1, activation="linear", name='aux_lead2')(attention_weighted_text)
            aux_lead3 = Dense(1, activation="linear", name='aux_lead3')(attention_weighted_text)
            aux_lead4 = Dense(1, activation="linear", name='aux_lead4')(attention_weighted_text)

        model = Model(inputs=[texts_in],
                      outputs=[prediction,
                               aux_happiness, aux_sadness, aux_anger, aux_fear,
                               au05, au17, au20, au25,
                               aux_lag1, aux_lag2, aux_lag3, aux_lag4,
                               aux_lead1, aux_lead2, aux_lead3, aux_lead4])

        aux_mainweight = hyperparams['weights_main']
        weights_prosody_tone_happiness = hyperparams['weights_prosody_tone_happiness']
        weights_prosody_tone_sadness = hyperparams['weights_prosody_tone_sadness']
        weights_prosody_tone_anger = hyperparams['weights_prosody_tone_anger']
        weights_prosody_tone_fear = hyperparams['weights_prosody_tone_fear']
        weights_actions_au05 = hyperparams['weights_actions_au05']
        weights_actions_au17 = hyperparams['weights_actions_au17']
        weights_actions_au20 = hyperparams['weights_actions_au20']
        weights_actions_au25 = hyperparams['weights_actions_au25']
        weights_y_lag1 = hyperparams['weights_y_lag1']
        weights_y_lag2 = hyperparams['weights_y_lag2']
        weights_y_lag3 = hyperparams['weights_y_lag3']
        weights_y_lag4 = hyperparams['weights_y_lag4']
        weights_y_lead1 = hyperparams['weights_y_lead1']
        weights_y_lead2 = hyperparams['weights_y_lead2']
        weights_y_lead3 = hyperparams['weights_y_lead3']
        weights_y_lead4 = hyperparams['weights_y_lead4']

        lossweights_list = [aux_mainweight,
                            weights_prosody_tone_happiness,
                            weights_prosody_tone_sadness,
                            weights_prosody_tone_anger,
                            weights_prosody_tone_fear,
                            weights_actions_au05,
                            weights_actions_au17,
                            weights_actions_au20,
                            weights_actions_au25,
                            weights_y_lag1,
                            weights_y_lag2,
                            weights_y_lag3,
                            weights_y_lag4,
                            weights_y_lead1,
                            weights_y_lead2,
                            weights_y_lead3,
                            weights_y_lead4
                            ]
        metrics_list = None  # ['mae']

        model.summary()
        if hyperparams['code_development'] == 'y':
            plot_model(model, to_file='./flat_model.png', show_shapes=True, show_layer_names=True)

        if hyperparams['Optimizer'] == 'Adam':
            model.compile(optimizer=Adam(lr=hyperparams['Optimizer_Dict']['Adam']['lr'],
                                         beta_1=hyperparams['Optimizer_Dict']['Adam']['beta_1'],
                                         beta_2=hyperparams['Optimizer_Dict']['Adam']['beta_2'],
                                         clipnorm=1.),
                          loss=loss_list,  # loss='categorical_crossentropy',
                          metrics=metrics_list,
                          loss_weights=lossweights_list)

        if hyperparams['Optimizer'] == 'SGD':
            from keras.optimizers import SGD
            opt = SGD(lr=hyperparams['Optimizer_Dict']['SGD']['lr'],
                      momentum=hyperparams['Optimizer_Dict']['SGD']['momentum'],
                      clipnorm=1.0)
            model.compile(  # optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0),
                optimizer=opt,
                loss=loss_list,  # loss='categorical_crossentropy',
                metrics=metrics_list,
                loss_weights=lossweights_list)

        return model, lossweights_list

    def _build_model_rock_han(self, hyperparams={}, embeddings_path=False):

        l2_reg = regularizers.l2(hyperparams['l2_regularization'])
        # embedding_weights = np.random.normal(0, 1, (len(self.tokenizer.word_index) + 1, embedding_dim))
        # embedding_weights = np.zeros((len(self.tokenizer.word_index) + 1, embedding_dim))
        embedding_dim = hyperparams['WV_Dim']
        np.random.seed(hyperparams['train_seed'])
        embedding_weights = np.random.normal(0, 1, (len(self.tokenizer.word_index) + 1, embedding_dim))
        if embeddings_path:
            embedding_weights = self._generate_embedding(embeddings_path, embedding_dim)

        # Embeddings (
        sentence_in = Input(shape=(self.MAX_SENTENCE_LENGTH,), name='sentence_in', dtype='int32')
        embedded_word_seq = Embedding(
            self.VOCABULARY_SIZE,
            embedding_dim,
            weights=[embedding_weights],
            input_length=self.MAX_SENTENCE_LENGTH,
            trainable=True,
            mask_zero=True,
            name='embedded_word_seq', )(sentence_in) # (,120,300)

        print(embedded_word_seq)

        rock_word_enc = Bidirectional(GRU(hyperparams['Num_GRU_Aux'],return_sequences=True,kernel_regularizer=l2_reg), name='rock_word_enc')(embedded_word_seq)
        print(rock_word_enc)

        #rock_task_word_enc = Bidirectional(GRU(1,return_sequences=False,kernel_regularizer=l2_reg), name='rock_task_word_enc')(rock_word_enc)
        #print(rock_task_word_enc)

        rock_dense_transform_w = Dense(hyperparams['Num_GRU_Aux']*2,activation='relu',name='rock_dense_transform_w',kernel_regularizer=l2_reg)(rock_word_enc)
        print(rock_dense_transform_w)

        rock_attention_weighted_sentence = Model(
            sentence_in, Attention(name='rock_word_attention', regularizer=l2_reg)(rock_dense_transform_w),
            name='rock_attention_weighted_sentence')

        print(rock_attention_weighted_sentence)

        texts_in = Input(shape=(self.MAX_SENTENCE_COUNT, self.MAX_SENTENCE_LENGTH), name='texts_in', dtype='int32')
        rock_attention_weighted_sentences = TimeDistributed(rock_attention_weighted_sentence,name='rock_attention_weighted_sentences')(texts_in)
        rock_sentence_encoder = Bidirectional(
            GRU(hyperparams['Num_GRU_Aux'],
                return_sequences=True,
                dropout=hyperparams['GRU_dropout'],
                recurrent_dropout=hyperparams['GRU_recurrent_dropout'],
                kernel_regularizer=l2_reg), name='rock_sentence_encoder')(rock_attention_weighted_sentences)

        print(rock_sentence_encoder)

        rock_dense_transform_s = Dense(
            hyperparams['Num_GRU'] * 2,
            activation='relu',
            name='rock_dense_transform_s',
            kernel_regularizer=l2_reg)(rock_sentence_encoder)
        print(rock_dense_transform_s)


        rock_attention_weighted_text_hap = Attention(name='rock_sentence_attention_hap', regularizer=l2_reg)(rock_dense_transform_s)
        rock_attention_weighted_text_sad = Attention(name='rock_sentence_attention_sad', regularizer=l2_reg)(rock_dense_transform_s)
        rock_attention_weighted_text_ang = Attention(name='rock_sentence_attention_ang', regularizer=l2_reg)(rock_dense_transform_s)
        rock_attention_weighted_text_fear = Attention(name='rock_sentence_attention_fear', regularizer=l2_reg)(rock_dense_transform_s)
        rock_attention_weighted_text_au05 = Attention(name='rock_sentence_attention_au05', regularizer=l2_reg)(rock_dense_transform_s)
        rock_attention_weighted_text_au17 = Attention(name='rock_sentence_attention_au17', regularizer=l2_reg)(rock_dense_transform_s)
        rock_attention_weighted_text_au20 = Attention(name='rock_sentence_attention_au20', regularizer=l2_reg)(rock_dense_transform_s)
        rock_attention_weighted_text_au25 = Attention(name='rock_sentence_attention_au25', regularizer=l2_reg)(rock_dense_transform_s)

        rock_attention_weighted_text_lag1 = Attention(name='rock_sentence_attention_lag1', regularizer=l2_reg)(rock_dense_transform_s)
        rock_attention_weighted_text_lag2 = Attention(name='rock_sentence_attention_lag2', regularizer=l2_reg)(rock_dense_transform_s)
        rock_attention_weighted_text_lag3 = Attention(name='rock_sentence_attention_lag3', regularizer=l2_reg)(rock_dense_transform_s)
        rock_attention_weighted_text_lag4 = Attention(name='rock_sentence_attention_lag4', regularizer=l2_reg)(rock_dense_transform_s)

        rock_attention_weighted_text_lead1 = Attention(name='rock_sentence_attention_lead1', regularizer=l2_reg)(rock_dense_transform_s)
        rock_attention_weighted_text_lead2 = Attention(name='rock_sentence_attention_lead2', regularizer=l2_reg)(rock_dense_transform_s)
        rock_attention_weighted_text_lead3 = Attention(name='rock_sentence_attention_lead3', regularizer=l2_reg)(rock_dense_transform_s)
        rock_attention_weighted_text_lead4 = Attention(name='rock_sentence_attention_lead4', regularizer=l2_reg)(rock_dense_transform_s)





        # Aux Outputs
        aux_happiness = Dense(1, activation="linear", name='aux_happiness')(rock_attention_weighted_text_hap)
        aux_sadness = Dense(1, activation="linear", name='aux_sadness')(rock_attention_weighted_text_sad)
        aux_anger = Dense(1, activation="linear", name='aux_anger')(rock_attention_weighted_text_ang)
        aux_fear = Dense(1, activation="linear", name='aux_fear')(rock_attention_weighted_text_fear)

        au05 = Dense(1, activation="linear", name='aux_au05')(rock_attention_weighted_text_au05)
        au17 = Dense(1, activation="linear", name='aux_au17')(rock_attention_weighted_text_au17)
        au20 = Dense(1, activation="linear", name='aux_au20')(rock_attention_weighted_text_au20)
        au25 = Dense(1, activation="linear", name='aux_au25')(rock_attention_weighted_text_au25)

        # Lead/lag aux labels can be softmax or regression

        if hyperparams['y_config'] == 'c':
            # class count +1 because i group the other emotions into others
            aux_lag1 = Dense(self.class_count+1, activation='softmax', name='aux_lag1')(rock_attention_weighted_text_lag1)
            aux_lag2 = Dense(self.class_count+1, activation='softmax', name='aux_lag2')(rock_attention_weighted_text_lag2)
            aux_lag3 = Dense(self.class_count+1, activation='softmax', name='aux_lag3')(rock_attention_weighted_text_lag3)
            aux_lag4 = Dense(self.class_count+1, activation='softmax', name='aux_lag4')(rock_attention_weighted_text_lag4)

            aux_lead1 = Dense(self.class_count+1, activation='softmax', name='aux_lead1')(rock_attention_weighted_text_lead1)
            aux_lead2 = Dense(self.class_count+1, activation='softmax', name='aux_lead2')(rock_attention_weighted_text_lead2)
            aux_lead3 = Dense(self.class_count+1, activation='softmax', name='aux_lead3')(rock_attention_weighted_text_lead3)
            aux_lead4 = Dense(self.class_count+1, activation='softmax', name='aux_lead4')(rock_attention_weighted_text_lead4)
        else:
            aux_lag1 = Dense(1, activation="linear", name='aux_lag1')(rock_attention_weighted_text_lag1)
            aux_lag2 = Dense(1, activation="linear", name='aux_lag2')(rock_attention_weighted_text_lag2)
            aux_lag3 = Dense(1, activation="linear", name='aux_lag3')(rock_attention_weighted_text_lag3)
            aux_lag4 = Dense(1, activation="linear", name='aux_lag4')(rock_attention_weighted_text_lag4)

            aux_lead1 = Dense(1, activation="linear", name='aux_lead1')(rock_attention_weighted_text_lead1)
            aux_lead2 = Dense(1, activation="linear", name='aux_lead2')(rock_attention_weighted_text_lead2)
            aux_lead3 = Dense(1, activation="linear", name='aux_lead3')(rock_attention_weighted_text_lead3)
            aux_lead4 = Dense(1, activation="linear", name='aux_lead4')(rock_attention_weighted_text_lead4)

        word_encoder = Bidirectional(
            GRU(hyperparams['Num_GRU'],
                return_sequences=True,
                dropout=hyperparams['GRU_dropout'],
                recurrent_dropout=hyperparams['GRU_recurrent_dropout'],
                kernel_regularizer=l2_reg), name = 'word_encoder')(embedded_word_seq)

        dense_transform_w = Dense(
            hyperparams['Num_GRU'] * 2,
            activation='relu',
            name='dense_transform_w',
            kernel_regularizer=l2_reg)(word_encoder)

        attention_weighted_sentence = Model(
            sentence_in, Attention(name='word_attention', regularizer=l2_reg)(dense_transform_w),
            name='attention_weighted_sentence')

        self.word_attention_model = attention_weighted_sentence
        attention_weighted_sentence.summary()
        if hyperparams['code_development'] == 'y':
            plot_model(attention_weighted_sentence, to_file='./rock_attention_weighted_sentence.png', show_shapes=True, show_layer_names=True)

        # Generate sentence-attention-weighted document scores

        attention_weighted_sentences = TimeDistributed(attention_weighted_sentence, name='attention_weighted_sentences')(texts_in)
        sentence_encoder = Bidirectional(
            GRU(hyperparams['Num_GRU'],
                return_sequences=True,
                dropout=hyperparams['GRU_dropout'],
                recurrent_dropout=hyperparams['GRU_recurrent_dropout'],
                kernel_regularizer=l2_reg),name = 'sentence_encoder')(attention_weighted_sentences)
        dense_transform_s = Dense(
            hyperparams['Num_GRU'] * 2,
            activation='relu',
            name='dense_transform_s',
            kernel_regularizer=l2_reg)(sentence_encoder)
        attention_weighted_text = Attention(name='sentence_attention', regularizer=l2_reg)(dense_transform_s)

        # Information Fusion Module
        if hyperparams['num_translator'] == 'same as prediction':
            translator_count = 1
            if hyperparams['y_config'] == 'c':
                translator_count = self.class_count+1


        if hyperparams['num_translator'] != 'None':
            tanh_rock_attention_weighted_text_hap = Dense(translator_count, activation='tanh',
                                                          name='tanh_rock_attention_weighted_text_hap',
                                                          kernel_initializer=initializers.Ones(),
                                                          bias_initializer=initializers.Zeros(),
                                                          trainable=True
                                                          ) (rock_attention_weighted_text_hap)

            tanh_rock_attention_weighted_text_sad = Dense(translator_count, activation='tanh',
                                                          name='tanh_rock_attention_weighted_text_sad',
                                                          kernel_initializer=initializers.Ones(),
                                                          bias_initializer=initializers.Zeros(),
                                                          trainable=True
                                                          )(rock_attention_weighted_text_sad)

            tanh_rock_attention_weighted_text_ang = Dense(translator_count, activation='tanh',
                                                          name='tanh_rock_attention_weighted_text_ang',
                                                          kernel_initializer=initializers.Ones(),
                                                          bias_initializer=initializers.Zeros(),
                                                          trainable=True
                                                          )(rock_attention_weighted_text_ang)

            tanh_rock_attention_weighted_text_fear = Dense(translator_count, activation='tanh',
                                                          name='tanh_rock_attention_weighted_text_fear',
                                                           kernel_initializer=initializers.Ones(),
                                                           bias_initializer=initializers.Zeros(),
                                                           trainable=True
                                                          )(rock_attention_weighted_text_fear)

            tanh_rock_attention_weighted_text_au05 = Dense(translator_count, activation='tanh',
                                                          name='tanh_rock_attention_weighted_text_au05',
                                                           kernel_initializer=initializers.Ones(),
                                                           bias_initializer=initializers.Zeros(),
                                                           trainable=True
                                                          )(rock_attention_weighted_text_au05)

            tanh_rock_attention_weighted_text_au17 = Dense(translator_count, activation='tanh',
                                                          name='tanh_rock_attention_weighted_text_au17',
                                                           kernel_initializer=initializers.Ones(),
                                                           bias_initializer=initializers.Zeros(),
                                                           trainable=True
                                                          )(rock_attention_weighted_text_au17)

            tanh_rock_attention_weighted_text_au20 = Dense(translator_count, activation='tanh',
                                                          name='tanh_rock_attention_weighted_text_au20',
                                                           kernel_initializer=initializers.Ones(),
                                                           bias_initializer=initializers.Zeros(),
                                                           trainable=True
                                                          )(rock_attention_weighted_text_au20)

            tanh_rock_attention_weighted_text_au25 = Dense(translator_count, activation='tanh',
                                                          name='tanh_rock_attention_weighted_text_au25',
                                                           kernel_initializer=initializers.Ones(),
                                                           bias_initializer=initializers.Zeros(),
                                                           trainable=True
                                                          )(rock_attention_weighted_text_au25)

            tanh_rock_attention_weighted_text_lag1 = Dense(translator_count, activation='tanh',
                                                          name='tanh_rock_attention_weighted_text_lag1',
                                                           kernel_initializer=initializers.Ones(),
                                                           bias_initializer=initializers.Zeros(),
                                                           trainable=True
                                                          )(rock_attention_weighted_text_lag1)

            tanh_rock_attention_weighted_text_lag2 = Dense(translator_count, activation='tanh',
                                                          name='tanh_rock_attention_weighted_text_lag2',
                                                           kernel_initializer=initializers.Ones(),
                                                           bias_initializer=initializers.Zeros(),
                                                           trainable=True
                                                          )(rock_attention_weighted_text_lag2)

            tanh_rock_attention_weighted_text_lag3 = Dense(translator_count, activation='tanh',
                                                          name='tanh_rock_attention_weighted_text_lag3',
                                                           kernel_initializer=initializers.Ones(),
                                                           bias_initializer=initializers.Zeros(),
                                                           trainable=True
                                                          )(rock_attention_weighted_text_lag3)

            tanh_rock_attention_weighted_text_lag4 = Dense(translator_count, activation='tanh',
                                                          name='tanh_rock_attention_weighted_text_lag4',
                                                           kernel_initializer=initializers.Ones(),
                                                           bias_initializer=initializers.Zeros(),
                                                           trainable=True
                                                          )(rock_attention_weighted_text_lag4)

            tanh_rock_attention_weighted_text_lead1 = Dense(translator_count, activation='tanh',
                                                          name='tanh_rock_attention_weighted_text_lead1',
                                                            kernel_initializer=initializers.Ones(),
                                                            bias_initializer=initializers.Zeros(),
                                                            trainable=True
                                                          )(rock_attention_weighted_text_lead1)

            tanh_rock_attention_weighted_text_lead2 = Dense(translator_count, activation='tanh',
                                                            name='tanh_rock_attention_weighted_text_lead2',
                                                            kernel_initializer=initializers.Ones(),
                                                            bias_initializer=initializers.Zeros(),
                                                            trainable=True
                                                            )(rock_attention_weighted_text_lead2)

            tanh_rock_attention_weighted_text_lead3 = Dense(translator_count, activation='tanh',
                                                            name='tanh_rock_attention_weighted_text_lead3',
                                                            kernel_initializer=initializers.Ones(),
                                                            bias_initializer=initializers.Zeros(),
                                                            trainable=True
                                                            )(rock_attention_weighted_text_lead3)

            tanh_rock_attention_weighted_text_lead4 = Dense(translator_count, activation='tanh',
                                                            name='tanh_rock_attention_weighted_text_lead4',
                                                            kernel_initializer=initializers.Ones(),
                                                            bias_initializer=initializers.Zeros(),
                                                            trainable=True
                                                            )(rock_attention_weighted_text_lead4)

            tanh_attention_weighted_text = Dense(translator_count, activation='tanh',
                                                            name='tanh_attention_weighted_text',
                                                 kernel_initializer=initializers.Ones(),
                                                 bias_initializer=initializers.Zeros(),
                                                 trainable=True
                                                            )(attention_weighted_text)


            pre_primary_concat = Concatenate(axis=-1, name='pre_primary_concat')([tanh_rock_attention_weighted_text_hap,
                                                                                   tanh_rock_attention_weighted_text_sad,
                                                                                   tanh_rock_attention_weighted_text_ang,
                                                                                   tanh_rock_attention_weighted_text_fear,
                                                                                   tanh_rock_attention_weighted_text_au05,
                                                                                   tanh_rock_attention_weighted_text_au17,
                                                                                   tanh_rock_attention_weighted_text_au20,
                                                                                   tanh_rock_attention_weighted_text_au25,
                                                                                   tanh_rock_attention_weighted_text_lag1,
                                                                                  tanh_rock_attention_weighted_text_lag2,
                                                                                  tanh_rock_attention_weighted_text_lag3,
                                                                                  tanh_rock_attention_weighted_text_lag4,
                                                                                  tanh_rock_attention_weighted_text_lead1,
                                                                                  tanh_rock_attention_weighted_text_lead2,
                                                                                  tanh_rock_attention_weighted_text_lead3,
                                                                                  tanh_rock_attention_weighted_text_lead4,
                                                                                  tanh_attention_weighted_text]) # 16 aux + 1 main
            pre_primary_concat_reshape = Reshape((17, translator_count), name='pre_primary_concat_reshape')(
                pre_primary_concat)


        elif hyperparams['num_translator'] == 'None':
            pre_primary_concat = Concatenate(axis=-1, name='pre_primary_concat')([rock_attention_weighted_text_hap,
                                                                                  rock_attention_weighted_text_sad,
                                                                                  rock_attention_weighted_text_ang,
                                                                                  rock_attention_weighted_text_fear,
                                                                                  rock_attention_weighted_text_au05,
                                                                                  rock_attention_weighted_text_au17,
                                                                                  rock_attention_weighted_text_au20,
                                                                                  rock_attention_weighted_text_au25,
                                                                                  rock_attention_weighted_text_lag1,
                                                                                  rock_attention_weighted_text_lag2,
                                                                                  rock_attention_weighted_text_lag3,
                                                                                  rock_attention_weighted_text_lag4,
                                                                                  rock_attention_weighted_text_lead1,
                                                                                  rock_attention_weighted_text_lead2,
                                                                                  rock_attention_weighted_text_lead3,
                                                                                  rock_attention_weighted_text_lead4,
                                                                                  attention_weighted_text])  # 16 aux + 1 main
            pre_primary_concat_reshape = Reshape((17, hyperparams['Num_GRU'] * 2), name='pre_primary_concat_reshape')(
                pre_primary_concat)





        refined_attention = Attention(name='refined_attention', regularizer=l2_reg)(pre_primary_concat_reshape)

        if hyperparams['y_config'] == 'c':
            prediction = Dense(self.class_count, activation='softmax', name='primary')(refined_attention)  # change from classification to regression
            loss_list = ['sparse_categorical_crossentropy'] + ['mean_absolute_error'] * 8 + ['sparse_categorical_crossentropy']*8

        else:
            prediction = Dense(1, activation="linear", name='primary')(refined_attention)
            loss_list = ['mean_absolute_error'] * 17



        model = Model(inputs=[texts_in],
                      outputs=[prediction,
                               aux_happiness, aux_sadness, aux_anger, aux_fear,
                               au05, au17, au20, au25,
                               aux_lag1, aux_lag2, aux_lag3, aux_lag4,
                               aux_lead1, aux_lead2, aux_lead3, aux_lead4])

        # static and dynamic loss weights both use K.variable
        # the difference is in the callback -- whether the callback update these K.variable
        # K.variable don't allow me to save the model!
        aux_mainweight = hyperparams['weights_main']
        weights_prosody_tone_happiness = hyperparams['weights_prosody_tone_happiness']
        weights_prosody_tone_sadness = hyperparams['weights_prosody_tone_sadness']
        weights_prosody_tone_anger = hyperparams['weights_prosody_tone_anger']
        weights_prosody_tone_fear = hyperparams['weights_prosody_tone_fear']
        weights_actions_au05 = hyperparams['weights_actions_au05']
        weights_actions_au17 = hyperparams['weights_actions_au17']
        weights_actions_au20 = hyperparams['weights_actions_au20']
        weights_actions_au25 = hyperparams['weights_actions_au25']
        weights_y_lag1 = hyperparams['weights_y_lag1']
        weights_y_lag2 = hyperparams['weights_y_lag2']
        weights_y_lag3 = hyperparams['weights_y_lag3']
        weights_y_lag4 = hyperparams['weights_y_lag4']
        weights_y_lead1 = hyperparams['weights_y_lead1']
        weights_y_lead2 = hyperparams['weights_y_lead2']
        weights_y_lead3 = hyperparams['weights_y_lead3']
        weights_y_lead4 = hyperparams['weights_y_lead4']

        lossweights_list = [aux_mainweight,
                            weights_prosody_tone_happiness,
                            weights_prosody_tone_sadness,
                            weights_prosody_tone_anger,
                            weights_prosody_tone_fear,
                            weights_actions_au05,
                            weights_actions_au17,
                            weights_actions_au20,
                            weights_actions_au25,
                            weights_y_lag1,
                            weights_y_lag2,
                            weights_y_lag3,
                            weights_y_lag4,
                            weights_y_lead1,
                            weights_y_lead2,
                            weights_y_lead3,
                            weights_y_lead4
                            ]

        metrics_list = None  # ['mae']

        model.summary()
        if hyperparams['code_development'] == 'y':
            plot_model(model, to_file='./rock_model.png', show_shapes=True, show_layer_names=True)

        if hyperparams['Optimizer'] == 'Adam':
            model.compile(optimizer=Adam(lr=hyperparams['Optimizer_Dict']['Adam']['lr'],
                                         beta_1=hyperparams['Optimizer_Dict']['Adam']['beta_1'],
                                         beta_2=hyperparams['Optimizer_Dict']['Adam']['beta_2'],
                                         clipnorm=1.),
                          loss=loss_list,  # loss='categorical_crossentropy',
                          metrics=metrics_list,
                          loss_weights=lossweights_list)

        if hyperparams['Optimizer'] == 'SGD':
            from keras.optimizers import SGD
            opt = SGD(lr=hyperparams['Optimizer_Dict']['SGD']['lr'],
                      momentum=hyperparams['Optimizer_Dict']['SGD']['momentum'],
                      clipnorm=1.0)
            model.compile(  # optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0),
                optimizer=opt,
                loss=loss_list,  # loss='categorical_crossentropy',
                metrics=metrics_list,
                loss_weights=lossweights_list)

        return model, lossweights_list

    def _build_model(self, hyperparams={}, embeddings_path=False):
        if hyperparams['y_config'] == 'c':
            if hyperparams['dataset_config'] == 'i':
                self.class_count = 4  # 4 emotions labels - anger. sad. neutral. happy.
                
        if hyperparams['aux_hierarchy_config'] == 'flat':
            return self._build_model_flat_han(hyperparams, embeddings_path)
        if hyperparams['aux_hierarchy_config'] == 'rock':
            return self._build_model_rock_han(hyperparams, embeddings_path)



    def load_weights(self, saved_model_dir, saved_model_filename):
        with CustomObjectScope({'Attention': Attention}):
            self.model = load_model(os.path.join(saved_model_dir, saved_model_filename))
            self.word_attention_model = self.model.get_layer('rock_word_enc').layer
            tokenizer_path = os.path.join(
                saved_model_dir, self._get_tokenizer_filename(saved_model_filename))
            tokenizer_state = pickle.load(open(tokenizer_path, "rb"))
            self.tokenizer = tokenizer_state['tokenizer']
            self.MAX_SENTENCE_COUNT = tokenizer_state['maxSentenceCount']
            self.MAX_SENTENCE_LENGTH = tokenizer_state['maxSentenceLength']
            self.VOCABULARY_SIZE = tokenizer_state['vocabularySize']
            self._create_reverse_word_index()

    def _get_tokenizer_filename(self, saved_model_filename):
        return saved_model_filename + '.tokenizer'

    def _fit_on_texts(self, texts, limit_sentence_count, limit_sentence_length):
        self.tokenizer = Tokenizer(filters='"()*,-/;[\]^_`{|}~', oov_token='UNK');
        all_sentences = []
        max_sentence_count = 0
        max_sentence_length = 0
        for text in texts:
            sentence_count = len(text)
            if sentence_count > max_sentence_count:
                max_sentence_count = sentence_count
            for sentence in text:
                sentence_length = len(sentence)
                if sentence_length > max_sentence_length:
                    max_sentence_length = sentence_length
                all_sentences.append(sentence)

        self.MAX_SENTENCE_COUNT = min(max_sentence_count, limit_sentence_count)
        self.MAX_SENTENCE_LENGTH = min(max_sentence_length, limit_sentence_length)
        self.tokenizer.fit_on_texts(all_sentences)
        self.VOCABULARY_SIZE = len(self.tokenizer.word_index) + 1
        self._create_reverse_word_index()

    def _create_reverse_word_index(self):
        self.reverse_word_index = {value: key for key, value in self.tokenizer.word_index.items()}

    def _encode_texts(self, texts):
        encoded_texts = np.zeros((len(texts), self.MAX_SENTENCE_COUNT, self.MAX_SENTENCE_LENGTH))
        for i, text in enumerate(texts):
            encoded_text = np.array(pad_sequences(
                self.tokenizer.texts_to_sequences(text),
                maxlen=self.MAX_SENTENCE_LENGTH))[:self.MAX_SENTENCE_COUNT]
            encoded_texts[i][-len(encoded_text):] = encoded_text
        return encoded_texts

    def _save_tokenizer_on_epoch_end(self, path, epoch):
        if epoch == 0:
            tokenizer_state = {
                'tokenizer': self.tokenizer,
                'maxSentenceCount': self.MAX_SENTENCE_COUNT,
                'maxSentenceLength': self.MAX_SENTENCE_LENGTH,
                'vocabularySize': self.VOCABULARY_SIZE
            }
            pickle.dump(tokenizer_state, open(path, "wb"))

    def train(self, train_x, train_y, dev_x, dev_y,test_x, test_y, file_paths, hyperparams):

        # Extract hyperparams
        batch_size = hyperparams['Batch_Size']
        epochs = hyperparams['Max_Epoch']
        embedding_dim = hyperparams['WV_Dim']
        embeddings_path = file_paths["EMBEDDINGS_PATH"]
        limit_sentence_count = hyperparams['Sentence_Count']
        limit_sentence_length = hyperparams['Sentence_Length']
        saved_model_dir = file_paths["expt_model_dir"]
        saved_model_filename = file_paths["SAVED_MODEL_FILENAME"]

        # fit tokenizer
        self._fit_on_texts(np.hstack((train_x, dev_x,test_x)), limit_sentence_count, limit_sentence_length)
        self.model, self.lossweights_list = self._build_model(
            hyperparams=hyperparams,
            embeddings_path=embeddings_path)
        encoded_train_x = self._encode_texts(train_x)
        encoded_dev_x = self._encode_texts(dev_x)
        encoded_test_x = self._encode_texts(test_x)


        # callbacks
        callbacks = [
            LossHistory(hyperparams=hyperparams,
                        encoded_train_x=encoded_train_x,
                        train_y=train_y[:, 0],  # only pass in the primary task
                        encoded_test_x=encoded_test_x,
                        test_y=test_y[:,0], # only pass in the primary task
                        aux_mainweight=self.lossweights_list[0],
                        weights_prosody_tone_happiness=self.lossweights_list[1],
                        weights_prosody_tone_sadness=self.lossweights_list[2],
                        weights_prosody_tone_anger=self.lossweights_list[3],
                        weights_prosody_tone_fear=self.lossweights_list[4],
                        weights_actions_au05=self.lossweights_list[5],
                        weights_actions_au17 = self.lossweights_list[6],
                        weights_actions_au20 = self.lossweights_list[7],
                        weights_actions_au25 = self.lossweights_list[8],
                        weights_y_lag1 = self.lossweights_list[9],
                        weights_y_lag2 = self.lossweights_list[10],
                        weights_y_lag3 = self.lossweights_list[11],
                        weights_y_lag4 = self.lossweights_list[12],
                        weights_y_lead1 = self.lossweights_list[13],
                        weights_y_lead2 = self.lossweights_list[14],
                        weights_y_lead3 = self.lossweights_list[15],
                        weights_y_lead4 = self.lossweights_list[16]
                        ),
            EarlyStopping(
                monitor='val_loss',
                patience=hyperparams['Patience'],
            ),

            # keras.callbacks.TensorBoard(
            #    log_dir="logs/expt/"+ \
            #    str(hyperparams['Model_Unixtime']),
            #     histogram_freq=1,
            #     write_graph=True,
            #     write_images=True
            # ),
            LambdaCallback(
                on_epoch_end=lambda epoch, logs: self._save_tokenizer_on_epoch_end(
                    os.path.join(saved_model_dir,
                                 self._get_tokenizer_filename(saved_model_filename)), epoch))
        ]

        if hyperparams['ReduceLROnPlateau'] == 1:
            callbacks.append(ReduceLROnPlateau())

        if saved_model_filename:
            filepath = "{epoch:02d}.h5"
            callbacks.append(
                ModelCheckpoint(
                    filepath=os.path.join(saved_model_dir, filepath),
                    # monitor='val_acc',
                    period=hyperparams['save_period'],
                    save_best_only=False,
                    save_weights_only=False,
                )
            )

        # class weights
        class_weight = None
        if hyperparams['y_config']== 'c':
            class_weight = compute_class_weight('balanced', np.unique(train_y[0]), train_y[0])
            class_weight = [class_weight,
                            None, None, None, None,
                            None, None, None, None,
                            None, None, None, None,
                            None, None, None, None]
        # fit
        if hyperparams['multilabel_aux'] in ['None', 'ap', 'aph','aphf']:
            self.model.fit(x=encoded_train_x,
                           y=[train_y[:,0],
                              train_y[:, 1],
                              train_y[:, 2],
                              train_y[:, 3],
                              train_y[:, 4],
                              train_y[:, 5],
                              train_y[:, 6],
                              train_y[:, 7],
                              train_y[:, 8],
                              train_y[:, 9],
                              train_y[:, 10],
                              train_y[:, 11],
                              train_y[:, 12],
                              train_y[:, 13],
                              train_y[:, 14],
                              train_y[:, 15],
                              train_y[:, 16]
                              ],
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=2,
                           callbacks=callbacks,
                           validation_data=(
                               encoded_dev_x,
                           [dev_y[:, 0],
                           dev_y[:, 1],
                           dev_y[:, 2],
                           dev_y[:, 3],
                           dev_y[:, 4],
                            dev_y[:, 5],
                            dev_y[:, 6],
                            dev_y[:, 7],
                            dev_y[:, 8],
                            dev_y[:, 9],
                            dev_y[:, 10],
                            dev_y[:, 11],
                            dev_y[:, 12],
                            dev_y[:, 13],
                            dev_y[:, 14],
                            dev_y[:, 15],
                            dev_y[:, 16]]),
                           class_weight=class_weight
                           )

    def _encode_input(self, x, log=False):
        x = np.array(x)
        if not x.shape:
            x = np.expand_dims(x, 0)
        texts = np.array([normalize(text) for text in x])
        return self._encode_texts(texts)

    def predict(self, x):
        encoded_x = self._encode_texts(x)
        return self.model.predict(encoded_x)

    def activation_maps(self, text, websafe=False):
        normalized_text = normalize(text)
        encoded_text = self._encode_input(text)[0]

        # get word activations
        hidden_word_encoding_out = Model(inputs=self.word_attention_model.input,
                                         outputs=self.word_attention_model.get_layer('dense_transform_w').output)
        hidden_word_encodings = hidden_word_encoding_out.predict(encoded_text)
        word_context = self.word_attention_model.get_layer('word_attention').get_weights()[0]
        u_wattention = encoded_text * np.exp(np.squeeze(np.dot(hidden_word_encodings, word_context)))
        if websafe:
            u_wattention = u_wattention.astype(float)

        # generate word, activation pairs
        nopad_encoded_text = encoded_text[-len(normalized_text):]
        nopad_encoded_text = [list(filter(lambda x: x > 0, sentence)) for sentence in nopad_encoded_text]
        reconstructed_texts = [[self.reverse_word_index[int(i)]
                                for i in sentence] for sentence in nopad_encoded_text]
        nopad_wattention = u_wattention[-len(normalized_text):]
        nopad_wattention = nopad_wattention / np.expand_dims(np.sum(nopad_wattention, -1), -1)
        nopad_wattention = np.array([attention_seq[-len(sentence):]
                                     for attention_seq, sentence in zip(nopad_wattention, nopad_encoded_text)])
        word_activation_maps = []
        for i, text in enumerate(reconstructed_texts):
            word_activation_maps.append(list(zip(text, nopad_wattention[i])))

        # get sentence activations
        hidden_sentence_encoding_out = Model(inputs=self.model.input,
                                             outputs=self.model.get_layer('dense_transform_s').output)
        hidden_sentence_encodings = np.squeeze(
            hidden_sentence_encoding_out.predict(np.expand_dims(encoded_text, 0)), 0)
        sentence_context = self.model.get_layer('sentence_attention').get_weights()[0]
        u_sattention = np.exp(np.squeeze(np.dot(hidden_sentence_encodings, sentence_context), -1))
        if websafe:
            u_sattention = u_sattention.astype(float)
        nopad_sattention = u_sattention[-len(normalized_text):]

        nopad_sattention = nopad_sattention / np.expand_dims(np.sum(nopad_sattention, -1), -1)

        activation_map = list(zip(word_activation_maps, nopad_sattention))

        return activation_map

if __name__ == '__main__':
    import util.runsettings as RS
    import util.researchdb as rdb
    hyperparams = RS.random_runsettings(input_config='vpa',
                                        dataset_config='s',
                                        y_config='a',
                                        num_gru_config = 25,
                                        num_translator = 'None',
                                        dynamic_loss_weights = 'n',
                                        multilabel_aux='ap',
                                        aux_weights_assignment='random',
                                        aux_hierarchy_config='flat',
                                        code_development='y')
    RS.write_runsettings(hyperparams)

    # Get hyperparams
    file_paths, hyperparams = RS.get_runsettings()

    (train_x, train_y), (dev_x, dev_y), (test_x, test_y) = rdb.load_data(hyperparams=hyperparams)

    train_y[0]
    h = HNATT()
    h.train(train_x, train_y, dev_x, dev_y,test_x, test_y, file_paths, hyperparams)


    train_y_list =[train_y[:,0],
                              train_y[:, 1],
                              train_y[:, 2],
                              train_y[:, 3],
                              train_y[:, 4],
                              train_y[:, 5],
                              train_y[:, 6],
                              train_y[:, 7],
                              train_y[:, 8],
                              train_y[:, 9],
                              train_y[:, 10],
                              train_y[:, 11],
                              train_y[:, 12],
                              train_y[:, 13],
                              train_y[:, 14],
                              train_y[:, 15],
                              train_y[:, 16]
                              ]

    len(train_y_list)
    train_y_list[0]
    train_y_list[16]

