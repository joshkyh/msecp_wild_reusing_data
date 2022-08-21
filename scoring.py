import keras
import pandas  as pd
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.preprocessing.sequence import pad_sequences
from keras.utils import CustomObjectScope

import util.researchdb as researchdb
from hnatt import HNATT
from util.runsettings import get_runsettings


def get_color(ww_std):
    if ww_std < 0: return 'None'
    if ww_std < 1: return 'Low'
    if ww_std < 2: return 'Medium'
    if ww_std >= 2: return 'High'


def _load_model(h):


    class Attention(keras.layers.Layer):
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

    import pickle

    with CustomObjectScope({'Attention': Attention}):
        h.model = keras.models.load_model(
            os.path.join(file_paths["expt_model_dir"], file_paths["SAVED_MODEL_FILENAME"]))
        h.rock_attention_weighted_sentences = h.model.get_layer(
            'rock_attention_weighted_sentences').layer
        h.attention_weighted_sentences = h.model.get_layer('attention_weighted_sentences').layer
        h.model.summary()
        tokenizer_path = os.path.join(file_paths["expt_model_dir"], 'rapport_model.h5.tokenizer')
        tokenizer_state = pickle.load(open(tokenizer_path, "rb"))
        h.tokenizer = tokenizer_state['tokenizer']
        h.MAX_SENTENCE_COUNT = tokenizer_state['maxSentenceCount']
        h.MAX_SENTENCE_LENGTH = tokenizer_state['maxSentenceLength']
        h.VOCABULARY_SIZE = tokenizer_state['vocabularySize']
        h._create_reverse_word_index()

    return h


def _load_encoded_text(h, texts):
    encoded_texts = np.zeros((h.MAX_SENTENCE_COUNT, h.MAX_SENTENCE_LENGTH))

    for i, text in enumerate(texts):
        print (i, text)
        int_text = h.tokenizer.texts_to_sequences([text])
        int_text = np.array(int_text[0])
        #print(int_text.shape)
        encoded_text = np.array(pad_sequences([int_text], maxlen=h.MAX_SENTENCE_LENGTH))
        #print(encoded_text.shape)
        encoded_text = np.squeeze(encoded_text, axis=0)

        #print(encoded_texts[i][-len(encoded_text):].shape)
        encoded_texts[i][-len(encoded_text):] = encoded_text
    return encoded_texts


def _get_word_attention_maps(branch, h, encoded_texts):
    assert branch in ('aux', 'pri')

    if branch =='aux':
        hidden_word_encoding_out = Model(inputs=h.rock_attention_weighted_sentences.input,
                                     outputs=h.rock_attention_weighted_sentences.get_layer(
                                    'rock_dense_transform_w').output)
        hidden_word_encodings = hidden_word_encoding_out.predict(encoded_texts)

        word_context= h.rock_attention_weighted_sentences.get_layer('rock_word_attention').get_weights()[0]


    else:
        hidden_word_encoding_out = Model(inputs=h.attention_weighted_sentences.input,
                                         outputs=h.attention_weighted_sentences.get_layer(
                                             'dense_transform_w').output)
        hidden_word_encodings = hidden_word_encoding_out.predict(encoded_texts)

        word_context = h.attention_weighted_sentences.get_layer('word_attention').get_weights()[0]

    u_wattention_aux = encoded_texts * np.exp(
    np.squeeze(np.dot(hidden_word_encodings, word_context)))

    # Loop through each talkturn, get word attention
    # For one talkturn target variable, we have multiple history talkturnS
    # Therefore for each talkturn we have one talkturns_activation_map_aux
    talkturns_activation_map = []

    i = 0
    talkturn_contexts = encoded_texts.shape[0]

    for i in range(talkturn_contexts):
        encoded_text = encoded_texts[i]
        if i < len(texts):
            text = texts[i]
            len(encoded_text)
            len(text.split())

            # generate word, activation pairs
            nopad_encoded_text = encoded_text[-len(text):]
            nopad_encoded_text = nopad_encoded_text[nopad_encoded_text>0]

            reconstructed_text = [h.reverse_word_index[int(word_int)] for word_int in nopad_encoded_text]
            nopad_wattention = u_wattention_aux[i, :]

            nopad_wattention = nopad_wattention / np.expand_dims(np.sum(nopad_wattention, -1), -1)

            # One talkturn_activation_map_aux contain a list of word attention in itself
            talkturn_activation_map = []
            for i, text in enumerate(reconstructed_text):
                word_attn = (text, nopad_wattention[-i])
                talkturn_activation_map.append(word_attn)

            talkturns_activation_map.append(talkturn_activation_map)

    return talkturns_activation_map



def _get_talkturn_attention_map(task_name, h, encoded_texts):

    assert (task_name in ('pri')) or ('aux' in task_name)
    # Load branch specific layers
    hidden_sentence_encoding_out = Model(inputs=h.model.input,
                                         outputs=h.model.get_layer('dense_transform_s').output)
    if task_name == 'pri':
        sentence_context = h.model.get_layer('sentence_attention').get_weights()[0]

    if 'aux' in task_name:
        hidden_sentence_encoding_out = Model(inputs=h.model.input,
                                             outputs=h.model.get_layer('rock_dense_transform_s').output)

        # 16 auxiliary tasks if statements
        # Prosody
        if task_name == 'aux_happiness':
            sentence_context = h.model.get_layer('rock_sentence_attention_hap').get_weights()[0]
        if task_name == 'aux_sad':
            sentence_context = h.model.get_layer('rock_sentence_attention_sad').get_weights()[0]
        if task_name == 'aux_anger':
            sentence_context = h.model.get_layer('rock_sentence_attention_ang').get_weights()[0]
        if task_name == 'aux_fear':
            sentence_context = h.model.get_layer('rock_sentence_attention_fear').get_weights()[0]

        # AU, historical, future
        if 'aux_au' in task_name or 'aux_lead' in task_name or 'aux_lag' in task_name:
            suffix_name = task_name[int(task_name.find('_'))+1:len(task_name)]
            sentence_context = h.model.get_layer('rock_sentence_attention_' + suffix_name).get_weights()[0]




    # Compute attention maps
    # get sentence activations

    i=0
    talkturn_contexts = encoded_texts.shape[0]
    hidden_sentence_encodings = np.squeeze(hidden_sentence_encoding_out.predict(np.expand_dims(encoded_texts, 0)), 0)
    u_sattention = np.exp(np.squeeze(np.dot(hidden_sentence_encodings, sentence_context), -1))

    nopad_sattention = u_sattention[-len(texts):]
    nopad_sattention = nopad_sattention / np.expand_dims(np.sum(nopad_sattention, -1), -1)



    return nopad_sattention

def _get_tasks_attention_map(h, encoded_texts):
    task_context = h.model.get_layer('refined_attention').get_weights()[0]

    hiddent_tasks_encoding_out = Model(inputs=h.model.input,
                                         outputs=h.model.get_layer('pre_primary_concat_reshape').output)


    hidden_tasks_encoding = np.squeeze(hiddent_tasks_encoding_out.predict(np.expand_dims(encoded_texts, 0)), 0)


    hidden_tasks_encoding.shape
    task_context.shape

    u_tattention = np.exp(np.squeeze(np.dot(hidden_tasks_encoding, task_context), -1))

    u_tattention = u_tattention / np.expand_dims(np.sum(u_tattention, -1), -1)

    tasks_attention_map = {}

    tasks_attention_map['hap'] = u_tattention[0]
    tasks_attention_map['sad'] = u_tattention[1]
    tasks_attention_map['ang'] = u_tattention[2]
    tasks_attention_map['fear'] = u_tattention[3]

    tasks_attention_map['au05'] = u_tattention[4]
    tasks_attention_map['au17'] = u_tattention[5]
    tasks_attention_map['au20'] = u_tattention[6]
    tasks_attention_map['au25'] = u_tattention[7]

    tasks_attention_map['lag1'] = u_tattention[8]
    tasks_attention_map['lag2'] = u_tattention[9]
    tasks_attention_map['lag3'] = u_tattention[10]
    tasks_attention_map['lag4'] = u_tattention[11]

    tasks_attention_map['lead1'] = u_tattention[12]
    tasks_attention_map['lead2'] = u_tattention[13]
    tasks_attention_map['lead3'] = u_tattention[14]
    tasks_attention_map['lead4'] = u_tattention[15]

    tasks_attention_map['pri'] = u_tattention[16]


    return tasks_attention_map


def _get_all_attention_maps(h, texts):
    encoded_texts = _load_encoded_text(h, texts)

    # Word Attention Maps
    print('Word Attention Maps')

    word_attention_maps = {}
    word_attention_maps['pri'] = _get_word_attention_maps('pri', h, encoded_texts)
    word_attention_maps['aux'] = _get_word_attention_maps('aux', h, encoded_texts)


    # Talkturn Attention Maps
    print('Talkturn Attention Maps')
    talkturn_attention_maps = {}
    talkturn_attention_maps['pri'] = _get_talkturn_attention_map('pri', h, encoded_texts)


    # Aux-Prosody
    print('Talkturn Attention Maps: Prosody')
    talkturn_attention_maps['aux_happiness'] = _get_talkturn_attention_map('aux_happiness', h, encoded_texts)
    talkturn_attention_maps['aux_sad'] = _get_talkturn_attention_map('aux_sad', h, encoded_texts)
    talkturn_attention_maps['aux_anger'] = _get_talkturn_attention_map('aux_anger', h, encoded_texts)
    talkturn_attention_maps['aux_fear'] = _get_talkturn_attention_map('aux_fear', h, encoded_texts)


    # Aux-AU
    print('Talkturn Attention Maps: AU')
    talkturn_attention_maps['aux_au05'] = _get_talkturn_attention_map('aux_au05', h, encoded_texts)
    talkturn_attention_maps['aux_au17'] = _get_talkturn_attention_map('aux_au17', h, encoded_texts)
    talkturn_attention_maps['aux_au20'] = _get_talkturn_attention_map('aux_au20', h, encoded_texts)
    talkturn_attention_maps['aux_au25'] = _get_talkturn_attention_map('aux_au25', h, encoded_texts)

    # Aux-historical + future
    print('Talkturn Attention Maps: Historical')
    talkturn_attention_maps['aux_lead1'] = _get_talkturn_attention_map('aux_lead1', h, encoded_texts)
    talkturn_attention_maps['aux_lead2'] = _get_talkturn_attention_map('aux_lead2', h, encoded_texts)
    talkturn_attention_maps['aux_lead3'] = _get_talkturn_attention_map('aux_lead3', h, encoded_texts)
    talkturn_attention_maps['aux_lead4'] = _get_talkturn_attention_map('aux_lead4', h, encoded_texts)

    print('Talkturn Attention Maps: Future')
    talkturn_attention_maps['aux_lag1'] = _get_talkturn_attention_map('aux_lag1', h, encoded_texts)
    talkturn_attention_maps['aux_lag2'] = _get_talkturn_attention_map('aux_lag2', h, encoded_texts)
    talkturn_attention_maps['aux_lag3'] = _get_talkturn_attention_map('aux_lag3', h, encoded_texts)
    talkturn_attention_maps['aux_lag4'] = _get_talkturn_attention_map('aux_lag4', h, encoded_texts)


    # Tasks Attention Maps
    print('Tasks Attention Maps')
    tasks_attention_map = _get_tasks_attention_map(h, encoded_texts)

    # Quality Checks
    print('Quality Checks')
    assert len(tasks_attention_map) == 17  # 17 tasks = 16 aux + 1 pri

    # For every talkturn attention maps, it must be as long as the input texts
    for key, val in talkturn_attention_maps.items():
        assert len(texts) == val.shape[0]

    # For every talkturn attention maps, it must be as long as the input texts
    for key, val in word_attention_maps.items():
        assert len(texts) == len(val)

    return word_attention_maps, talkturn_attention_maps, tasks_attention_map

def _get_predict_label(h, texts, label_encoder):
    encoded_texts = _load_encoded_text(h, texts)
    encoded_texts = np.expand_dims(encoded_texts, axis=0)
    dev_preds = h.model.predict(encoded_texts)
    dev_preds_pri = dev_preds[0] # Primary task
    dev_preds_pri = np.argmax(dev_preds_pri, axis=-1)[0]

    predict_label = label_encoder[dev_preds_pri]

    return predict_label


if __name__ == '__main__':

    file_paths, hyperparams = get_runsettings()

    # Specifying the exact model to train
    hyperparams['Batch_Size'] = 43
    hyperparams['Num_GRU'] = 256
    hyperparams['LR'] = 0.0062151287793353
    hyperparams['Sentence_Count'] = 19
    hyperparams['Sentence_Length'] = 120
    hyperparams['ReduceLROnPlateau'] = 0
    hyperparams['GRU_dropout'] = 0.27
    hyperparams['GRU_recurrent_dropout'] = 0.04
    hyperparams['l2_regularization'] = 2.24955630079819E-05
    hyperparams['multilabel_aux'] = 'aphf'
    hyperparams['aux_weights_assignment'] = 'mutualinfo softmax'
    hyperparams['context_k'] = 18
    hyperparams['weights_main'] = 0.84
    hyperparams['weights_prosody_tone_happiness'] = 0.00985108971539227
    hyperparams['weights_prosody_tone_sadness'] = 0.0101372888444364
    hyperparams['weights_prosody_tone_anger'] = 0.0101427945827504
    hyperparams['weights_prosody_tone_fear'] = 0.00978411214176146
    hyperparams['weights_actions_au05'] = 0.00936335690668846
    hyperparams['weights_actions_au17'] = 0.00937870477757095
    hyperparams['weights_actions_au20'] = 0.00943708735899583
    hyperparams['weights_actions_au25'] = 0.00943011639920448
    hyperparams['aux_hierarchy_config'] = 'rock'
    hyperparams['weights_y_lag1'] = 0.0102049359894246
    hyperparams['weights_y_lag2'] = 0.0105853991088323
    hyperparams['weights_y_lag3'] = 0.0101177049068559
    hyperparams['weights_y_lag4'] = 0.0102677900156184
    hyperparams['weights_y_lead1'] = 0.0102221838349905
    hyperparams['weights_y_lead2'] = 0.0106230743204514
    hyperparams['weights_y_lead3'] = 0.0101447784629519
    hyperparams['weights_y_lead4'] = 0.0103095826340747
    hyperparams['Num_GRU_Aux'] = 1

    hyperparams['save_period'] = 1
    hyperparams['Max_Epoch'] = 350

    # STL
    hyperparams['multilabel_aux'] = 'None'
    hyperparams['aux_weights_assignment'] = 'None'
    hyperparams['context_k'] = 18
    hyperparams['weights_main'] = 1.00
    hyperparams['weights_prosody_tone_happiness'] = 0
    hyperparams['weights_prosody_tone_sadness'] = 0
    hyperparams['weights_prosody_tone_anger'] = 0
    hyperparams['weights_prosody_tone_fear'] = 0
    hyperparams['weights_actions_au05'] = 0
    hyperparams['weights_actions_au17'] = 0
    hyperparams['weights_actions_au20'] = 0
    hyperparams['weights_actions_au25'] = 0
    hyperparams['aux_hierarchy_config'] = 'rock'
    hyperparams['weights_y_lag1'] = 0
    hyperparams['weights_y_lag2'] = 0
    hyperparams['weights_y_lag3'] = 0
    hyperparams['weights_y_lag4'] = 0
    hyperparams['weights_y_lead1'] = 0
    hyperparams['weights_y_lead2'] = 0
    hyperparams['weights_y_lead3'] = 0
    hyperparams['weights_y_lead4'] = 0

    # 1605266892570504 MTL
    # 1605436095910794 STL with HAN-ROCK architecture
    hyperparams['Model_Unixtime'] = 1605436095910794
    file_paths['expt_model_dir'] = 'saved_models/expt/' + str(1605436095910794)
    hyperparams['y_config'] = 'c'

    # Load data
    (train_x, train_y), (dev_x, dev_y), (test_x, test_y) = researchdb.load_data(
        hyperparams=hyperparams)

    label_encoder = {}
    label_encoder[0] = 'ang'
    label_encoder[1] = 'hap'
    label_encoder[2] = 'neu'
    label_encoder[3] = 'sad'



    # Load model
    h = HNATT()
    h = _load_model(h)

    # Get Y Prediction
    obs_i = 22
    texts = dev_x[obs_i]

    print(texts)

    actual_label = label_encoder[dev_y[obs_i][0]]
    predict_label = _get_predict_label(h, texts, label_encoder)

    print('actual:', actual_label,  'predict:', predict_label)
    if actual_label==predict_label:
        print('Prediction is Correct')
    else:
        print('Prediction is Wrong')


    # Get Attention Maps
    word_attention_maps, talkturn_attention_maps, tasks_attention_map = _get_all_attention_maps(h, texts)

    # Overall talkturn attention is tasks attention (scaler) * talkturn attention (vector: (talkturns,))
    overall_talkturn_attention = \
        tasks_attention_map['hap'] * talkturn_attention_maps['aux_happiness'] \
        + tasks_attention_map['sad'] * talkturn_attention_maps['aux_sad'] \
        + tasks_attention_map['ang'] * talkturn_attention_maps['aux_anger'] \
        + tasks_attention_map['fear'] * talkturn_attention_maps['aux_fear'] \
        + tasks_attention_map['au05'] * talkturn_attention_maps['aux_au05'] \
        + tasks_attention_map['au17'] * talkturn_attention_maps['aux_au17'] \
        + tasks_attention_map['au20'] * talkturn_attention_maps['aux_au20'] \
        + tasks_attention_map['au25'] * talkturn_attention_maps['aux_au25'] \
        + tasks_attention_map['lead1'] * talkturn_attention_maps['aux_lead1'] \
        + tasks_attention_map['lead2'] * talkturn_attention_maps['aux_lead2'] \
        + tasks_attention_map['lead3'] * talkturn_attention_maps['aux_lead3'] \
        + tasks_attention_map['lead4'] * talkturn_attention_maps['aux_lead4'] \
        + tasks_attention_map['lag1'] * talkturn_attention_maps['aux_lag1'] \
        + tasks_attention_map['lag2'] * talkturn_attention_maps['aux_lag2'] \
        + tasks_attention_map['lag3'] * talkturn_attention_maps['aux_lag3'] \
        + tasks_attention_map['lag4'] * talkturn_attention_maps['aux_lag4'] \
        + tasks_attention_map['pri'] * talkturn_attention_maps['pri']



    word_grid = np.empty(shape=(hyperparams['Sentence_Count'], hyperparams['Sentence_Length']), dtype='object')



    overall_word_attention = np.zeros((hyperparams['Sentence_Count'], hyperparams['Sentence_Length']))

    for tt_idx in range(len(word_attention_maps['pri'])):
        tt_words_pri = pd.DataFrame(word_attention_maps['pri'][tt_idx], columns=['word', 'pri_weight'])
        tt_words_aux = pd.DataFrame(word_attention_maps['aux'][tt_idx], columns=['word', 'aux_weight'])
        tt_words_overall = pd.concat([tt_words_pri, tt_words_aux['aux_weight']], axis=1)
        tt_words_overall['overall_weight'] = tasks_attention_map['pri'] * tt_words_overall['pri_weight'] + \
            (1-tasks_attention_map['pri']) * tt_words_overall['aux_weight']
        for word_idx, row in tt_words_overall.iterrows():
            word_grid[tt_idx, word_idx] = row['word']
            overall_word_attention[tt_idx, word_idx] = row['overall_weight']

    overall_talkturn_attention


    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    overall_talkturn_attention_std =  (overall_talkturn_attention - overall_talkturn_attention.mean())/overall_talkturn_attention.std()
    overall_word_attention_std = (overall_word_attention - overall_word_attention.mean()) / overall_word_attention.std()

    overall_talkturn_attention_std_colors = np.empty(shape=(hyperparams['Sentence_Count']), dtype='object')

    for i in range(len(overall_talkturn_attention_std)):
        val = overall_talkturn_attention_std[i]
        overall_talkturn_attention_std_colors[i] = get_color(val)



    overall_word_attention_std_colors = np.empty(shape=(hyperparams['Sentence_Count'], hyperparams['Sentence_Length']), dtype='object')

    for i in range(overall_word_attention_std.shape[0]):
        for j in range(overall_word_attention_std.shape[1]):
            val = overall_word_attention_std[i,j]
            overall_word_attention_std_colors[i,j] = get_color(val)

    overall_talkturn_attention_std_colors
    overall_word_attention_std_colors
    tasks_attention_map['pri']
