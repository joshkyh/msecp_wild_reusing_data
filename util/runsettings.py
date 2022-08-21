import os as os
import pickle
import random
import time

import numpy as np
import pandas as pd

import util.config as config


def get_fullrun_description():
    # Initialize the candidate result
    passed_unique_test = False
    fullrun_description = int(time.time() * 1000000)

    # While loop until we confirm that candidate is unique
    while (passed_unique_test == False):

        cnxn = config.get_connection()
        sql = f'SELECT COUNT(1) from iemocap.Model_Hyperparameter_Dict WHERE Model_Unixtime = {fullrun_description}'
        count = pd.read_sql(sql=sql, con=cnxn)
        cnxn.close()
        if count.iloc[0][0]==0:
            passed_unique_test = True
        else:
            fullrun_description+=1

    return fullrun_description


def get_mutualinfo_weights(hyperparams):
    dataset_config = hyperparams['dataset_config']
    y_config = hyperparams['y_config']
    multilabel_aux = hyperparams['multilabel_aux']


    # Pull the Y label and Aux Labels
    print('Looking Up Mutual Information Weights')
    cnxn = config.get_connection()
    sql = f'''
        SELECT *
        FROM iemocap.Model_Mutual_Information
        WHERE dataset_config = '{dataset_config}'
        AND y_config = '{y_config}'
        ORDER BY 1,2,3
        '''

    # print(sql)
    df = pd.read_sql(sql=sql, con=cnxn)
    cnxn.close()

    weight_dict = {}
    weight_dict['weights_prosody_tone_happiness'] = df[df['key'] == 'weights_prosody_tone_happiness']['value'].iloc[0]
    weight_dict['weights_prosody_tone_sadness'] = df[df['key'] == 'weights_prosody_tone_sadness']['value'].iloc[0]
    weight_dict['weights_prosody_tone_anger'] = df[df['key'] == 'weights_prosody_tone_anger']['value'].iloc[0]
    weight_dict['weights_prosody_tone_fear'] = df[df['key'] == 'weights_prosody_tone_fear']['value'].iloc[0]

    weight_dict['weights_actions_au05'] = df[df['key'] == 'weights_actions_au05']['value'].iloc[0]
    weight_dict['weights_actions_au17'] = df[df['key'] == 'weights_actions_au17']['value'].iloc[0]
    weight_dict['weights_actions_au20'] = df[df['key'] == 'weights_actions_au20']['value'].iloc[0]
    weight_dict['weights_actions_au25'] = df[df['key'] == 'weights_actions_au25']['value'].iloc[0]

    weight_dict['weights_y_lag1'] = df[df['key'] == 'weights_y_lag1']['value'].iloc[0]
    weight_dict['weights_y_lag2'] = df[df['key'] == 'weights_y_lag2']['value'].iloc[0]
    weight_dict['weights_y_lag3'] = df[df['key'] == 'weights_y_lag3']['value'].iloc[0]
    weight_dict['weights_y_lag4'] = df[df['key'] == 'weights_y_lag4']['value'].iloc[0]

    weight_dict['weights_y_lead1'] = df[df['key'] == 'weights_y_lead1']['value'].iloc[0]
    weight_dict['weights_y_lead2'] = df[df['key'] == 'weights_y_lead2']['value'].iloc[0]
    weight_dict['weights_y_lead3'] = df[df['key'] == 'weights_y_lead3']['value'].iloc[0]
    weight_dict['weights_y_lead4'] = df[df['key'] == 'weights_y_lead4']['value'].iloc[0]

    return weight_dict


def get_weight_dict(hyperparams):
    multilabel_aux = hyperparams['multilabel_aux']
    mainlabel_weight = hyperparams['mainlabel_weight']
    aux_weights_assignment = hyperparams['aux_weights_assignment']
    y_config = hyperparams['y_config']

    weight_dict = {}
    #mainlabel_weight = 0.9
    #multilabel_aux = 'ap'

    # Initialize
    weight_dict['weights_main'] = round(float(mainlabel_weight),4)


    # Initialize as single task

    weight_dict['weights_prosody_tone_happiness'] = 0
    weight_dict['weights_prosody_tone_sadness']   = 0
    weight_dict['weights_prosody_tone_anger']     = 0
    weight_dict['weights_prosody_tone_fear']      = 0
    weight_dict['weights_actions_au05']           = 0
    weight_dict['weights_actions_au17']           = 0
    weight_dict['weights_actions_au20']           = 0
    weight_dict['weights_actions_au25']           = 0
    weight_dict['weights_y_lag1'] = 0
    weight_dict['weights_y_lag2'] = 0
    weight_dict['weights_y_lag3'] = 0
    weight_dict['weights_y_lag4'] = 0
    weight_dict['weights_y_lead1'] = 0
    weight_dict['weights_y_lead2'] = 0
    weight_dict['weights_y_lead3'] = 0
    weight_dict['weights_y_lead4'] = 0

    if aux_weights_assignment in ['mutualinfo linear', 'mutualinfo softmax']:
        mi_weights_dict = get_mutualinfo_weights(hyperparams)

    # Overwrite initializations by sections
    if 'a' in multilabel_aux:
        if aux_weights_assignment == 'random':
            weight_dict['weights_actions_au05'] = random.randint(0, 100)
            weight_dict['weights_actions_au17'] = random.randint(0, 100)
            weight_dict['weights_actions_au20'] = random.randint(0, 100)
            weight_dict['weights_actions_au25'] = random.randint(0, 100)
        elif aux_weights_assignment in ['mutualinfo linear', 'mutualinfo softmax']:
            weight_dict['weights_actions_au05'] = mi_weights_dict['weights_actions_au05']
            weight_dict['weights_actions_au17'] = mi_weights_dict['weights_actions_au17']
            weight_dict['weights_actions_au20'] = mi_weights_dict['weights_actions_au20']
            weight_dict['weights_actions_au25'] = mi_weights_dict['weights_actions_au25']

    if 'p' in multilabel_aux:
        if aux_weights_assignment == 'random':
            weight_dict['weights_prosody_tone_happiness'] = random.randint(0, 100)
            weight_dict['weights_prosody_tone_sadness'] = random.randint(0, 100)
            weight_dict['weights_prosody_tone_anger'] = random.randint(0, 100)
            weight_dict['weights_prosody_tone_fear'] = random.randint(0, 100)
        elif aux_weights_assignment in ['mutualinfo linear', 'mutualinfo softmax']:
            weight_dict['weights_prosody_tone_happiness'] = mi_weights_dict['weights_prosody_tone_happiness']
            weight_dict['weights_prosody_tone_sadness'] = mi_weights_dict['weights_prosody_tone_sadness']
            weight_dict['weights_prosody_tone_anger'] = mi_weights_dict['weights_prosody_tone_anger']
            weight_dict['weights_prosody_tone_fear'] = mi_weights_dict['weights_prosody_tone_fear']

    if 'h' in multilabel_aux:
        if aux_weights_assignment == 'random':
            weight_dict['weights_y_lag1'] = random.randint(0, 100)
            weight_dict['weights_y_lag2'] = random.randint(0, 100)
            weight_dict['weights_y_lag3'] = random.randint(0, 100)
            weight_dict['weights_y_lag4'] = random.randint(0, 100)
        elif aux_weights_assignment in ['mutualinfo linear', 'mutualinfo softmax']:
            weight_dict['weights_y_lag1'] = mi_weights_dict['weights_y_lag1']
            weight_dict['weights_y_lag2'] = mi_weights_dict['weights_y_lag2']
            weight_dict['weights_y_lag3'] = mi_weights_dict['weights_y_lag3']
            weight_dict['weights_y_lag4'] = mi_weights_dict['weights_y_lag4']

    if 'f' in multilabel_aux:
        if aux_weights_assignment == 'random':
            weight_dict['weights_y_lead1'] = random.randint(0, 100)
            weight_dict['weights_y_lead2'] = random.randint(0, 100)
            weight_dict['weights_y_lead3'] = random.randint(0, 100)
            weight_dict['weights_y_lead4'] = random.randint(0, 100)
        elif aux_weights_assignment in ['mutualinfo linear', 'mutualinfo softmax']:
            weight_dict['weights_y_lead1'] = mi_weights_dict['weights_y_lead1']
            weight_dict['weights_y_lead2'] = mi_weights_dict['weights_y_lead2']
            weight_dict['weights_y_lead3'] = mi_weights_dict['weights_y_lead3']
            weight_dict['weights_y_lead4'] = mi_weights_dict['weights_y_lead4']



    # Converting Dict into Array
    x = np.array([weight_dict['weights_prosody_tone_happiness'],
                  weight_dict['weights_prosody_tone_sadness'],
                  weight_dict['weights_prosody_tone_anger'],
                  weight_dict['weights_prosody_tone_fear'],
                  weight_dict['weights_actions_au05'],
                  weight_dict['weights_actions_au17'],
                  weight_dict['weights_actions_au20'],
                  weight_dict['weights_actions_au25'],
                  weight_dict['weights_y_lag1'],
                  weight_dict['weights_y_lag2'],
                  weight_dict['weights_y_lag3'],
                  weight_dict['weights_y_lag4'],
                  weight_dict['weights_y_lead1'],
                  weight_dict['weights_y_lead2'],
                  weight_dict['weights_y_lead3'],
                  weight_dict['weights_y_lead4'],
                 ])

    m=x
    # Linear Weighting Or Softmax Weighting
    if sum(x)>0:
        m = (x)/ sum(x)
        np.sum(m)
    if aux_weights_assignment == 'mutualinfo softmax':
        m = np.exp(x) / sum(np.exp(x))
        np.sum(m)


    # Sum of weighting should be scaled downwards to sum up to remaining weight
    remaining_weights = round(1-mainlabel_weight,4)
    m = m * remaining_weights

    if multilabel_aux == 'None':  # single task
        for i in range(len(m)):
            m[i] = 0

    # Reassign back to dict
    weight_dict['weights_prosody_tone_happiness']=m[0]
    weight_dict['weights_prosody_tone_sadness']  =m[1]
    weight_dict['weights_prosody_tone_anger']    =m[2]
    weight_dict['weights_prosody_tone_fear']     =m[3]
    weight_dict['weights_actions_au05']          =m[4]
    weight_dict['weights_actions_au17']          =m[5]
    weight_dict['weights_actions_au20']          =m[6]
    weight_dict['weights_actions_au25']          =m[7]
    weight_dict['weights_y_lag1']                =m[8]
    weight_dict['weights_y_lag2']                =m[9]
    weight_dict['weights_y_lag3']                =m[10]
    weight_dict['weights_y_lag4']                =m[11]
    weight_dict['weights_y_lead1']               =m[12]
    weight_dict['weights_y_lead2']              =m[13]
    weight_dict['weights_y_lead3']              =m[14]
    weight_dict['weights_y_lead4']              =m[15]
    weight_dict['weights_main']                  =mainlabel_weight

    return weight_dict



def random_runsettings(input_config, dataset_config, y_config, multilabel_aux,
                       aux_weights_assignment, aux_hierarchy_config,
                       num_gru_config, num_translator, dynamic_loss_weights,
                       code_development):
    # input_config = 'vpa', dataset_config = 'i', y_config = 'v', multilabel_aux = 'aphf',
    # aux_weights_assignment = 'random', aux_hierarchy_config='rock', code_development='y'
    hyperparams={}
    hyperparams['WV_Type']                  = 'Glove'


    hyperparams['WV_Dim']                   = 300 # EMPATH 194

    min_batch_size                          = 32

    hyperparams['code_development'] = code_development
    hyperparams['Num_GRU'] = num_gru_config



    if code_development == 'n':
        # PROD
        hyperparams['context_k'] = random.randint(0, 30)
        max_batch_size = int(1024 / (hyperparams['context_k'] + 1))  # rule of thumb context_k x batch_size <= 1024
        hyperparams['Batch_Size']           = random.randint(min_batch_size,max_batch_size)

        hyperparams['Max_Epoch']            = 350
        hyperparams['Num_GRU_Aux']          = 257 - hyperparams['Num_GRU']  # Total of 257 bidirectional GRU

    else:
        # Code Devlopment
        hyperparams['context_k'] = 2
        max_batch_size = int(1024 / (hyperparams['context_k'] + 1))  # rule of thumb context_k x batch_size <= 1024
        hyperparams['Batch_Size']           = random.randint(min_batch_size,max_batch_size)
        hyperparams['Max_Epoch']            = 4  # 350
        hyperparams['Num_GRU_Aux']          = 30 - hyperparams['Num_GRU']
        assert hyperparams['Num_GRU_Aux'] > 0



    hyperparams['Direction']                = 'Bidirectional'

    hyperparams['Patience']                 = 2000
    hyperparams['Sentence_Count']           = hyperparams['context_k'] + 1
    hyperparams['Sentence_Length']          = 120
    hyperparams['ReduceLROnPlateau']        = 0
    
    hyperparams['GRU_dropout']              = random.randint(1,50) * 0.01
    hyperparams['GRU_recurrent_dropout']    = random.randint(1,50) * 0.01
    hyperparams['l2_regularization']        = 0.0 #2 ** (random.randint(-2000,-100) * 0.01)
    hyperparams['fullrun_description']      = get_fullrun_description()
    hyperparams['save_period']              = 500

    # Optimizer Hyperparams
    hyperparams['Optimizer_Dict']                    = {}
    hyperparams['Optimizer_Dict']['SGD']             = {}
    lr_power = random.randint(-1000, -500) * 0.01 #random.randint(-1100, -300) * 0.01
    hyperparams['Optimizer_Dict']['SGD']['lr']       = 2**lr_power
    hyperparams['Optimizer_Dict']['SGD']['momentum'] = 0.9 # untunable
        
    
    hyperparams['Optimizer']                = random.choice(['SGD'])
    hyperparams['LR']                       = hyperparams['Optimizer_Dict']['SGD']['lr']

    # Inputs
    hyperparams['input_config']   = input_config
    hyperparams['dataset_config'] = dataset_config
    hyperparams['y_config']       = y_config


    # Multilabel Hyperparameters
    hyperparams['multilabel_aux'] = multilabel_aux
    if hyperparams['multilabel_aux'] == 'None': # string none entered in the arg parse instead of None Python object
        mainlabel_weight = 1.0
        hyperparams['mainlabel_weight'] = mainlabel_weight
        hyperparams['aux_weights_assignment'] = 'None'

    else:
        mainlabel_weight = random.randint(51,99) * 0.01 # Main weight will surely be the highest
        hyperparams['mainlabel_weight'] = mainlabel_weight
        hyperparams['aux_weights_assignment'] = aux_weights_assignment

    weight_dict = get_weight_dict(hyperparams)
    hyperparams = dict(hyperparams, **weight_dict)  # union dictionaries

    # ROCK
    hyperparams['aux_hierarchy_config'] = aux_hierarchy_config

    # Translator before Attention
    hyperparams['num_translator'] = num_translator

    # Dynamic weight losses
    hyperparams['dynamic_loss_weights'] = dynamic_loss_weights

    hyperparams['train_seed'] = 0

    return (hyperparams)
    

def write_runsettings(hyperparams):
    # most hyperparams are given as inputs

    hyperparams['Model_Unixtime'] = hyperparams['fullrun_description']


    # some file_paths are dependent on hyperparams
    file_paths = {}
    file_paths["SAVED_MODEL_DIR"] = 'saved_models'
    file_paths["SAVED_MODEL_FILENAME"] = 'rapport_model.h5'
    
    # embeddings path is dependent on type of word vector        
    if hyperparams['WV_Type'] == 'empath': 
        file_paths["EMBEDDINGS_PATH"] = 'saved_models/empath.txt'
    
    elif hyperparams['WV_Type'] == 'vader':
        file_paths["EMBEDDINGS_PATH"] = 'saved_models/vader.txt'
    
    else:
        file_paths["EMBEDDINGS_PATH"] = 'saved_models/glove.6B.' + str(int(hyperparams['WV_Dim'])) + 'd.txt'

    file_paths["expt_model_dir"] = os.path.join(file_paths["SAVED_MODEL_DIR"], 'expt', 
             str(hyperparams['Model_Unixtime']))
    
    hyperparams_path = os.path.join('saved_models', 'hyperparams.pkl')
    file_paths_path = os.path.join('saved_models', 'file_paths.pkl')
    
    with open(hyperparams_path, 'wb') as handle:
        pickle.dump(hyperparams, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(file_paths_path, 'wb') as handle:
        pickle.dump(file_paths, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return(None)

def get_runsettings(): 
    
    hyperparams_path = os.path.join('saved_models', 'hyperparams.pkl')
    file_paths_path = os.path.join('saved_models', 'file_paths.pkl')
    
    with open(hyperparams_path, 'rb') as handle:
        hyperparams = pickle.load(handle)
        
    with open(file_paths_path, 'rb') as handle:
        file_paths = pickle.load(handle)
    
    return file_paths, hyperparams
