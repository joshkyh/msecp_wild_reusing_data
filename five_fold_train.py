import argparse
import logging
import os as os
import time
import traceback

import numpy as np
import pandas as pd
from keras import backend as K
from numpy.random import seed
from tensorflow import set_random_seed

import util.config as config
import util.researchdb as researchdb
from hnatt import HNATT
from util.model_logging import insert_model_log, insert_complete_run, update_model_hiplot, \
    insert_model_params_count
from util.runsettings import random_runsettings, write_runsettings, get_runsettings


def get_compute_config(compute_id):
    # compute_id ='home'
    cnxn = config.get_connection()


    sql = f'''
            SELECT *
            FROM iemocap.Model_Artemis
            WHERE compute_id = '{compute_id}'
            '''
    print(sql)
    df = pd.read_sql(sql=sql, con=cnxn)
    cnxn.close()

    compute_config  = {}
    for i in range(len(df)):
        row = df.iloc[i]
        compute_config[row['param_key']] = row['param_value']

    logging.info(compute_config)

    return compute_config



def get_y_config(input_config,
                dataset_config,
                multilabel_aux,
                aux_weights_assignment,
                aux_hierarchy_config,
                num_gru_config,
                num_translator,
                dynamic_loss_weights):
    '''
    Artemis load balancing
    Some tasks are inherently harder, there is no improvement after a long time
    therefore median stopping rule cannot help
    Instead, we load balance by the number of attempts made at the problem at hand
    '''
    # input_config = 'vpa'
    # dataset_config = 'i'
    # multilabel_aux = 'aphf'
    # aux_weights_assignment = 'mutualinfo softmax'
    # aux_hierarchy_config = 'rock'
    # num_gru_config = '128'
    # num_translator = 'same as prediction'


    def _get_counts(y_config):
        cnxn = config.get_connection()
        # input_config = 'vpa'
        # dataset_config = 'i'
        # multilabel_aux = 'aphf'
        # y_config = 'v'
        sql = f'''
                SELECT *
                FROM iemocap.Model_Hyperparameter_Dict
                WHERE HP_Name IN ('input_config'          
                ,'dataset_config'          
                ,'multilabel_aux'          
                ,'aux_weights_assignment'  
                ,'aux_hierarchy_config'    
                ,'num_gru'                 
                ,'y_config'                
                ,'num_translator'
                , 'dynamic_loss_weights')
                
                '''
        #print(sql)
        df = pd.read_sql(sql=sql, con=cnxn)
        cnxn.close()

        df = df.pivot(index='Model_Unixtime', columns='HP_Name', values='HP_Value')
        df = df.reset_index(drop=True)
        print('df',df)
        count = 0
        if len(df) >0:
            df2 = df.loc[(df['input_config'] == input_config)
                         & (df['dataset_config'] == dataset_config)
                         & (df['multilabel_aux'] == multilabel_aux)
                         & (df['dataset_config'] == dataset_config)
                         & (df['aux_weights_assignment'] == aux_weights_assignment)
                         & (df['aux_hierarchy_config'] == aux_hierarchy_config)
                         & (df['Num_GRU'] == str(num_gru_config))
                         & (df['num_translator'] == num_translator)
                         & (df['dynamic_loss_weights'] == dynamic_loss_weights)
                         & (df['y_config'] == y_config)
                         ]

            count = len(df2.index)
            print('df2', df2)



        result = {}
        result['input_config'] = input_config
        result['dataset_config'] = dataset_config
        result['multilabel_aux'] = multilabel_aux
        result['aux_weights_assignment'] = aux_weights_assignment
        result['aux_hierarchy_config'] = aux_hierarchy_config
        result['num_gru_config'] = num_gru_config
        result['num_translator'] = num_translator
        result['dynamic_loss_weights'] = dynamic_loss_weights

        result['y_config'] = y_config
        result['count'] = count
        result_df = pd.DataFrame(result, index=[0])
        print(result_df)

        return result_df



    if dataset_config == 'i':
        counts_df = _get_counts(y_config='c')
        counts_df = pd.concat([counts_df,_get_counts(y_config='v')])
        counts_df = pd.concat([counts_df,_get_counts(y_config='a')])
        counts_df = pd.concat([counts_df,_get_counts(y_config='d')])
        counts_df['randomness'] = np.random.randint(1,100, counts_df.shape[0])
        counts_df = counts_df.sort_values(by=['count', 'randomness'], ascending=True)
        y_config_to_run = counts_df.iloc[0]['y_config']

        print(counts_df[['y_config', 'count', 'randomness']])
        return y_config_to_run

    if dataset_config == 's':
        counts_df = _get_counts(y_config='a')
        counts_df = pd.concat([counts_df, _get_counts(y_config='i')])
        counts_df = pd.concat([counts_df, _get_counts(y_config='p')])
        counts_df = pd.concat([counts_df, _get_counts(y_config='v')])
        counts_df['randomness'] = np.random.randint(1, 100, counts_df.shape[0])
        counts_df = counts_df.sort_values(by=['count', 'randomness'], ascending=True)
        y_config_to_run = counts_df.iloc[0]['y_config']

        print(counts_df[['y_config', 'count', 'randomness']])
        return y_config_to_run


def one_fold_train (hyperparams):

    write_runsettings(hyperparams)

    # Get hyperparams
    file_paths, hyperparams = get_runsettings()

    # test
    print('')
    print('-----------------------------------------------------------------------------------------------')
    print(hyperparams)
    code_development = hyperparams['code_development']


    # Get Data
    (train_x, train_y), (dev_x, dev_y), (test_x, test_y) = researchdb.load_data(hyperparams=hyperparams)

    # log hyperparams before training
    if code_development=='n':
        insert_model_log(hyperparams)

    # initialize HNATT
    if not os.path.exists(file_paths["expt_model_dir"]):
        os.makedirs(file_paths["expt_model_dir"])

    # Training
    h = HNATT()
    h.train(train_x, train_y, dev_x, dev_y,test_x, test_y, file_paths, hyperparams)

    # Training complete
    if code_development=='n':
        # Log Completion
        insert_complete_run(hyperparams)

        # Log Model parameters count
        trainable_count = int(
            np.sum([K.count_params(p) for p in set(h.model.trainable_weights)]))
        non_trainable_count = int(
            np.sum([K.count_params(p) for p in set(h.model.non_trainable_weights)]))
        insert_model_params_count(hyperparams, trainable_count, non_trainable_count)

        # Update hiplot
        update_model_hiplot()

    # Clean up
    del (file_paths)
    del (train_x)
    del (train_y)
    del (dev_x)
    del (dev_y)
    del (h)

    return(None)        
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Argument Parsing')
    parser.add_argument("--compute_id", type=str, help="eqcl / avec / ham / home", default='home')
    parser.add_argument("--code_development", type=str, help="y/n", default='y')

    args = parser.parse_args()
    code_development = args.code_development
    compute_id = args.compute_id

    code_development = 'n'
    compute_id = 'home'

    assert code_development in ['y', 'n']
    assert compute_id in ['eqcl', 'avec', 'home', 'ham']

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s')





    # Bash is doing the while true loop, alternating with git pull
    try:
        if code_development == 'y':
            input_config = 'vpa'
            dataset_config = 's'
            multilabel_aux = 'aphf'
            aux_weights_assignment = 'mutualinfo softmax'
            aux_hierarchy_config = 'flat'
            num_gru_config = 10
            num_translator = 'None'
            dynamic_loss_weights = 'n'

        else:
            # Control what to run on Artemis with SQL Server
            compute_config = get_compute_config(compute_id)
            input_config = compute_config['input_config']
            dataset_config = compute_config['dataset_config']
            multilabel_aux = compute_config['multilabel_aux']
            aux_weights_assignment = compute_config['aux_weights_assignment']
            aux_hierarchy_config = compute_config['aux_hierarchy_config']
            num_gru_config = int(compute_config['num_gru_config'])
            num_translator = compute_config['num_translator']
            dynamic_loss_weights = 'n' #compute_config['dynamic_loss_weights']

        # Input checks
        assert input_config in ['v', 'vpa']
        assert dataset_config in ['i', 's']
        assert multilabel_aux in ['None', 'ap', 'aphf']
        assert aux_weights_assignment in ['None', 'random', 'mutualinfo linear', 'mutualinfo softmax']
        assert aux_hierarchy_config in ['flat', 'rock']
        assert num_gru_config <= 256
        assert num_translator in ['None', 'same as prediction']
        assert dynamic_loss_weights in ['n', 'y']


        if multilabel_aux == 'None':
            assert aux_weights_assignment == 'None'
    except Exception as ex:
        logging.exception('Something is wrong for input')
        print(ex)
        traceback.print_exception(type(ex), ex, ex.__traceback__)
        logging.exception('Wait 5 seconds')
        time.sleep(5)


    # Load Balancing
    y_config = get_y_config(input_config,
                            dataset_config,
                            multilabel_aux,
                            aux_weights_assignment,
                            aux_hierarchy_config,
                            num_gru_config,
                            num_translator,
                            dynamic_loss_weights)
    if code_development == 'y':
        y_config = 'v'

    y_config = 'c'
    print('y_config', y_config)
    # Get Hyperparameters
    hyperparams = random_runsettings(input_config, dataset_config, y_config, multilabel_aux,
                                     aux_weights_assignment, aux_hierarchy_config, num_gru_config,
                                     num_translator,
                                     dynamic_loss_weights, code_development)

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

    try:
        # Ensuring reproducibility for layers initialization
        seed(hyperparams['train_seed'])
        set_random_seed(hyperparams['train_seed'])

        one_fold_train(hyperparams)
    except Exception as ex:
        print(ex)
        traceback.print_exception(type(ex), ex, ex.__traceback__)
        logging.exception('Wait 5 seconds')
        time.sleep(5)
