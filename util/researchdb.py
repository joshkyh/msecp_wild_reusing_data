# -*- coding: utf-8 -*-
'''
To retrieve model data from db
'''

import logging
import os

import pandas as pd
import numpy as np
from tqdm import tqdm

from util.text_util import normalize
import util.config as config
import util.runsettings as RS

tqdm.pandas()


def chunk_to_arrays(chunk, hyperparams):
    # chunk=train_set
    x = chunk['text_tokens'].values

    # Primary Task
    if hyperparams['dataset_config'] == 'i':  # IEMOCAP
        if hyperparams['y_config'] == 'v':
            y = chunk['val'].values
        if hyperparams['y_config'] == 'a':
            y = chunk['act'].values
        if hyperparams['y_config'] == 'd':
            y = chunk['dom'].values
        if hyperparams['y_config'] == 'c':
            y = chunk['categorical_label'].values

    elif hyperparams['dataset_config'] == 's':  # SEMAINE
        if hyperparams['y_config'] == 'a':
            y = chunk['act'].values
        if hyperparams['y_config'] == 'i':
            y = chunk['int'].values
        if hyperparams['y_config'] == 'p':
            y = chunk['pow'].values
        if hyperparams['y_config'] == 'v':
            y = chunk['val'].values

    chunk.columns
    # Aux Task, we always append. If we don't use it, the weight of aux is 0.
    y = np.stack([y,
                  chunk['tone_happiness'].values, #1
                  chunk['tone_sadness'].values, #2
                  chunk['tone_anger'].values, #3
                  chunk['tone_fear'].values, #4
                  chunk['AU_05'].values, #5
                  chunk['AU_17'].values, #6
                  chunk['AU_20'].values, #7
                  chunk['AU_25'].values, #8
                  chunk['y_lag1'].values, #9
                  chunk['y_lag2'].values, #10
                  chunk['y_lag3'].values, #11
                  chunk['y_lag4'].values, #12
                  chunk['y_lead1'].values, #13
                  chunk['y_lead2'].values, #14
                  chunk['y_lead3'].values, #15
                  chunk['y_lead4'].values #16
                  ], axis=1)

    return x, y


def get_input_set(hyperparams=None):
    input_config = hyperparams['input_config']

    required_families = []

    if 'vpa' in input_config:
        required_families += ["'vpa'"]


    elif 'vp' in input_config:
        required_families += ["'vp'"]
    elif 'v' in input_config:
        required_families += ["'v'"]

    in_string = ','.join(required_families)
    in_string = '(' + in_string + ')'
    return in_string


def download_fine_df(hyperparams=None):
    # IEMOCAP
    if hyperparams['dataset_config'] == 'i':
        sql_like_string = get_input_set(hyperparams)
        print('Downloading Fine Text Blob')
        cnxn = config.get_connection()
        sql = 'SELECT P.session_id, P.scenario_id, P.talkturn_ID, f.text_blob as text, E.val, E.act, E.dom, ' #ang 0, hap_exc 1, neu 2, sad 3
        sql += '''CASE WHEN E.categorical_label IN ('ang') THEN 0  
               WHEN E.categorical_label IN ('hap', 'exc') THEN 1 
               WHEN E.categorical_label IN ('neu') THEN 2  
               WHEN E.categorical_label IN ('sad') THEN 3  
               ELSE -1 END AS categorical_label,  '''
        sql += 'P.Fold_Num, P.talkturn_sequence  '
        sql += 'FROM [iemocap].[Narrative_Fine] F WITH(NOLOCK) '
        sql += "INNER JOIN [iemocap].[emoevaluation] E ON F.Video_ID = CONCAT(E.session_id, '_', E.scenario_id) AND F.talkturn_ID = E.talkturn_id "
        sql += 'INNER JOIN iemocap.Model_Partition P ON E.scenario_id = P.scenario_id AND E.session_id = P.session_id AND E.talkturn_id = P.talkturn_id '
        sql += f'WHERE F.family IN {sql_like_string} '
        sql += 'ORDER BY P.session_id, P.scenario_id, P.talkturn_sequence'

        print(sql)

        fine_df = pd.read_sql(sql=sql, con=cnxn)
        cnxn.close()

        fine_df['categorical_label'] = fine_df['categorical_label'].astype(int)

        # We cannot filter for categorical label here, need to form the context first. Else missing talkturns for context
        fine_df.columns
        fine_df = fine_df.sort_values(['session_id','scenario_id','talkturn_sequence'])

    # SEMAINE
    elif hyperparams['dataset_config'] == 's':
        sql_like_string = get_input_set(hyperparams)
        print('Downloading Fine Text Blob')
        cnxn = config.get_connection()
        sql = f'''
        SELECT 
        M.Recording AS session_id,
        M.[order] AS scenario_id,
        M.talkturn_number AS talkturn_ID,
        N.Turn_ID AS talkturn_sequence, 
        M.[text] as [text],
        M.act,
        M.int,
        M.pow,
        M.val,
        M.Fold_Num,
        M.complete_aux,
        M.complete_y
        FROM SEMAINE.MODEL_PARTITION M
        INNER JOIN semaine.Narrative_Fine N ON M.Recording = N.Recording AND M.[order] = n.[Order] AND M.talkturn_number = N.talkturn_number AND m.family = N.family
        where m.family='vpa'
        order by 1,2,3,4
        '''

        print(sql)

        fine_df = pd.read_sql(sql=sql, con=cnxn)
        cnxn.close()



        # We cannot filter for categorical label here, need to form the context first. Else missing talkturns for context
        fine_df.columns

        fine_df = fine_df.sort_values(['session_id','scenario_id','talkturn_sequence'])

    return fine_df


def get_context_text_df(fine_df, hyperparams):
    # Extract the value of k
    context_k = hyperparams['context_k']

    # Self join with itself, with no constraints
    merged = fine_df.merge(fine_df, on=['session_id', 'scenario_id'])
    merged.columns

    # Filter the self join for an non-equi join
    merged = merged[(merged.talkturn_sequence_x >= merged.talkturn_sequence_y) *
                    (merged.talkturn_sequence_x <= merged.talkturn_sequence_y + context_k)
                    ]

    # Sort df in order to get ready for text concatenation across talkturns
    merged = merged[['session_id', 'scenario_id', 'talkturn_ID_x', 'text_y', 'talkturn_sequence_y']]
    merged = merged.sort_values(['session_id', 'scenario_id', 'talkturn_ID_x', 'talkturn_sequence_y'],
                                ascending=[True, True, True, False])

    # text concat across talkturns
    merged['context_text'] = merged[['session_id', 'scenario_id', 'talkturn_ID_x', 'text_y']]. \
        groupby(['session_id', 'scenario_id', 'talkturn_ID_x'])['text_y'].transform(lambda x: ' '.join(x))
    merged = merged[['session_id', 'scenario_id', 'talkturn_ID_x', 'context_text']].drop_duplicates()
    merged.columns = [['session_id', 'scenario_id', 'talkturn_ID', 'context_text']]

    # Drop the multiindex after group by
    merged.columns = merged.columns.get_level_values(0)
    fine_df = fine_df.reset_index(drop=True)

    # Export
    result = merged.merge(fine_df, on=['session_id', 'scenario_id', 'talkturn_ID'])

    if hyperparams['dataset_config'] == 'i':

        result = result[
            ['session_id', 'scenario_id', 'talkturn_ID', 'context_text', 'val', 'act', 'dom', 'categorical_label',
             'Fold_Num', 'talkturn_sequence']]

        # Remove talkturns associated to negative classification label, out of scope
        if hyperparams['y_config'] == 'c':
            result = result[result['categorical_label'] != -1]

    elif hyperparams['dataset_config'] == 's':
        result.columns

        # drop rows with incomplete y labels or aux labels
        keep_rows = result['complete_y'] * result['complete_aux']
        result = result[keep_rows == 1]

        result = result[
            ['session_id', 'scenario_id', 'talkturn_ID', 'context_text',
             'act', 'int', 'pow', 'val',
             'Fold_Num', 'talkturn_sequence']]
    result = result.sort_values(['session_id', 'scenario_id', 'talkturn_sequence'])

    return result

def download_aux_labels(hyperparams):
    print('Downloading Aux Labels')

    cnxn = config.get_connection()

    if hyperparams['dataset_config'] == 'i':
        sql = '''
        SELECT V.session_id, V.scenario_id, V.talkturn_id
        , percentile_happiness, percentile_sadness, percentile_anger, percentile_fear 
        ,A.Percentile_AU05, A.Percentile_AU17, A.Percentile_AU20, A.Percentile_AU25
        ,H.[cat_Lag4]
        ,H.[cat_Lag3]
        ,H.[cat_Lag2]
        ,H.[cat_Lag1]
        ,H.[cat_Lead1]
        ,H.[cat_Lead2]
        ,H.[cat_Lead3]
        ,H.[cat_Lead4]
        ,H.[val_Lag4]
        ,H.[val_Lag3]
        ,H.[val_Lag2]
        ,H.[val_Lag1]
        ,H.[val_Lead1]
        ,H.[val_Lead2]
        ,H.[val_Lead3]
        ,H.[val_Lead4]
        ,H.[act_Lag4]
        ,H.[act_Lag3]
        ,H.[act_Lag2]
        ,H.[act_Lag1]
        ,H.[act_Lead1]
        ,H.[act_Lead2]
        ,H.[act_Lead3]
        ,H.[act_Lead4]
        ,H.[dom_Lag4]
        ,H.[dom_Lag3]
        ,H.[dom_Lag2]
        ,H.[dom_Lag1]
        ,H.[dom_Lead1]
        ,H.[dom_Lead2]
        ,H.[dom_Lead3]
        ,H.[dom_Lead4]
        FROM iemocap.aux_vokaturi  V
        INNER JOIN iemocap.aux_au  A ON V.session_id = A.session_id AND V.scenario_id = A.scenario_id AND V.talkturn_id = A.talkturn_id
        INNER JOIN iemocap.aux_history_future H ON V.session_id = H.session_id AND V.scenario_id = H.scenario_id AND V.talkturn_id = H.talkturn_id
        ORDER BY 1,2,3
        '''

    elif hyperparams['dataset_config'] == 's':
        sql = '''
        SELECT P.Recording as session_id, P.[ORDER] as scenario_id, P.talkturn_number as talkturn_ID 
        , V.percentile_happiness, V.percentile_sadness, V.percentile_anger, V.percentile_fear
        , A.percentile_AU05, A.Percentile_AU17, A.percentile_AU20, A.percentile_AU25
        , act_Lag4, act_Lag3, act_Lag2,	act_Lag1, act_Lead1, act_Lead2, act_Lead3, act_Lead4
        , int_Lag4,	int_Lag3, int_Lag2,	int_Lag1, int_Lead1, int_Lead2,	int_Lead3, int_Lead4
        , pow_Lag4,	pow_Lag3, pow_Lag2,	pow_Lag1, pow_Lead1, pow_Lead2,	pow_Lead3, pow_Lead4
        , val_Lag4,	val_Lag3, val_Lag2,	val_Lag1, val_Lead1, val_Lead2,	val_Lead3, val_Lead4
        
        FROM semaine.model_partition P
        INNER JOIN semaine.aux_vokaturi V             ON P.Recording = V.Recording AND P.[Order] = V.[Order] AND P.[talkturn_number] = V.talkturn_number
        INNER JOIN semaine.aux_au A                   ON P.Recording = A.Recording AND P.[Order] = A.[Order] AND P.[talkturn_number] = A.talkturn_number
        INNER JOIN semaine.aux_history_future H       ON P.Recording = H.Recording AND P.[Order] = H.[Order] AND P.[talkturn_number] = H.talkturn_number
        WHERE P.Complete_aux = 1
        AND P.Complete_y = 1
        AND family = 'v' --hard coded here, doesn't matter as we're not using the text
        '''
    # print(sql)
    df = pd.read_sql(sql=sql, con=cnxn)
    cnxn.close()



    # Rename columns
    df.columns
    if hyperparams['dataset_config'] == 'i':
        df.columns = ['session_id', 'scenario_id', 'talkturn_ID',
                      'tone_happiness', 'tone_sadness', 'tone_anger', 'tone_fear',
                      'AU_05', 'AU_17', 'AU_20', 'AU_25',
                      'cat_lag4', 'cat_lag3', 'cat_lag2', 'cat_lag1',
                      'cat_lead1', 'cat_lead2', 'cat_lead3', 'cat_lead4',
                      'val_lag4', 'val_lag3', 'val_lag2', 'val_lag1',
                      'val_lead1', 'val_lead2', 'val_lead3', 'val_lead4',
                      'act_lag4', 'act_lag3', 'act_lag2', 'act_lag1',
                      'act_lead1', 'act_lead2', 'act_lead3', 'act_lead4',
                      'dom_lag4', 'dom_lag3', 'dom_lag2', 'dom_lag1',
                      'dom_lead1', 'dom_lead2', 'dom_lead3', 'dom_lead4'
                      ]
    elif hyperparams['dataset_config'] == 's':
        df.columns = ['session_id', 'scenario_id', 'talkturn_ID',
                      'tone_happiness', 'tone_sadness', 'tone_anger', 'tone_fear',
                      'AU_05', 'AU_17', 'AU_20', 'AU_25',
                      'act_lag4', 'act_lag3', 'act_lag2', 'act_lag1','act_lead1', 'act_lead2', 'act_lead3', 'act_lead4',
                      'int_lag4', 'int_lag3', 'int_lag2', 'int_lag1','int_lead1', 'int_lead2','int_lead3', 'int_lead4',
                      'pow_lag4', 'pow_lag3', 'pow_lag2', 'pow_lag1','pow_lead1', 'pow_lead2', 'pow_lead3', 'pow_lead4',
                      'val_lag4', 'val_lag3', 'val_lag2', 'val_lag1','val_lead1', 'val_lead2', 'val_lead3', 'val_lead4'
                      ]

    return df


def get_aux_labels(hyperparams):

    df = download_aux_labels(hyperparams)

    # Depending on the dataset and Y variable, we need to scale the percentile to the same scale of Y variable
    if hyperparams['dataset_config'] == 'i':  # IEMOCAP DATASET
        if hyperparams['y_config'] in ['v', 'd', 'a']:  # The continuous labels range [1,5] = Percentiles [0,1]*4+1
            df['tone_happiness'] = df['tone_happiness'] * 4 + 1
            df['tone_sadness'] = df['tone_sadness'] * 4 + 1
            df['tone_anger'] = df['tone_anger'] * 4 + 1
            df['tone_fear'] = df['tone_fear'] * 4 + 1

            df['AU_05'] = df['AU_05'] * 4 + 1
            df['AU_17'] = df['AU_17'] * 4 + 1
            df['AU_20'] = df['AU_20'] * 4 + 1
            df['AU_25'] = df['AU_25'] * 4 + 1

            # Y config dependent future and history
            if hyperparams['y_config'] == 'v':
                df['y_lag1'] = df['val_lag1']
                df['y_lag2'] = df['val_lag2']
                df['y_lag3'] = df['val_lag3']
                df['y_lag4'] = df['val_lag4']
                df['y_lead1'] = df['val_lead1']
                df['y_lead2'] = df['val_lead2']
                df['y_lead3'] = df['val_lead3']
                df['y_lead4'] = df['val_lead4']
            elif hyperparams['y_config'] == 'a':
                df['y_lag1'] = df['act_lag1']
                df['y_lag2'] = df['act_lag2']
                df['y_lag3'] = df['act_lag3']
                df['y_lag4'] = df['act_lag4']
                df['y_lead1'] = df['act_lead1']
                df['y_lead2'] = df['act_lead2']
                df['y_lead3'] = df['act_lead3']
                df['y_lead4'] = df['act_lead4']
            elif hyperparams['y_config'] == 'd':
                df['y_lag1'] = df['dom_lag1']
                df['y_lag2'] = df['dom_lag2']
                df['y_lag3'] = df['dom_lag3']
                df['y_lag4'] = df['dom_lag4']
                df['y_lead1'] = df['dom_lead1']
                df['y_lead2'] = df['dom_lead2']
                df['y_lead3'] = df['dom_lead3']
                df['y_lead4'] = df['dom_lead4']

        elif hyperparams['y_config'] in ['c']:
            # Classification labels Percentiles [0,1]
            df['tone_happiness'] = df['tone_happiness']
            df['tone_sadness'] = df['tone_sadness']
            df['tone_anger'] = df['tone_anger']
            df['tone_fear'] = df['tone_fear']
            df['AU_05'] = df['AU_05']
            df['AU_17'] = df['AU_17']
            df['AU_20'] = df['AU_20']
            df['AU_25'] = df['AU_25']

            df['y_lag1'] = df['cat_lag1']
            df['y_lag2'] = df['cat_lag2']
            df['y_lag3'] = df['cat_lag3']
            df['y_lag4'] = df['cat_lag4']
            df['y_lead1'] = df['cat_lead1']
            df['y_lead2'] = df['cat_lead2']
            df['y_lead3'] = df['cat_lead3']
            df['y_lead4'] = df['cat_lead4']


    elif hyperparams['dataset_config'] == 's':  # IEMOCAP DATASET
        if hyperparams['y_config'] in ['a','i','p','v']:  # The continuous labels range [-1,1] = Percentiles [0,1]*2-11
            df['tone_happiness'] = df['tone_happiness'] * 2 - 1
            df['tone_sadness'] = df['tone_sadness'] * 2 - 1
            df['tone_anger'] = df['tone_anger'] * 2 - 1
            df['tone_fear'] = df['tone_fear'] * 2 - 1

            df['AU_05'] = df['AU_05'] * 2 - 1
            df['AU_17'] = df['AU_17'] * 2 - 1
            df['AU_20'] = df['AU_20'] * 2 - 1
            df['AU_25'] = df['AU_25'] * 2 - 1

            # Y config dependent future and history
            if hyperparams['y_config'] == 'a':
                df['y_lag1'] = df['act_lag1']
                df['y_lag2'] = df['act_lag2']
                df['y_lag3'] = df['act_lag3']
                df['y_lag4'] = df['act_lag4']
                df['y_lead1'] = df['act_lead1']
                df['y_lead2'] = df['act_lead2']
                df['y_lead3'] = df['act_lead3']
                df['y_lead4'] = df['act_lead4']
            elif hyperparams['y_config'] == 'i':
                df['y_lag1'] = df['int_lag1']
                df['y_lag2'] = df['int_lag2']
                df['y_lag3'] = df['int_lag3']
                df['y_lag4'] = df['int_lag4']
                df['y_lead1'] = df['int_lead1']
                df['y_lead2'] = df['int_lead2']
                df['y_lead3'] = df['int_lead3']
                df['y_lead4'] = df['int_lead4']
            elif hyperparams['y_config'] == 'p':
                df['y_lag1'] = df['pow_lag1']
                df['y_lag2'] = df['pow_lag2']
                df['y_lag3'] = df['pow_lag3']
                df['y_lag4'] = df['pow_lag4']
                df['y_lead1'] = df['pow_lead1']
                df['y_lead2'] = df['pow_lead2']
                df['y_lead3'] = df['pow_lead3']
                df['y_lead4'] = df['pow_lead4']
            elif hyperparams['y_config'] == 'v':
                df['y_lag1'] = df['val_lag1']
                df['y_lag2'] = df['val_lag2']
                df['y_lag3'] = df['val_lag3']
                df['y_lag4'] = df['val_lag4']
                df['y_lead1'] = df['val_lead1']
                df['y_lead2'] = df['val_lead2']
                df['y_lead3'] = df['val_lead3']
                df['y_lead4'] = df['val_lead4']

    return df


def load_data(hyperparams=None):
    contains_any_uppercase = any(x.isupper() for x in hyperparams['input_config'])
    contains_any_lowercase = any(x.islower() for x in hyperparams['input_config'])

    assert (int(contains_any_uppercase) + int(contains_any_lowercase) > 0)  # must have at least an lower or uppercase

    if contains_any_lowercase:
        fine_df = download_fine_df(hyperparams)

        fine_df = get_context_text_df(fine_df, hyperparams)

        if hyperparams['dataset_config'] == 'i':
            fine_df = fine_df.groupby(
            ['session_id', 'scenario_id', 'talkturn_ID', 'val', 'act', 'dom', 'categorical_label', 'Fold_Num'])[
            'context_text'].apply(lambda x: ''.join(x)).reset_index()

        if hyperparams['dataset_config'] == 's':
            fine_df.columns
            fine_df = fine_df.groupby(
            ['session_id', 'scenario_id', 'talkturn_ID', 'act', 'int', 'pow',
             'val', 'Fold_Num'])['context_text'].apply(lambda x: ''.join(x)).reset_index()

        df = fine_df
        # Convert text to token
        print('Normalize x')
        df['text_tokens'] = df['context_text'].progress_apply(lambda x: normalize(x))

        aux_df = get_aux_labels(hyperparams)
        df = df.merge(aux_df)
        df.columns

    # training + validation set

    train_set = df[df.Fold_Num < 3].copy()
    train_set['len'] = train_set['text_tokens'].apply(lambda x: len(x))
    train_set = train_set.sample(frac=1, random_state = hyperparams['train_seed']) #shuffle
    train_x, train_y = chunk_to_arrays(chunk=train_set, hyperparams=hyperparams)

    dev_set = df[df.Fold_Num == 3].copy()
    dev_set = dev_set.sample(frac=1, random_state = hyperparams['train_seed']) #shuffle
    dev_x, dev_y = chunk_to_arrays(dev_set, hyperparams)

    test_set = df[df.Fold_Num == 4].copy()
    test_set = test_set.sample(frac=1, random_state = hyperparams['train_seed']) #shuffle
    test_x, test_y = chunk_to_arrays(test_set, hyperparams)

    print('finished loading')

    return (train_x, train_y), (dev_x, dev_y), (test_x, test_y)


if __name__ == "__main__":
    PROJECT_PATH = config.get_project_path()

    hyperparams = RS.random_runsettings(input_config='vpa',
                                        #dataset_config='i',y_config='a',
                                        dataset_config='s', y_config='i',
                                        multilabel_aux='None',
                                        aux_weights_assignment='None',
                                        num_gru_config=20,
                                        num_translator='None',
                                        aux_hierarchy_config='rock',
                                        dynamic_loss_weights='n',
                                        code_development='y')
    hyperparams['y_config']
    (train_x, train_y), (dev_x, dev_y), (test_x, test_y) = load_data(hyperparams=hyperparams)

    np.shape(train_x)
    np.shape(train_y)
    np.shape(dev_x)
    np.shape(dev_y)
    np.shape(test_x)
    np.shape(test_y)
