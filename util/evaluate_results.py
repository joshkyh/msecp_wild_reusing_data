import util.researchdb as researchdb

import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from sklearn.metrics import *
from scipy.stats import pearsonr, describe
import math

import os
import pandas as pd

import time


import util.config as config
import keras
import numpy as np
import logging


def get_classwise_accuracy(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    result = matrix.diagonal()/matrix.sum(axis=1)
    return result


def get_performance_df(epoch,h,hyperparams, encoded_x, y):
    if hyperparams['y_config']!='c':
        y_score = h.predict(encoded_x)
        y_score = y_score[0]  # we are always multilabel, take the first column
        mae = mean_absolute_error(y_true=y, y_pred=y_score)
        rmse = mean_squared_error(y_true=y, y_pred=y_score) ** 0.5
        r_sq = r2_score(y_true=y, y_pred=y_score)
        var = describe(y_score).variance[0]

        flat_pred = y_score.flatten()
        flat_actual = y.flatten()
        r, prob = pearsonr(x=flat_pred, y=flat_actual)
        if math.isnan(r):
            if np.std(flat_pred) == 0:
                print('Predictions constant')
            if np.std(flat_actual) == 0:
                print('Actuals constant')
            r = -1.0

        df = pd.DataFrame([
            {'key': 'mae', 'value': mae},
            {'key': 'rmse', 'value': rmse},
            {'key': 'r_sq', 'value': r_sq},
            {'key': 'r', 'value': r},
            {'key': 'var', 'value': var}
        ])


    if hyperparams['y_config']=='c':
        y_prob = h.predict(encoded_x)
        y_prob = np.array(y_prob[0])
        y_score = y_prob.argmax(axis=-1)


        # Need cross-entropy loss for median stopping rule
        cross_entropy_loss = log_loss(y_true=y, y_pred=y_prob)

        # For reporting in paper
        multiclass_unweighted_acc = accuracy_score(y_true=y, y_pred=y_score)
        multiclass_weighted_acc = balanced_accuracy_score(y_true=y, y_pred=y_score)

        multiclass_unweighted_f1 = f1_score(y_true=y, y_pred=y_score, average='macro')
        multiclass_weighted_f1 = f1_score(y_true=y, y_pred=y_score, average='weighted')

        class_report = classification_report(y_true=y, y_pred=y_score, output_dict=True)

        if hyperparams['dataset_config'] == 'i':
            classwise_accuracy = get_classwise_accuracy(y_true=y, y_pred=y_score)
            # ang 0, hap_exc 1, neu 2, sad 3
            ang_acc = classwise_accuracy[0]
            h_e_acc = classwise_accuracy[1]
            neu_acc = classwise_accuracy[2]
            sad_acc = classwise_accuracy[3]

            ang_f1 = class_report['0.0']['f1-score']
            h_e_f1 = class_report['1.0']['f1-score']
            neu_f1 = class_report['2.0']['f1-score']
            sad_f1 = class_report['3.0']['f1-score']

            df = pd.DataFrame([
                {'key': 'cel', 'value': cross_entropy_loss},
                {'key': 'multiclass_unweighted_acc', 'value': multiclass_unweighted_acc},
                {'key': 'multiclass_weighted_acc', 'value': multiclass_weighted_acc},
                {'key': 'multiclass_unweighted_f1', 'value': multiclass_unweighted_f1},
                {'key': 'multiclass_weighted_f1', 'value': multiclass_weighted_f1},
                {'key': 'ang_acc', 'value': ang_acc},
                {'key': 'h_e_acc', 'value': h_e_acc},
                {'key': 'neu_acc', 'value': neu_acc},
                {'key': 'sad_acc', 'value': sad_acc},
                {'key': 'ang_f1', 'value': ang_f1},
                {'key': 'h_e_f1', 'value': h_e_f1},
                {'key': 'neu_f1', 'value': neu_f1},
                {'key': 'sad_f1', 'value': sad_f1}
            ])



    df['Model_Unixtime'] = int(hyperparams['Model_Unixtime'])
    df['Epoch'] = int(epoch)

    df = df[['Model_Unixtime', 'Epoch', 'key', 'value']]
    #print(df)  # disable printing so that we can see which dev fold number is this

    return df

def get_median_running_avg(model_unixtime, metric_name, epoch):
    while True:
        try:
            conn = config.get_connection()
            #model_unixtime = 1595992858941806
            #metric_name =  'mae'
            #epoch =

            # Since we are getting median, it doesn't matter whether we are maximising or minising the metric
            # because the median is the same
            query = f'SET NOCOUNT ON; EXEC iemocap.get_median_running_average_mhdict {model_unixtime}, {metric_name}, {epoch}'
            #print(query)
            df = pd.read_sql_query(query, conn)
            median_running_avg = df.iloc[0][0]
            return median_running_avg
        except:
            logging.exception('pyodbc error, try again.')
            time.sleep(60)
            continue



def get_best_dev_mae(model_unixtime):
    while True:
        try:
            conn = config.get_connection()

            query = f'''
            SELECT MIN(Value) AS best_dev_mae
            FROM iemocap.Model_Epoch_Metrics
            WHERE dev_test = 'dev'
            AND [key] = 'mae'
            AND Model_Unixtime = {model_unixtime}
            '''

            df = pd.read_sql_query(query, conn)
            best_dev_mae = df.iloc[0][0]
            return best_dev_mae

        except:
            logging.exception('pyodbc error, try again.')
            time.sleep(60)
            continue

def get_best_dev_cross_entropy(model_unixtime):
    while True:
        try:
            conn = config.get_connection()

            query = f'''
            SELECT MIN(Value) AS best_dev_cross_entropy_loss
            FROM iemocap.Model_Epoch_Metrics
            WHERE dev_test = 'dev'
            AND [key] = 'cel'
            AND Model_Unixtime = {model_unixtime}
            '''

            df = pd.read_sql_query(query, conn)
            best_dev_cross_entropy_loss = df.iloc[0][0]
            return best_dev_cross_entropy_loss

        except:
            logging.exception('pyodbc error, try again.')
            time.sleep(60)
            continue


def get_stop_training(epoch, hyperparams):
    # Should we continue training? The median stopping policy below.
    stop_training = False
    # delay evaluation till 5th epoch, evaluate at every 2 epoch thereafter (balance database load with Hyperparameter tuning).

    delay_evaluation = 5
    check_every = 2

    if hyperparams['code_development'] == 'y':
        delay_evaluation = 1
        check_every = 1

    if epoch >= delay_evaluation and epoch % check_every == 0:
        if hyperparams['y_config'] != 'c':
            median_running_avg = get_median_running_avg(model_unixtime=hyperparams['Model_Unixtime'],
                                                        metric_name='mae',
                                                        epoch=epoch)

            # If this is the first run, median would equal to curent best, don't want to stop training
            # Stop training only if current_best > median
            best_dev_mae = get_best_dev_mae(model_unixtime=hyperparams['Model_Unixtime'])
            if best_dev_mae > median_running_avg:
                print("Median Stopping Policy in Effect")
                print(f"Current Best Dev MAE: {best_dev_mae}", f"Median Running Average: {median_running_avg}")
                stop_training = True

        if hyperparams['y_config'] == 'c':
            median_running_avg = get_median_running_avg(model_unixtime=hyperparams['Model_Unixtime'],
                                                        metric_name='cel',
                                                        epoch=epoch)
            # If this is the first run, median would equal to curent best, don't want to stop training
            # Stop training only if current_best > median
            best_dev_cross_entropy = get_best_dev_cross_entropy(model_unixtime=hyperparams['Model_Unixtime'])
            if best_dev_cross_entropy > median_running_avg:
                print("Median Stopping Policy in Effect")
                print(f"Current Best Dev CEL: {best_dev_cross_entropy}", f"Median Running Average: {median_running_avg}")
                stop_training = True
    return stop_training

def log_epoch_end_performances(epoch, h, hyperparams, encoded_dev_x, dev_y, encoded_test_x, test_y):

    if hyperparams['code_development'] == 'n':
        # PRODUCTION environment
        dev_df = get_performance_df(epoch,h,hyperparams, encoded_x=encoded_dev_x, y=dev_y)
        dev_df['dev_test'] = 'dev'
        print('dev_df', dev_df)
        dev_data = dev_df.values.tolist()

        test_df = get_performance_df(epoch, h, hyperparams, encoded_x=encoded_test_x, y=test_y)
        test_df['dev_test'] = 'test'
        print('test_df', test_df)
        test_data = test_df.values.tolist()

        #print(data)

        while True:
            try:
                # do stuff
                cnxn = config.get_connection()
                cursor = cnxn.cursor()
                sql = "INSERT INTO IEMOCAP.Model_Epoch_Metrics (Model_Unixtime, epoch,[key],[value],dev_test) VALUES (?,?,?,?,?)"
                cursor.executemany(sql,dev_data)
                cnxn.commit()
                cnxn.close()

                cnxn = config.get_connection()
                cursor = cnxn.cursor()
                sql = "INSERT INTO IEMOCAP.Model_Epoch_Metrics (Model_Unixtime, epoch,[key],[value],dev_test) VALUES (?,?,?,?,?)"
                cursor.executemany(sql, test_data)
                cnxn.commit()
                cnxn.close()

            except:
                logging.exception('pyodbc error, try again.')
                time.sleep(60)
                continue

            break

        stop_training = get_stop_training(epoch, hyperparams)

        return stop_training

    else:
        # Test Environment
        stop_training = False
        return stop_training



class LossHistory(keras.callbacks.Callback):
    def __init__(self, hyperparams, encoded_train_x, train_y, encoded_test_x, test_y,
                 aux_mainweight,
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
                 ):

        self.validation_data = None
        self.model = None
        self.hyperparams = hyperparams

        self.encoded_train_x = encoded_train_x
        self.train_y = train_y

        self.encoded_test_x = encoded_test_x
        self.test_y = test_y

        self.aux_mainweight = aux_mainweight,
        self.weights_prosody_tone_happiness = weights_prosody_tone_happiness,
        self.weights_prosody_tone_sadness = weights_prosody_tone_sadness,
        self.weights_prosody_tone_anger = weights_prosody_tone_anger,
        self.weights_prosody_tone_fear = weights_prosody_tone_fear,
        self.weights_actions_au05 = weights_actions_au05,
        self.weights_actions_au17 = weights_actions_au17,
        self.weights_actions_au20 = weights_actions_au20,
        self.weights_actions_au25 = weights_actions_au25,
        self.weights_y_lag1 = weights_y_lag1,
        self.weights_y_lag2 = weights_y_lag2,
        self.weights_y_lag3 = weights_y_lag3,
        self.weights_y_lag4 = weights_y_lag4,
        self.weights_y_lead1 = weights_y_lead1,
        self.weights_y_lead2 = weights_y_lead2,
        self.weights_y_lead3 = weights_y_lead3,
        self.weights_y_lead4 = weights_y_lead4


    def on_epoch_end(self, epoch, logs=None):

        stop_training = log_epoch_end_performances(epoch=epoch, h=self.model,
                                   hyperparams=self.hyperparams,

                                   encoded_dev_x=self.validation_data[0], dev_y=self.validation_data[1],
                                   encoded_test_x=self.encoded_test_x, test_y=self.test_y)
        if stop_training:
            self.model.stop_training = True


if __name__ == '__main__':
    pass