# -*- coding: utf-8 -*-

import logging
import os
import time



import pandas as pd
import numpy as np
from tqdm import tqdm

from util.text_util import normalize
import util.config as config
#import mlflow

tqdm.pandas()

def insert_model_log(input_dict):
    '''
    Insert many rows in a dictionary format. Because the number of hyperparameters is changing in the course of research.
    It is not practical to fix the number of columns in the target table.
    '''

    Model_Unixtime = input_dict['Model_Unixtime']
    export_dict = input_dict.copy()
    export_dict.pop('Model_Unixtime')
    export_dict.pop('Optimizer_Dict')
    input_df = pd.DataFrame.from_dict(export_dict, orient='index')
    input_df.reset_index(drop=False, inplace=True)
    input_df.columns = ['HP_Name','HP_Value']

    input_df['Model_Unixtime'] = Model_Unixtime
    input_df = input_df[['Model_Unixtime','HP_Name','HP_Value']]

    data = input_df.values.tolist()

    # Export to DB - SWITCH the number of question marks
    logging.info("Exporting Results")
    cnxn = config.get_connection()
    cursor = cnxn.cursor()
    sql = "INSERT INTO IEMOCAP.Model_Hyperparameter_Dict VALUES (?,?,?)"
    cursor.executemany(sql,data)
            
    cnxn.commit()
    cnxn.close()

def insert_complete_run(input_dict):
    
    '''
    Insert a row into the database notifying the completion of a run
    '''
    export = pd.Series()
    
    export['Model_Unixtime']  = input_dict['Model_Unixtime']
    export['fivefold_name']   = input_dict['fullrun_description']
    
    data = export.values.tolist()
    data = [data]
    
    logging.info("Exporting Completed Run")
    cnxn = config.get_connection()
    cursor = cnxn.cursor()
    sql = "INSERT INTO IEMOCAP.Model_Hyperparameter_Fullrun VALUES (?,?)"
    cursor.executemany(sql,data)
    cnxn.commit()
    cnxn.close()


def update_model_hiplot():
    '''
    model_hiplot is a SQL table that visualizes the 5-fold results using hiplot
    :return: none
    '''
    conn = config.get_connection()

    query = 'SET NOCOUNT ON; EXEC iemocap.update_model_hiplot'
    conn.execute(query)
    conn.commit()

def insert_model_params_count(hyperparams, trainable_count, non_trainable_count):
    Model_Unixtime = hyperparams['Model_Unixtime']
    total_count = trainable_count + non_trainable_count

    export = pd.Series()

    export['Model_Unixtime'] = Model_Unixtime
    export['total_count'] = total_count
    export['trainable_count'] = trainable_count
    export['non_trainable_count'] = non_trainable_count

    data = export.values.tolist()
    data = [data]

    logging.info("Exporting Model Params Count")
    cnxn = config.get_connection()
    cursor = cnxn.cursor()
    sql = "INSERT INTO IEMOCAP.Model_Params_Count VALUES (?,?,?,?)"
    cursor.executemany(sql, data)
    cnxn.commit()
    cnxn.close()



if __name__ == "__main__":

    #insert_model_log({})
    update_model_hiplot()
