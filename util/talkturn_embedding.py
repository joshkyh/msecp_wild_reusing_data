# -*- coding: utf-8 -*-
'''
Generating the equivalant of glove.6B.50d.txt
t1 0.1 0.2 0.3
t2 0.4 0.5 0.6
'''
import pandas as pd
import util.config as config
import os as os


if __name__ == '__main__':
    # wv_type = empath or vader
    wv_type = 'empath'
    
    if wv_type == 'empath':
        print('Downloading wv_type', wv_type)
        cnxn = config.get_connection()
        sql = 'SELECT E.* '
        sql += 'FROM dbo.talkturn_empath E WITH(NOLOCK) '
            
        print(sql)
            
        df = pd.read_sql(sql = sql, con = cnxn)
            
        cnxn.close()
        
        PROJECT_PATH = config.get_project_path()
        export_path = os.path.join(PROJECT_PATH, 'saved_models/empath.txt')
        
        df2 = df.copy()
        df2['talkturn_ID'] = 't' + df['talkturn_ID'].astype(str)
        
        df2 = df2.sort_values(by=['talkturn_ID'])
        df2.to_csv(export_path, sep=' ', header = False, index = False, doublequote = False)
        


