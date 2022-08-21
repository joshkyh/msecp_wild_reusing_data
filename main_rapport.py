import util.researchdb as researchdb
from hnatt import HNATT
import os as os
from util.model_logging import insert_model_log, insert_complete_run
from util.runsettings import get_runsettings, write_runsettings



if __name__ == '__main__':

    file_paths, hyperparams = get_runsettings()
    write_runsettings(hyperparams)
    (train_x, train_y), (dev_x, dev_y) = researchdb.load_data()
    # log hyperparams
    insert_model_log(hyperparams)

    # initialize HNATT 
    if not os.path.exists(file_paths["expt_model_dir"]): 
        os.makedirs(file_paths["expt_model_dir"])
    
    h = HNATT()    
    h.train(train_x, train_y, dev_x, dev_y,file_paths, hyperparams)

    
    # Training complete - log completion
    insert_complete_run(hyperparams)
    
