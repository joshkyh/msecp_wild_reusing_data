'''
Adhoc one off to calculate MI and insert into database manually
'''
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelBinarizer

import util.config as config
import util.runsettings as RS
import util.researchdb as researchdb


if __name__ == "__main__":

    PROJECT_PATH = config.get_project_path()
    hyperparams = RS.random_runsettings(input_config='vpa',
                                        # dataset_config='i',y_config='a',
                                        dataset_config='s', y_config='v',
                                        multilabel_aux='None',
                                        aux_weights_assignment='None',
                                        num_gru_config=20,
                                        num_translator='None',
                                        aux_hierarchy_config='rock',
                                        dynamic_loss_weights='n',
                                        code_development='y')

    (train_x, train_y), (dev_x, dev_y), (test_x, test_y) = researchdb.load_data(hyperparams=hyperparams)
    train_y_cp = train_y[~np.isnan(train_y).any(axis=1)]

    if hyperparams['y_config'] == 'c':
        aux = train_y_cp[:, 9]
        y = np.char.mod('%d', train_y[:, 0]).astype('object')
        lb = LabelBinarizer()
        lb.fit(aux)
        aux = lb.transform(aux)
        mi = mutual_info_classif(y=y, X=aux, discrete_features=True)
        print(np.average(mi))
    else:
        aux = train_y_cp[:, 1:17]
        y = train_y_cp[:,0]
        y.shape
        aux.shape
        np.where(np.isnan(y))
        np.where(np.isnan(aux))



        mi = mutual_info_regression(y=y, X=aux)
        np.set_printoptions(suppress=True)
        print(np.round(mi,4))

