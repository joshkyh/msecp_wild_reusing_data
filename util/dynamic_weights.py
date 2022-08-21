



if __name__ == '__main__':
    import util.runsettings as RS
    import util.config as config
    import util.researchdb as researchdb


    PROJECT_PATH = config.get_project_path()

    hyperparams = RS.random_runsettings(input_config='vpa', dataset_config='i', y_config='i', multilabel_aux='aphf',
                                     aux_weights_assignment='random', aux_hierarchy_config='rock', num_gru_config=2,
                                     num_translator='None',dynamic_loss_weights='y', code_development='y')

    (train_x, train_y), (dev_x, dev_y), (test_x, test_y) = researchdb.load_data(hyperparams=hyperparams)
