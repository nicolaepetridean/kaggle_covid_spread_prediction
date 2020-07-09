import yaml


def read_yml_config_file(conf_file):
    config = {}
    with open(conf_file) as f:
        docs = yaml.load_all(f, Loader=yaml.FullLoader)
        for doc in docs:
            for k, v in doc.items():
                if k == 'training':
                    config['train_data'] = v['input_train_rel_path']
                    config['test_data'] = v['input_test_rel_path']
                    config['should_ensemble'] = v['ensemble']
                    config['algorithms_to_train'] = v['algorithms']
                if k == 'external_data_dir':
                    config['external_data_dir'] = v['rel_paths']
                if k == 'inference':
                    config['inference_output_path'] = v['output_template_rel_path']
                if k == 'indexing':
                    config['indexing_server'] = v['server']
                    config['indexing_server_port'] = v['port']
                    config['cntm_measures_es_index'] = v['cntm_measures_es_index']
                    config['country_grouping_es_index'] = v['country_grouping_es_index']
                    config['confirmed_prediction_index'] = v['confirmed_prediction_index']

    return config


if __name__ == '__main__':
    files_to_index = read_yml_config_file()