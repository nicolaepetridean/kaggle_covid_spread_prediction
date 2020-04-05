from python_src.indexing.write_to_elk import elk_connect, check_mapping_exists, write_data_from_dataframe
from multiprocessing import Pool
import python_src.runtime.config as config
import python_src.indexing.read_external_data as external_data_provider
import pandas as pd
import numpy as np


def index_file(file_metadata):
    config_set = config.read_yml_config_file('runtime\\run_local.yaml')
    elk_client = elk_connect(config_set)
    check_mapping_exists(config_set['es_index'], elk_client)
    data = pd.read_csv(file_metadata['file_path'])

    # available data in the 'data' csv is : (just in case we need other columns
    # ['ID', 'Applies To', 'Country', 'Date Start', 'Date end intended',
    #  'Description of measure implemented', 'Exceptions', 'Implementing City',
    #  'Implementing State/Province', 'Keywords', 'Quantity', 'Source',
    #  'Target city', 'Target country', 'Target region', 'Target state']

    # select for now several columns
    data_to_write = data[['Keywords', 'Applies To', 'Country', 'Date Start', 'Date end intended', 'Description of measure implemented']]
    data_to_write = data_to_write.rename(columns={"Applies To": "applies_to", "Date Start": "start_date",
                                  "Date end intended": "intended_end_date", "Description of measure implemented": "measures",
                                  "Keywords": "keywords", "Country": "country"})

    #  dummy post process data for now. TODO: investigate what is needed @Tobias
    data_to_write['applies_to'] = data_to_write['applies_to'].replace(np.nan, '')
    data_to_write['keywords'] = data_to_write['keywords'].replace(np.nan, '')
    data_to_write['intended_end_date'] = data_to_write['intended_end_date'].replace(np.nan, '')
    data_to_write['start_date'] = data_to_write['start_date'].replace(np.nan, '')
    data_to_write['country'] = data_to_write['country'].replace(np.nan, '')
    data_to_write['measures'] = data_to_write['measures'].replace(np.nan, '')

    write_data_from_dataframe(dataframe=data_to_write, es_index=config_set['es_index'], es_client=elk_client)


if __name__ == '__main__':
    config_set = config.read_yml_config_file('runtime\\run_local.yaml')
    files = external_data_provider.get_files_for_indexing(config_set['external_data_dir'])

    workers_pool = Pool(1)
    workers_pool.map(index_file, files)
