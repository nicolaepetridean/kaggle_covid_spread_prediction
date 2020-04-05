from python_src.indexing.write_to_elk import elk_connect, check_mapping_exists, write_data_from_dataframe
from multiprocessing import Pool
import python_src.runtime.config as config
import python_src.indexing.read_external_data as external_data_provider
import os

def index_file(file_metadata):
    elk_client = elk_connect()
    config_set = config.read_yml_config_file('runtime\\run_local.yaml')
    check_mapping_exists(config_set['es_index'], elk_client)
    write_data_from_dataframe(dataframe=eval_df, metadata=file_metadata, es_client=elk_client)


if __name__ == '__main__':
    config_set = config.read_yml_config_file('runtime\\run_local.yaml')
    files = external_data_provider.get_files_for_indexing(os.getcwd() + config_set['external_data_dir'])

    workers_pool = Pool(1)
    workers_pool.map(index_file, files)
