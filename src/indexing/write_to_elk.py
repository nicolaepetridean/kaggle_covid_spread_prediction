import elasticsearch
from elasticsearch import helpers


def create_data_structure(elk_client, index_name):
	"""create necessary data structures to write data to elastic search"""

	print("creating 'logger_data_bulk' index...")
	try:
		elk_client.indices.create(index=index_name)
	except Exception as exception:
		print("exception while creating + " + exception)
		return False

	return True


def elk_connect():
	# configure elasticsearch
	config = {
		'host': '10.38.4.75'
	}
	return elasticsearch.Elasticsearch([config, ], timeout=300)


def write_data_from_dataframe(dataframe, metadata, es_client):
	print('pushing {} to elasticsearch'.format(metadata['file_id']))
	def generator():
		for idx, row in dataframe.iterrows():
			doc = row.to_dict()
			doc['trip'] = metadata['trip']
			yield doc
	helpers.bulk(es_client, generator(), index=metadata['es_index'])


def check_mapping_exists(index_name, es_client):
	"""
	Check if the existing connection to an elastic search server contains a specific mapping
	:param es_client : client connection to an running elastic search server
	:param index_name : index name to search for.
	:return : True/False depending on the index being found
	"""

	try:
		return es_client.indices.get_mapping(index=index_name) is not None
	except:
		return False
