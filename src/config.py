import os
from configparser import SafeConfigParser

def load_config(file_path):
    parser = SafeConfigParser()
    parser.read(file_path)
    config = {
        'service': parser.get('application_plane', 'Endpoint'),
        'period': int(parser.get('application_plane', 'Period')),
        'span_states_file_txt': parser.get('application_plane', 'SpanStatesFileTxt'),
        'app': parser.get('application_plane', 'Application'),
        'elim_percentile': int(parser.get('application_plane', 'EliminationPercentile')),
        'epsilon': int(parser.get('application_plane', 'Epsilon')),
        'result_dir': "results",
        'reward_field': "Var_sum",
        'confidence': 0.95,
        'aws_access_key_id': parser.get('application_plane', 'aws_access_key_id'),
        'aws_secret_access_key': parser.get('application_plane', 'aws_secret_access_key'),
        'bucket_name': parser.get('application_plane', 'bucket_name'),
        's3_key': parser.get('application_plane', 's3_key'),
        's3_url': parser.get('application_plane', 's3_url')
    }
    return config
