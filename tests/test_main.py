import os
import io
import time
import json
import socket
import base64
import psutil
import shutil
import sqlite3
import tempfile
import requests
import subprocess
import numpy as np
import pandas as pd


from calc_metrics import write_img_return_base64
from db import empty_db, execute_query, execute_read_query, create_connection


def launch_server():
    """
    Launches app.py and returns psutil process object
    """
    process = subprocess.Popen(
        ['python3', 'app.py'], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
    )
    process.daemon = True
    return psutil.Process(process.pid)

def shutdown_server(psutil_process):
    """
    Shutodowns the server
    """
    try:
        children = psutil_process.children(recursive=True)
        for child_proc in children:
            try:
                child_proc.kill()
            except:
                pass

        try:
            psutil_process.kill()
        except:
                pass

    except psutil.NoSuchProcess:
        pass

def wait_until_server_responds(psutil_process=None, host='127.0.0.1', port=5000):
    """
    Waits until the server responds.
    If the psutil_process is passed, it is checked by is_running
    """
    while socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect_ex((host, port)) != 0:
        time.sleep(0.1)
        if psutil_process:
            assert psutil_process.is_running()
    print('Server responded')

def add_test_servce_into_db(id, name, token):
    with create_connection() as conn:
        execute_query(conn, f'INSERT INTO services (id, name, token) VALUES ("{id}", "{name}", "{token}");')

def create_test_files_pack(dataset, n):
    tmp_dir = os.path.join('datasets', dataset)
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    tmp_files = []
    for i in range(n):
        f_name = f'file_{i}.zip'
        tmp_files.append(f_name)
        f_path = os.path.join(tmp_dir, f_name)
        with open(file=f_path, mode='w') as f:
            f.write(f'file content {i}')
        
    return tmp_dir, tmp_files

def add_test_dataset_into_db(title='test_dataset'):
    tmp_dir, tmp_files = create_test_files_pack(title, 10)
    with create_connection() as conn:
        for i, f in enumerate(tmp_files):
            execute_query(conn, f'INSERT INTO datasets (id, title, filename) VALUES ("{i}", "{title}", "{f}");')
    return tmp_dir, tmp_files


def auth(base_url, name, token, dataset):
    """
    returns dict with keys:
        session_token,
        expire_in,
        number_of_items
    """
    data = {
        "name": name,
        "token": token,
        "dataset": dataset,
    }
    return requests.post(base_url + '/auth', json=data).json()

def pull(base_url, name, session_token):
    data = {
        "name": name,
        "session_token": session_token,
    }
    return requests.post(base_url + '/pull', json=data)

def push(base_url, name, session_token, testing_item_id):
    data = {
        'name': name,
        'session_token': session_token,
        'testing_item_id': testing_item_id,
        'ct': 0.0,
        'left': {
            'affected_part': 0.0,
            'total_volume': 0.0,
            'affected_volume': 0.0,
        },
        'right': {
            'affected_part': 0.0,
            'total_volume': 0.0,
            'affected_volume': 0.0,
        },
        'viewer_url': 'localhost',
        'description': 'testing',
    }
    return requests.post(base_url + '/push', json=data).json()
    
def push_mask(base_url, return_type='json', input_type='list'):
    mask_1 = np.zeros((100, 100)).astype(int)
    mask_2 = mask_1
    data = {
        'return_type': return_type,
        'input_type': input_type,
    }
    if input_type == 'list':
        data['mask_1'] = mask_1.tolist()
        data['mask_2'] = mask_2.tolist()

    elif input_type == 'base64':
        data['mask_1'] = write_img_return_base64(mask_1)
        data['mask_2'] = write_img_return_base64(mask_2)

    return requests.post(base_url + '/push_mask', json=data)


class Test_main:
    """
    Class to test launchability of the application
    """
    def setup_class(self):
        # This method is triggered before all
        self.host = '127.0.0.1'
        self.port = 5000
        self.base_url = f'http://{self.host}:{self.port}'
        self.server_process = launch_server()
        wait_until_server_responds(self.server_process)
        empty_db()
        self.name, self.token, self.dataset = 'test_service', '123qwerty654', 'test_dataset'
        self.tmp_dir, self.tmp_files = add_test_dataset_into_db(self.dataset)

    def teardown_class(self):
        # This method is triggered after all is done
        shutdown_server(self.server_process)
        empty_db()
        shutil.rmtree(self.tmp_dir)

    def test_root_route(self):
        # testing /
        responce = requests.get(self.base_url)
        assert responce.status_code == 200
        assert isinstance(responce.json()['message'], str)


    def test_auth_true(self):
        # testing /auth with correct data
        add_test_servce_into_db(0, self.name, self.token)
        responce = auth(self.base_url, self.name, self.token, self.dataset)
        assert isinstance(responce.get('session_token'), str)
        assert 'expire_in' in responce
        assert isinstance(responce.get('number_of_items'), int)
        assert responce['number_of_items'] == len(self.tmp_files)

    def test_auth_false_1(self):
        # testing /auth with invalid name
        responce = auth(self.base_url, self.name+'123123', self.token, self.dataset)
        assert isinstance(responce.get('error_message'), str)
        assert '401' in responce['error_message']

    def test_auth_false_2(self):
        # testing /auth with invalid token
        responce = auth(self.base_url, self.name, self.token+'123123', self.dataset)
        assert isinstance(responce.get('error_message'), str)
        assert '401' in responce['error_message']


    def test_pull_true_1(self):
        # testing single request /pull
        auth_responce = auth(self.base_url, self.name, self.token, self.dataset)
        assert 'session_token' in auth_responce
        pull_responce = pull(self.base_url, self.name, auth_responce['session_token'])
        assert 'file content' in pull_responce.content.decode('utf-8')

    def test_pull_false_1(self):
        # testing /pull with invalid name
        auth_responce = auth(self.base_url, self.name, self.token, self.dataset)
        assert 'session_token' in auth_responce
        pull_responce = pull(self.base_url, self.name+'123', auth_responce['session_token'])
        assert '401' in pull_responce.json()['error_message']

    def test_pull_false_2(self):
        # testing /pull with session_token
        auth_responce = auth(self.base_url, self.name, self.token, self.dataset)
        assert 'session_token' in auth_responce
        pull_responce = pull(self.base_url, self.name, auth_responce['session_token']+'123')
        assert '401' in pull_responce.json()['error_message']

    def test_pull_true_2(self):
        # testing /pull & /push
        auth_responce = auth(self.base_url, self.name, self.token, self.dataset)
        assert 'session_token' in auth_responce

        for _ in range(len(self.tmp_files)):
            pull_responce = pull(self.base_url, self.name, auth_responce['session_token'])
            assert 'file content' in pull_responce.content.decode('utf-8')
            push_responce = push(self.base_url, self.name, auth_responce['session_token'], pull_responce.headers['testing_item_id'])
            assert 'message' in push_responce and 'time_to_response' in push_responce

        pull_responce = pull(self.base_url, self.name, auth_responce['session_token'])
        assert 'No available items for this session_token' in pull_responce.json()['error_message']

    def test_push_mask_json_send_list(self):
        responce = push_mask(self.base_url, return_type='json', input_type='list').json()
        assert 'metrics' in responce
        assert isinstance(responce['metrics'], dict)

    def test_push_mask_csv_send_list(self):
        responce = push_mask(self.base_url, return_type='csv', input_type='list')
        assert 'filename' in responce.headers
        assert 'metrics.csv' == responce.headers['filename']
        assert len(responce.content) > 0
        df = pd.read_csv(io.BytesIO(responce.content), encoding='utf-8')
        assert df.shape[1] > 1

    def test_push_mask_json_send_base64(self):
        responce = push_mask(self.base_url, return_type='json', input_type='base64').json()
        assert 'metrics' in responce
        assert isinstance(responce['metrics'], dict)

    def test_push_mask_csv_send_base64(self):
        responce = push_mask(self.base_url, return_type='csv', input_type='base64')
        assert 'filename' in responce.headers
        assert 'metrics.csv' == responce.headers['filename']
        assert len(responce.content) > 0
        df = pd.read_csv(io.BytesIO(responce.content), encoding='utf-8')
        assert df.shape[1] > 1
