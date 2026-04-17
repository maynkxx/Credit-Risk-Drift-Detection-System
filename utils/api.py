import os
import requests
import streamlit as st
import time
API_BASE_URL = os.getenv('API_BASE_URL', 'https://credit-risk-api-scq0.onrender.com')

def get_health():
    start_time = time.time()
    try:
        response = requests.get(f'{API_BASE_URL}/health', timeout=20)
        delay = time.time() - start_time
        response.raise_for_status()
        data = response.json()
        data['delay_seconds'] = delay
        return data
    except requests.exceptions.RequestException as e:
        delay = time.time() - start_time
        return {'status': 'error', 'message': str(e), 'delay_seconds': delay}

def get_model_info():
    try:
        response = requests.get(f'{API_BASE_URL}/model/info', timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {'status': 'error', 'message': str(e)}

def predict_single(data: dict):
    try:
        response = requests.post(f'{API_BASE_URL}/predict', json=data, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {'status': 'error', 'message': str(e)}

def predict_batch(file):
    try:
        files = {'file': (file.name, file, 'text/csv')}
        response = requests.post(f'{API_BASE_URL}/predict/batch', files=files, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {'status': 'error', 'message': str(e)}