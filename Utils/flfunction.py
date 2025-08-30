import os
import pickle
import numpy as np
import random
import simpy
import pandas as pd
import networkx as nx
from datetime import datetime
import time
import torch
from Utils.logger import Logger
from configure import *
from globalvar import *
from Utils.utilsfunction import *

def generate_test_data(num_samples, include_not_avail=False):
    data = []
    queue_values = np.arange(0, 11)  # Possible queue values from 0 to 10
    # Set probabilities: 0 at 35%, 10 at 20%, and 5% each for values 1-9
    queue_probs = [0.35] + [0.05] * 9 + [0.20]

    for _ in range(num_samples):
        sample = []
        if diff_lastHop:
            sample.append(random.randint(0, 4))
        # Queue Scores for each direction: Up, Down, Right, Left (4 scores each)
        for _ in range(4):
            # Queue scores biased towards 0 and 10
            sample.extend(np.random.choice(queue_values, 4, p=queue_probs))
            
            # Relative positions for each direction: latitude and longitude
            sample.append(np.random.uniform(-2, 2))  # Latitude relative position
            sample.append(np.random.uniform(-2, 2))  # Longitude relative position
        
        # Absolute positions
        sample.append(np.random.uniform(0, 9))  # Absolute latitude normalized
        sample.append(np.random.uniform(0, 18))  # Absolute longitude normalized
        
        # Destination differential coordinates
        sample.append(np.random.uniform(-2, 2))  # Destination differential latitude
        sample.append(np.random.uniform(-2, 2))  # Destination differential longitude
        
        # Optionally include not available values
        if include_not_avail and np.random.rand() < 0.1:  # 10% chance to introduce a -1 value
            idx_to_replace = np.random.choice(len(sample), int(0.1 * len(sample)), replace=False)
            sample[idx_to_replace] = -1
        
        data.append(sample)
    
    return np.array(data)


def get_models(earth):
    models = []
    model_names = []
    for plane in earth.LEO:
        for sat in plane.sats:
            models.append(sat.DDQNA.qNetwork)
            model_names.append(sat.ID)
    return models, model_names


def average_model_weights(models): # return a dict
    """
    Average the parameters of a list of PyTorch models (same architecture).
    Returns a new state_dict with averaged parameters.
    """
    avg_state_dict = {}
    n_models = len(models)

    # Convert all state_dicts to list of named parameters
    state_dicts = [model.state_dict() for model in models]

    for key in state_dicts[0].keys():
        # Stack and average each parameter tensor
        stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
        avg_state_dict[key] = torch.mean(stacked, dim=0)

    return avg_state_dict # 

def full_federated_learning(models):
    """
    Perform federated averaging across a list of PyTorch models with identical architecture.
    Updates each model in-place with the averaged parameters.
    """
    # 提取并平均每一层的参数
    state_dicts = [model.state_dict() for model in models]
    averaged_state_dict = {}

    for key in state_dicts[0].keys():
        stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
        averaged_state_dict[key] = torch.mean(stacked, dim=0)

    # 将平均权重分发回每个模型
    for model in models:
        model.load_state_dict(averaged_state_dict)

def federate_by_plane(models, model_names):
    """
    Perform Federated Averaging within each orbital plane.
    models: list of PyTorch models (e.g., qNetwork or qTarget)
    model_names: list of names like 'plane0_sat1', 'plane1_sat3', ...
    """
    from collections import defaultdict

    # 分组：每个 plane -> 模型列表
    plane_dict = defaultdict(list)
    for model, name in zip(models, model_names):
        plane = name.split('_')[0]
        plane_dict[plane].append(model)

    # 每组模型执行联邦平均
    for plane_models in plane_dict.values():
        state_dicts = [m.state_dict() for m in plane_models]
        averaged_state_dict = {}

        for key in state_dicts[0].keys():
            stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
            averaged_state_dict[key] = torch.mean(stacked, dim=0)

        # 将平均参数加载回每个模型
        for model in plane_models:
            model.load_state_dict(averaged_state_dict)

def model_anticipation_federate(models, model_names):
    """
    Perform Model Anticipation Federated Learning using PyTorch models.
    Each satellite (after the first) in a plane averages its model with its predecessor's.
    """
    from collections import defaultdict

    # Step 1: group models by orbital plane
    plane_dict = defaultdict(list)
    for model, name in zip(models, model_names):
        plane = name.split('_')[0]
        plane_dict[plane].append((model, name))

    # Step 2: process each plane separately
    for plane_models in plane_dict.values():
        # Sort satellites within the plane by their numeric ID
        plane_models.sort(key=lambda x: int(x[1].split('_')[1]))  # name format: planeX_satY

        # Step 3: from second satellite onward, average with previous one
        for i in range(1, len(plane_models)):
            prev_model = plane_models[i - 1][0]
            curr_model = plane_models[i][0]

            prev_state = prev_model.state_dict()
            curr_state = curr_model.state_dict()

            averaged_state = {}
            for key in curr_state.keys():
                averaged_state[key] = (curr_state[key].float() + prev_state[key].float()) / 2.0

            curr_model.load_state_dict(averaged_state)

def update_sats_models(earth, models, model_names):
    '''Update each satellite model for the updated one'''
    print('Updating satellites models...')
    for model, satID in zip(models, model_names):
        sat = findByID(earth, satID)
        sat.DDQNA.qNetwork = model
        if ddqn:
            sat.DDQNA.qTarget = model

# def compute_full_cka_matrix(models, data):
#     pass

# def compute_average_cka(cka_matrix):

def perform_FL(earth):#, outputPath):

    # path = outputPath + 'FL' + str(len(earth.gateways)) + 'GTs/'
    # os.makedirs(path, exist_ok=True) 
    print('----------------------------------')
    print(f'Federated Learning. Performing: {FL_tech}')

    data = generate_test_data(num_samples, include_not_avail=False)
    models, model_names = get_models(earth)

    # CKA_Values_before = compute_full_cka_matrix(models, data)

    if FL_tech == 'nothing':
        # return CKA_Values_before, CKA_Values_before
        return
        
    if FL_tech == 'modelAnticipation':
        model_anticipation_federate(models, model_names)
    elif FL_tech == 'plane':
        federate_by_plane(models, model_names)
    elif FL_tech == 'full':
        full_federated_learning(models)
    elif FL_tech == 'combination':
        global FL_counter
        if FL_counter == 1:
            print(f'Model Anticipation, counter = {FL_counter}')
            FL_counter += 1
            model_anticipation_federate(models, model_names)

        elif FL_counter == 2:
            print(f'Plane FL, counter = {FL_counter}')
            FL_counter += 1
            federate_by_plane(models, model_names)

        elif FL_counter > 2:
            print(f'Global FL, counter = {FL_counter}')
            FL_counter = 1
            full_federated_learning(models)

    # CKA_Values_after = compute_full_cka_matrix(models, data)
    update_sats_models(earth, models, model_names)

    print('----------------------------------')
    # return CKA_Values_before, CKA_Values_after

