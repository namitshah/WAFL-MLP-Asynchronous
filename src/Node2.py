import json
import os
import socket, threading, time, pickle, traceback
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from net import Net 
from torch.utils.data.dataset import Subset

import socket, threading, time

HOST = "127.0.0.1"
PORT_START = 65430  
node_code = 2
n_device = 4
self_parameters = None
self_train_epoch = 50
max_epoch = 1000

def snd_parameters():
    PORTi = PORT_START + node_code
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORTi))
        s.listen()
        print("BOUND")
        while True:
            try:
                print("WAITING")
                conn, addr = s.accept()
                data = conn.recv(4096)
                if not data or data != b"MDLREQ":
                    conn.close()
                    continue
                print(str(data))
                conn.sendall(pickle.dumps(self_parameters))
                print("SENT")
                conn.close()
            except Exception as e:
                time.sleep(1)
                traceback.print_exc()
                print(str(e)[:50] + "...")
                print("The Exception occurred. Continuing on. (SND)")

def req_parameters(node_code):
    PORTo = PORT_START + node_code
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                print("REQUESTING")
                s.connect((HOST, PORTo))
                s.sendall(b"MDLREQ")
                data = b""
                while True:
                    packet = s.recv(1024)
                    if not packet:
                        break
                    data += packet
            data = pickle.loads(data)
            if data is None:
                raise Exception("RETRY")
            break
        except Exception as e:
            time.sleep(1)
            print(str(e)[:50] + "...")
            print("The Exception occurred. Continuing on. (REQ)")
    print("RECV")
    return data

if __name__ == "__main__":
    threading.Thread(target=snd_parameters, daemon=False, args=[]).start()
    experiment_case = "line"
    cp_filename = f"../data/contact_pattern/static_line_n10.json"
    # Hyperparameters
    batch_size = 32  # default 32
    learning_rate = 0.001  # default 0.001
    fl_coefficiency = 1.0  # defaulr 1.0  (WAFL's aggregation co efficiency)
    # Fix the seed
    torch.random.manual_seed(1)
    # Prepare the model folder (generated from the specified experiment_case)
    TRAINED_NET_PATH = f"../trained_net/{experiment_case}"
    if not os.path.exists(TRAINED_NET_PATH):
        os.makedirs(TRAINED_NET_PATH)
    #Loading the Training Data
    trainloader = pickle.load(open("../data/pickles/trainloader.pickled", "rb"))[node_code]
    # Processor setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device", device)
    # Model Parameter Configuration
    net = [Net().to(device) for i in range(n_device)]
    local_model = [{} for i in range(n_device)]
    recv_models = [[] for i in range(n_device)]
    # Setting up Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = [
        optim.Adam(net[i].parameters(), lr=learning_rate) for i in range(n_device)
    ]
    # Run "pre-self training"
    for epoch in range(self_train_epoch):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [x_train, y_train]
            x_train, y_train = data
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            # zero the parameter gradients
            optimizer[node_code].zero_grad()
            # forward + backward + optimize
            y_output = net[node_code](x_train) 
            loss = criterion(y_output, y_train)
            loss.backward()
            optimizer[node_code].step()
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print(f"Pre-self training: [{node_code}, {epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.5f}")
                running_loss = 0.0
    # Load Contact list for mobility / static network simulation
    self_parameters = net[node_code].state_dict()
    contact_list = []
    print(f"Loading ... contact pattern file: {cp_filename}")
    with open(cp_filename) as f:
        contact_list = json.load(f)
    # WAFL Training
    for epoch in range(max_epoch):  # loop over the dataset multiple times
        # print(global_model['fc2.bias'][1])
        contact = contact_list[epoch]
        contact[str(n_device - 1)] = contact[str(n_device - 1)][:1]
        print(f"at t={epoch} : ", contact)
        # receive local_models from contacts
        local_model[node_code] = net[node_code].state_dict()
        nbr = contact[str(node_code)]
        recv_models[node_code] = []
        for k in nbr:
            recv_models[node_code].append(req_parameters(k))
        update_model = recv_models[node_code]
        n_nbr = len(update_model)
        print(f"at {node_code} n_nbr={n_nbr}")
        for k in range(n_nbr):
            for key in update_model[k]:
                update_model[k][key] = recv_models[node_code][k][key] - local_model[node_code][key]

        for k in range(n_nbr):
            for key in update_model[k]:
                local_model[node_code][key] += (
                    update_model[k][key] * fl_coefficiency / (n_nbr + 1)
                )
        nbr = contact[str(node_code)]
        if len(nbr) > 0:
            net[node_code].load_state_dict(local_model[node_code])
        nbr = contact[str(node_code)]
        if len(nbr) == 0:
            continue
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [x_train, y_train]
            x_train, y_train = data
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            # zero the parameter gradients
            optimizer[node_code].zero_grad()
            # forward + backward + optimize
            y_output = net[node_code](x_train)
            loss = criterion(y_output, y_train)
            loss.backward()
            optimizer[node_code].step()
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print(
                    f"[{node_code}, {epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.5f}"
                )
                running_loss = 0.0
        self_parameters = net[node_code].state_dict()
        # print(net.state_dict())
        print(f"Model saving ... at {epoch+1}")
        torch.save(
            net[node_code].state_dict(),
            f"{TRAINED_NET_PATH}/mnist_net_{node_code}_{epoch+1:04d}.pth",
        )
    print(f"Finished Training. Models were saved in {TRAINED_NET_PATH}. Next, run draw_trend_graph.py and draw_confusion_matrix.py")
