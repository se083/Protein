#!/usr/bin/env python
import torch
import torch.utils.data
from utils_pytorch import *
from torch.nn import functional as F
import numpy as np
import pandas as pd
from collections import defaultdict
import wandb
import copy


def train(model,train_loader,optimizer, **loss_kwargs):
    model.train() # sets the model into training mode
    losses_dict = defaultdict(int)
    for (data, ) in train_loader:
        data = data.to(torch.device('cuda'))
        optimizer.zero_grad()
        recon_batch = model(data)
        loss_dict = model.loss_function(recon_x=recon_batch, x=data, **loss_kwargs)
        loss_dict['hamming_dist'] = torch.true_divide(torch_hamming_dist(torch_onehot_to_index(recon_batch),torch_onehot_to_index(data)), len(data))
        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()
        for key, value in loss_dict.items():
            losses_dict[key] += value.item()

    nr_batches = len(train_loader)
    for key, value in losses_dict.items():
        value = value/nr_batches
        losses_dict[key] = value
        print(key + ':{:.4f}  '.format(value),  end='')
    wandb.log({'train_' + key:value/nr_batches for (key, value) in losses_dict.items()})
    print('')
    return dict(losses_dict)


def test(model,test_loader, **loss_kwargs):
    model.eval()
    losses_dict = defaultdict(int)
    with torch.no_grad():
        for (data, ) in test_loader:
            data = data.to(torch.device('cuda'))
            recon_batch = model(data)
            loss_dict = model.loss_function(recon_batch, data, **loss_kwargs)
            loss_dict['hamming_dist'] = torch.true_divide(torch_hamming_dist(torch_onehot_to_index(recon_batch),torch_onehot_to_index(data)), len(data))
            loss = loss_dict['loss']
            for key, value in loss_dict.items():
                losses_dict[key] += value.item()

    nr_batches = len(test_loader)
    print('   Test', end=' - ')
    for key, value in losses_dict.items():
        value = value/nr_batches
        losses_dict[key] = value
        print(key + ':{:.4f}  '.format(value),  end='')
    wandb.log({'test_' + key:value/nr_batches for (key, value) in losses_dict.items()})
    print('')
    return dict(losses_dict)


def model_training(model, x_train, x_test, epochs, batch_size, loss_kwargs={}, optimizer_kwargs={}, hyperparameter_kwargs={}):
    input_shape = x_train.shape[1:]
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.FloatTensor(x_train)), shuffle = True, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.FloatTensor(x_test)), shuffle = True, batch_size=batch_size)

    model = model.to(torch.device('cuda'))
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)

    train_losses = []
    test_losses = []
    final_beta = loss_kwargs.get('beta',1) # for vanilla vae mmd
    beta_ramping = loss_kwargs.get('beta_ramping',True)

    wandb.init(
        # set the wandb project where this run will be logged
        project="protein-generation",
        
        # track hyperparameters and run metadata
        config={
            "batch_size": batch_size,
            "epochs": epochs,
            "beta": final_beta,
            **optimizer_kwargs,
            **hyperparameter_kwargs
        }
    )
    
    max_k = 5
    grace_epochs = 10

    min_loss = float('inf')
    above_min_epochs = 0
    best_model = None

    for epoch in range(0, epochs):
        # increasing beta for vae
        if beta_ramping:
            loss_kwargs['beta'] = (final_beta / epochs) * epoch * (1 + 1/epochs)

        print('Epoch ' + str(epoch+1), end = ' - ')
        train_loss_dict = train(model=model, train_loader=train_loader, optimizer=optimizer, **loss_kwargs)
        train_loss_dict['Epoch'] = epoch+1
        train_losses += [train_loss_dict]

        test_loss_dict = test(model=model, test_loader=test_loader, **loss_kwargs)
        test_loss_dict['Epoch'] = epoch+1
        test_losses += [test_loss_dict]

        print('')
        test_loss = test_loss_dict['loss']

        if test_loss <= min_loss:
            min_loss = test_loss
            above_min_epochs = 0
            best_model = copy.deepcopy(model)
            # min_train_losses = train_loss_dict
            # min_test_losses = test_loss_dict
        else:
            above_min_epochs += 1

        if epoch >= grace_epochs and above_min_epochs > max_k:
            print(f'early stopping because performance has not improved for {max_k} epochs')
            break
    
    train_losses = pd.DataFrame(train_losses)
    train_losses['Type'] = 'Training_data'
    test_losses = pd.DataFrame(test_losses)
    test_losses['Type'] = 'Test_data'
    loss_df = pd.concat([train_losses,test_losses])
    best_model = best_model.to(torch.device('cuda'))
    wandb.finish()

    return best_model, loss_df


def ztrack_training(model, x_train, x_test, optimizer_kwargs, loss_kwargs):
    input_shape = x_train.shape[1:]
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.FloatTensor(x_train)), shuffle = True, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.FloatTensor(x_test)), shuffle = True, batch_size=batch_size)

    # make dataset for save_epoch_data
    x_train_part_index = np.random.choice(x_train.shape[0],1000,replace = False)
    x_train_part = x_train[x_train_part_index]
    x_train_part = torch.tensor(x_train_part.astype('float32')).to(torch.device('cuda'))
    save_list = []

    model = model.to(torch.device('cuda'))
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)

    train_losses = []
    test_losses = []

    for epoch in range(0, epochs):
        epoch_beta = (beta / epochs) * epoch * (1 + 1/epochs)
        print('Epoch ' + epoch+1, end = '')
        train_losses += [train(model=model, train_loader=train_loader, optimizer=optimizer, **loss_kwargs)]
        test_losses += [test(model=model, test_loader=test_loader, **loss_kwargs)]
        with torch.no_grad():
            save = list(model.encoder(x_train_part))[0].cpu().numpy()
            save_df = pd.DataFrame(save, columns = [''.join(['X',str(s)]) for s in range(save.shape[1])])
            save_df['Epoch'] = epoch+1
            save_df['input_index'] = x_train_part_index
            save_list.append(save_df)

    z_track = pd.concat(save_list)
    train_losses = pd.DataFrame(train_losses)
    train_losses['Type'] = 'Training_data'
    test_losses = pd.DataFrame(test_losses)
    test_losses['Type'] = 'Test_data'
    loss_df = pd.concat([train_losses,test_losses])

    return model, loss_df, z_track



def model_predict(model, input_data, batch_size = 1):
    if batch_size < 1:
        input_data = torch.tensor(input_data.astype('float32')).to(torch.device('cuda'))
        with torch.no_grad():
            output = model(input_data)
            if type(output) is tuple:
                output = list(output)[0]
        return output.cpu().numpy()
    else:
        data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.FloatTensor(input_data.astype('float32'))), shuffle = False, batch_size=batch_size)
        with torch.no_grad():
            output = []
            for i, (data, ) in enumerate(data_loader):
                data = data.to(torch.device('cuda'))
                out = model(data)
                if type(out) is tuple:
                    out = list(out)[0]
                output.append(out.cpu().numpy())
        return np.vstack(output)

