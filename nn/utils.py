import torch
import numpy as np


class FullyConnectedNN(torch.nn.Module):
    '''
    
    https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
    '''
    def __init__(self, indim, outdim, hdim, num_hidden):
        '''
        '''
        super(FullyConnectedNN, self).__init__()
        
        self.indim = indim
        self.outdim = outdim
        self.hdim = hdim
        self.num_hidden = num_hidden
        
        # define layers
        # if there were two hidden layers processing would look like:
        # indim --> hdim --> nonlinearity --> hdim --> nonlinearity --> outdim
        # which would require the following mappings:
        # (indim --> hdim), (nonlinearity), (hdim --> hdim), (nonlinearity), (hdim --> outdim)
        in2hidden = torch.nn.Linear(indim, hdim)
        nonlinearity = torch.nn.ReLU()
        
        layers = [in2hidden, nonlinearity]
        for i in range(num_hidden - 1):
            hidden2hidden = torch.nn.Linear(hdim, hdim)
            nonlinearity = torch.nn.ReLU()
            layers.extend([hidden2hidden, nonlinearity])
            
        hidden2out = torch.nn.Linear(hdim, outdim)
        layers.append(hidden2out)
        
        self.model = torch.nn.Sequential(*layers)
        # print(self.model)
    
    def forward(self, data):
        return self.model.forward(data)
            

def compute_confusion_matrix(prediction_label_data):
    confusion_matrix = np.ones((10,10))
    mistakes = []
    for tripple in prediction_label_data:
        x, y, data = tripple
        confusion_matrix[x,y] += 1
        if x != y:
            mistakes.append(tripple)
            
    return confusion_matrix, mistakes