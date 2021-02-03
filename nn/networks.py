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
        num_examples, chan, height, width = data.shape
        return self.model.forward(data.reshape(-1, height*width))
            

class ConvolutionalNN(torch.nn.Module):
    '''
    
    https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
    '''
    def __init__(self, filter_shapes, outdim):
        '''
        filter_shapes : list of tuples
            each tuple provides a filter shape (in channels, out channels, filter height)
        '''
        super(ConvolutionalNN, self).__init__()
        
        self.filter_shapes = filter_shapes
        self.num_layers = len(filter_shapes)
        
        layers = []
        for i in range(self.num_layers):
            conv_layer = torch.nn.Conv2d(*self.filter_shapes[i])
            nonlinearity = torch.nn.ReLU()
            layers.extend([conv_layer, nonlinearity])
            
        layers.append(torch.nn.Flatten())
        # implement a function to compute this number!!!
        # https://cs231n.github.io/convolutional-networks/#conv
        # https://stackoverflow.com/questions/34739151/calculate-dimension-of-feature-maps-in-convolutional-neural-network
        num_final_feature_params = 512
        hidden2out = torch.nn.Linear(num_final_feature_params, outdim)
        # print(hidden2out)
        layers.append(hidden2out)
        
        self.model = torch.nn.Sequential(*layers)
        # print(self.model)
    
    def forward(self, data):
        return self.model.forward(data)
    

