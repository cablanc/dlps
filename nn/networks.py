import torch
import numpy as np


class FullyConnectedNN(torch.nn.Module):
    '''
    fully connected neural network
    
    Attributes:
    ----------
    indim : int
        dimension of the input
    outdim : int
        dimension of the output
    hdim : int
        dimension of hidden layers
    num_hidden : int
        number of hidden layers
    model : torch.nn.Sequential
        
    Methods:
    -------
    forward(data)
        returns network output
    '''
    def __init__(self, indim: int, outdim: int, hdim: int, num_hidden: int):
        '''
        builds model
        
        
        Attributes:
        ----------
        indim : int
            dimension of the input
        outdim : int
            dimension of the output
        hdim : int
            dimension of hidden layers
        num_hidden : int
            number of hidden layers
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
            
        # build final layer without nonlinearity
        hidden2out = torch.nn.Linear(hdim, outdim)
        layers.append(hidden2out)
        
        self.model = torch.nn.Sequential(*layers)
        # print(self.model)
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        '''
        returns network output
        
        data : torch.Tensor
            input data
        '''
        batch_size = data.shape[0]
        return self.model.forward(data.reshape(batch_size, -1))
            

class ConvolutionalNN(torch.nn.Module):
    '''
    convolutional neural network
    
    Attributes:
    ----------
    filter_shapes : list of tuples
        each tuple provides a filter shape (in channels, out channels, filter height)
    outdim : int
        dimension of the output
    model : torch.nn.Sequential
        
    Methods:
    -------
    forward(data)
        returns network output
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
        # TODO: implement a function to compute this number!!!
        # https://cs231n.github.io/convolutional-networks/#conv
        # https://stackoverflow.com/questions/34739151/calculate-dimension-of-feature-maps-in-convolutional-neural-network
        num_final_feature_params = 512
        hidden2out = torch.nn.Linear(num_final_feature_params, outdim)
        # print(hidden2out)
        layers.append(hidden2out)
        
        self.model = torch.nn.Sequential(*layers)
        # print(self.model)
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        '''
        returns network output
        
        data : torch.Tensor
            input data
        '''
        return self.model.forward(data)
    

class RecurrentNN(torch.nn.Module):
    '''
    recurrent neural network
    
    Attributes:
    ----------
    indim : int
    hdim : int
    outdim : int
    num_layers : int
    sequence_len : int
    rnn : torch.nn.RNN
    fc : torch.nn.Linear
        map from hidden space to input space
        
    Methods:
    -------
    forward(data)    
        returns network output
    '''
    def __init__(self, indim: int, hdim: int, outdim: int, num_layers: int, sequence_len: int):
        '''
        Parameters:
        ----------
        indim : int
        hdim : int
        outdim : int
        num_layers : int
        sequence_len : int
        '''
        super(RecurrentNN, self).__init__()
        
        self.indim = indim
        self.hdim = hdim
        self.outdim = outdim
        self.num_layers = num_layers
        self.sequence_len = sequence_len
        
        self.rnn = torch.nn.RNN(self.indim, self.hdim, num_layers=self.num_layers, batch_first=True)
        self.fc = torch.nn.Linear(self.hdim, self.outdim)
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        '''
        returns network output
        
        data : torch.Tensor
            input data
        '''
        batch_size = data.shape[0]
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hdim)
        
        # output : the output at each unrolling with shape (batch_size, sequence_len, hidden_dim)
        # hidden : the output at each layer of the final unrolling with shape (batch_size, num_layers, hidden_dim)
        outputs, hidden = self.rnn(data.squeeze(1), h0)
        final_output = outputs[:, -1, :]
        
        prediction = self.fc(final_output)
            
        return prediction
