import torch
import numpy as np


def compute_confusion_matrix(prediction_label_data):
    confusion_matrix = np.ones((10,10))
    mistakes = []
    for tripple in prediction_label_data:
        x, y, data = tripple
        confusion_matrix[x,y] += 1
        if x != y:
            mistakes.append(tripple)
            
    return confusion_matrix, mistakes


def train(num_epochs, print_every, trainloader, loss_fcn, optimizer, net):
    '''
    Arguments:
    ---------
    num_epochs : int
    trainloader : torch.utils.data.Dataloader
    net : neural network
    
    Returns:
    -------
    training_loss : list
    '''
    training_loss = []
    for epoch in range(num_epochs):
        for iteration, sample in enumerate(trainloader):
            data, labels = sample
            num_examples, chan, height, width = data.shape

            # pass sample through net
            output = net(data)

            # compute loss
            loss = loss_fcn(output, labels)

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute accuracy
            _, prediction = torch.max(output.data, 1)
            batch_accuracy = (prediction == labels).sum().item() / num_examples

            batch_loss = loss.item()

            if iteration % print_every == 0:
                print('Epoch: {epoch}, Iteration: {iteration}, Loss: {loss:.2f}, Acc: {acc:.2f}'.format(epoch=epoch, iteration=iteration, loss=batch_loss, acc=batch_accuracy))

            training_loss.append(batch_loss)
    
    return training_loss


def evaluate(testloader, loss_fcn, net):
    '''
    computes average test loss and accuracy
    
    Arguments:
    ---------
    testloader : torch.utils.data.Dataloader
    net : neural network
    
    Returns:
    -------
    average_accuracy : float
    average_loss : float
    '''
    # 
    prediction_label_data = []
    total_loss = 0
    total_correct = 0
    total_examples = 0
    for iteration, sample in enumerate(testloader):
        data, labels = sample
        num_examples, chan, height, width = data.shape

        # pass sample net
        output = net(data)

        # compute loss
        loss = loss_fcn(output, labels)

        # compute accuracy
        _, batch_prediction = torch.max(output.data, 1)
        batch_correct = (batch_prediction == labels).sum().item()
        batch_accuracy = batch_correct / num_examples

        batch_loss = loss.item()

        total_loss += batch_loss
        total_correct += batch_correct
        total_examples += num_examples

        np_batch_prediction = batch_prediction.data.numpy()
        np_batch_labels = labels.data.numpy()
        np_data = data.data.numpy()
        prediction_label_data.extend(list(zip(np_batch_prediction, np_batch_labels, np_data)))


    average_accuracy = total_correct / total_examples
    average_loss = total_loss / iteration
    
    return average_accuracy, average_loss, prediction_label_data