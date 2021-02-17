import plotly.graph_objects as go
import networkx as nx
import numpy as np
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
            num_examples = data.shape[0]
            # print(data.shape, labels.shape)

            # pass sample through net
            output = net(data)
            # print(output.shape)

            # compute loss
            loss = loss_fcn(output, labels)

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()

            if iteration % print_every == 0:
                print('Epoch: {epoch}, Iteration: {iteration}, Loss: {loss:.2f}'.format(epoch=epoch, iteration=iteration, loss=batch_loss))

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
        num_examples = data.shape[0]

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


def map_to_simplex(Z_t, No, r, R):
    return r/(R*No)*Z_t + r/No


def generate_random_relative_options(num_agents, num_options):
    '''
    generate random relative options Z size (num_agents, num_opinions)
        num_opinions should sum to zero for each agent
        
    Parameters:
    ----------
    num_agents : int
    num_options : int
    
    Returns:
    -------
    A : ndarray
    '''
    A = np.random.rand(num_agents, num_options) - 0.5
    last_row = -A[:, :-1].sum(axis=1)
    A[:, -1] = last_row
    return A


def run_homogeneous_simulation(T, 
                        dt, 
                        num_agents, 
                        num_options, 
                        gamma, 
                        delta,
                        alpha,
                        beta,
                        d,
                        u,
                        b,
                        A_tilde,
                        Z):
    '''
    run homogeneous simulation
    
    Parameters:
    ----------
    T : int
    dt : float
    num_agents : int
    num_options : int
    gamma : float
    delta : float
    alpha : float
    beta : float
    d : 
    u : 
    b : 
    A_tilde : {0, 1}
    Z : 
    
    Returns:
    -------
    '''
    
    A = build_homogeneous_A(A_tilde, num_options, alpha, beta, gamma, delta)
    
    #print(A)
    
    # Drift Coefficient Matrix
    D = np.ones((num_agents, num_options)) * d
    
    # Bias Matrix
    B = np.ones((num_agents, num_options)) * b
    
    # Attention parameter
    u = np.ones((num_agents, 1)) * u
    
    return euler_integration(Z, dt, T, D, u, B, A)


def build_homogeneous_A(A_tilde, num_options, alpha, beta, gamma, delta):
    '''
    constructs homogeneous A
    
    Parameters:
    ----------
    A_tilde :
    num_options :
    alpha :
    beta :
    gamma :
    delta :
    
    Returns:
    -------
    A : ndarray
    '''
    num_agents = A_tilde.shape[0]
    A = np.zeros((num_options, num_options, num_agents, num_agents))
    
    # fill in A
    for option_j in range(num_options):
        for option_l in range(num_options):
            for agent_i in range(num_agents):
                for agent_k in range(num_agents):
                    # Ajjii = α
                    if option_j == option_l and agent_i == agent_k:
                        A[option_j, option_j, agent_i, agent_i] = alpha
                    # Ajlii = β   
                    elif agent_i == agent_k:
                        A[option_j, option_l, agent_i, agent_i] = beta
                    # Ajjik = γ ̃aik
                    elif option_j == option_l:
                        A[option_j, option_j, agent_i, agent_k] = gamma * A_tilde[agent_i, agent_k]
                    else:
                        A[option_j, option_l, agent_i, agent_k] = delta * A_tilde[agent_i, agent_k]
    
    return A


def euler_integration(z0, dt, T, D, u, B, A):
    '''
    integrates forward using Euler update
    
    Parameters:
    ----------
    z0 :
    dt :
    T :
    D :
    u :
    B :
    A :
    
    Returns:
    -------
    ndarray
        sequence of Zs
    '''
    Z_t = [z0]
    for t in np.arange(0, T, dt):
        Z_cur = Z_t[-1]
        Z_next = Z_cur + dt * compute_zdot(D, u, Z_cur, A, B)
        Z_t.append(Z_next)
        
    return np.asarray(Z_t)


def split_agent_option_matrix(A):
    '''
    split A into inter_agent_same_option and inter_agent_inter_option
    
    Arguments:
    ---------
    A: matrix
        Size: (number of options x number of options x number of agents x number of agents)
    
    Returns:
    -------
    inter_agent_same_option:
        Size: (number of options x number of agents x number of agents)
        Inter-agent, same-option coupling: Ajj_ik, i neq k 
        
    inter_agent_inter_option:
        Size: (number of options x number of options x number of agents x number of agents)
        Inter-agent, inter-option coupling: Ajl_ik, i neq k,j neq l
    '''
    num_options = A.shape[0]
    num_agents = A.shape[2]
    
    inter_agent_same_option = np.zeros((num_options, num_agents, num_agents))
    inter_agent_inter_option = A.copy()
    
    for option_idx in range(num_options):
        inter_agent_same_option[option_idx, :, :] = inter_agent_inter_option[option_idx, option_idx, :, :]
        inter_agent_inter_option[option_idx, option_idx, :, :] = 0
        
    return inter_agent_same_option, inter_agent_inter_option


def compute_udot(u, A_bar, Z, tau_u):
    '''
    computes the state feedback law for attention which is
    
    tau_u * u_i_dot = -u_i + Su(1/No^2 \sum_j \sum_k \sum_l (Abar_ik^jl * Zkl)^2)
                    = -u_i + Su(1/No^2 ||Abar_i * Z||^2)
                    
    Note that Abar is distinct from the adjacency tensor A, and 
    that Z must be flattened in the matrix multiplication, as in the paper. 
    Hence Abar_i must also be flattened. 
    
    Arguments:
    ----------
    u: matrix - attention (engagement) of each agent
        Size: number of agents
        
    A_bar: list of matrices - unweighted edge connection {0,1} to other agents
        Size: num_agents x num_options x (num_agents*num_options)
        Inter-agent, inter-option coupling: Ajl_ik, i neq k,j neq l 
    
    Z: matrix 
        Size: (number of agents x number of options)
        
    tau_u: scalar
        Size: (number of agents x 1)
        
    Returns:
    -------
    udot: matrix
        Size: number of agents 
    '''

    num_options = A_bar[0].shape[0]

    Z_flat = np.matrix.flatten(Z)

    connection_products_magnitude  = np.array([np.linalg.norm(A @ Z_flat)**2 for A in A_bar])

    agent_attention_magnitude_saturation = np.tanh((1/num_options**2) * connection_products_magnitude)

    return 1./tau_u * (-u + agent_attention_magnitude_saturation)


def compute_social_term(A, Z, u):
    '''
    computes the social term which is
    the  product  of  an attention parameter ui ≥ 0 
    and a saturating function of weighted sums of agent opinions that are available 
    to agent i and influence its opinion  of  option j
    
    Arguments:
    ----------
    inter_agent_same_option: matrix
        Size: number of options x number of agents x number of agents
        Inter-agent, same-option coupling: Ajj_ik, i neq k 
        
    inter_agent_inter_option: matrix
        Size: (number of options - 1) x number of options x number of agents x number of agents)
        Inter-agent, inter-option coupling: Ajl_ik, i neq k,j neq l 
    
    Z: matrix
        Size: (number of agents x number of options)
        
    u: vector:
        Size: (number of agents x 1)
        
    Returns:
    -------
    social_term: matrix
        Size: number of agents x number of options
    '''
    inter_agent_same_option, inter_agent_inter_option = split_agent_option_matrix(A)
    #print(inter_agent_same_option, inter_agent_inter_option)
    # print('inter_agent_same_option shape: {}'.format(inter_agent_same_option.shape))
    # print('inter_agent_inter_option shape: {}'.format(inter_agent_inter_option.shape))

    weighted_inter_agent_same_option = np.einsum('jik,kj->ij', inter_agent_same_option, Z)
    thresholded_weighted_inter_agent_same_option = np.tanh(weighted_inter_agent_same_option)
    # print('weighted_inter_agent_same_option shape: {}'.format(weighted_inter_agent_same_option.shape))
    
    weighted_inter_agent_inter_option = np.einsum('jlik,kl->jli', inter_agent_inter_option, Z)
    thresh_weighted_inter_agent_inter_option = np.tanh(weighted_inter_agent_inter_option)
    # print('weighted_inter_agent_inter_option shape: {}'.format(weighted_inter_agent_inter_option.shape))
    # print('weighted_inter_agent_inter_option: {}'.format(weighted_inter_agent_inter_option))

    # Sum the activations over each distinct option l
    cumulative_thresh_weighted_inter_agent_inter_option = \
        np.einsum('jli->ij', thresh_weighted_inter_agent_inter_option)
    # print('cumulative_thresh_weighted_inter_agent_inter_option shape: {}'.format(cumulative_thresh_weighted_inter_agent_inter_option.shape))

    social_term = u * (thresholded_weighted_inter_agent_same_option + \
                       cumulative_thresh_weighted_inter_agent_inter_option)
    
    return social_term


def compute_drift(D, u, Z, A, B):
    '''
    computes drift using S_1 = S_2 = tanh
    
    F_ij(Z) = -d_ij z_ij + u_i ( S_1(\sum_k Ajj_ik z_kj) + \sum_(l neq j) S_2(\sum_k Ajl_ik z_kl)) + b_ij
    
    
    Arguments:
    ---------
    D: matrix 
        size - (number of agents x number of options)
        resistance term. In the general model, a larger dij implies a greater resistance of agent i to 
        forming a non-neutral opinion about option j.
    
    u: vector
        len(U) = number of agents
        attention parameter.
        
    Z: matrix
        size - (number of agents x number of options)
        
    A: tuple
        (
    B: matrix
        size - (number of agents x number of options)
        represents an input signal from the environment or a bias or predisposition that  
        directly affects agent i’s opinion of option j
    '''
    social_term = compute_social_term(A, Z, u)
    
    # F_ij(Z) = -d_ij z_ij + u_i ( S_1(\sum_k Ajj_ik z_kj) + \sum_(l neq j) S_2(\sum_k Ajl_ik z_kl)) + b_ij
    return -D * Z + social_term + B


def compute_zdot(D, u, Z, A, B):
    '''
    computes equation of motion of relative opinions
    
    Zij_dot = F_ij(Z) - (1/No) * \sum_(l = 1)^No F_il(Z)
    
    Arguments:
    ---------
    D: matrix 
        size - (number of agents x number of options)
        resistance term. In the general model, a larger dij implies a greater resistance of agent i to 
        forming a non-neutral opinion about option j.
    
    u: vector
        len(U) = number of agents
        attention parameter.
        
    Z: matrix
        size - (number of agents x number of options)
        
    A: tuple
        (Inter-agent, same-option coupling: Ajj_ik, i neq k (Size: No x Na x Na)
         Inter-agent, inter-option coupling: Ajl_ik, i neq k,j neq l (Size: (No-1) x No x Na x Na) )
         
    B: matrix
        size - (number of agents x number of options)
        represents an input signal from the environment or a bias or predisposition that  
        directly affects agent i’s opinion of option j
        
    Returns:
    -------
    Z_dot: matrix
        size - ()
        Zij_dot = F_ij(Z) - (1/No) * \sum_(l = 1)^No F_il(Z)
    '''
    
    num_options = D.shape[-1]
    
    F = compute_drift(D, u, Z, A, B)
    
    average_option_drift = 1/num_options * np.einsum('il->i', F)

    return F - average_option_drift[:, np.newaxis]


def get_edges(graph):
    '''
    returns x,y coordinates of edges in a list
    
    Arguments:
    ---------
    graph: networkx.Graph
    
    Returns:
    -------
    edge_x: list
    edge_y: list
    '''
    edge_x = []
    edge_y = []
    
    for edge in graph.edges():
        
        x0, y0 = graph.nodes[edge[0]]['pos']
        x1, y1 = graph.nodes[edge[1]]['pos']
        
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        
    return edge_x, edge_y


def get_nodes(graph):
    
    '''
    returns x,y coordinates of node in a list
    
    Arguments:
    ---------
    graph: networkx.Graph
    
    Returns:
    -------
    node_x: list
    node_y: list
    '''
    node_x = []
    node_y = []
    
    for node in graph.nodes():
        x, y = graph.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        
    return node_x, node_y


def get_degree(graph):
    
    '''
    returns degree of nodes in a list
    
    Arguments:
    ---------
    graph: networkx.Graph
    
    Returns:
    -------
    node_degree: list
    node_text: list
    '''
    node_degree = []
    node_text = []
    
    for adjacencies in graph.adjacency():
        neighbors = adjacencies[1]
        degree = len(neighbors)
        
        node_degree.append(degree)
        node_text.append('# of connections: '+str(degree))
        
    return node_degree, node_text


def build_all_to_all(num_nodes):
    '''
    returns an all-to-all graph
    
    Arguments:
    ---------
    num_nodes: int
    
    Returns:
    -------
    graph: networkx.Graph
        all-to-all graph with num_nodes nodes
    '''
    
    # create all-to-all edge tuples
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            # no self edges
            if i == j:
                continue
                
            edges.append((i + 1, j + 1)) # networkx.Graph starts from 1 not 0
            
    G = nx.Graph()
    G.add_edges_from(edges)
    # print(edges)
    
    return G
