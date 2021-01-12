import torch
import numpy as np


def build_mlp(dims):
    '''
    build multi-layer perceptron; fully-connected neural network
    
    default initialization of linear layers described here:
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    custom initialization can be implemented as described in weights_init here:
    https://github.com/pytorch/examples/blob/master/dcgan/main.py#L95
    
    Arguments:
    ---------
    dims: list
    
    Returns:
    -------
    mlp: torch.nn.Sequential
    '''
    # print(dims)
    indims = dims[:-2]
    outdims = dims[1:-1]

    feature_layers = [torch.nn.Linear(indim, outdim) for indim, outdim in zip(indims, outdims)]
    num_feature_layers = len(feature_layers)
    nonlinearities = [torch.nn.ReLU() for i in range(num_feature_layers)]

    layers = []
    for l in zip(feature_layers, nonlinearities):
        layers.extend(l)
        
    # append final layer with no bias
    layers.append(torch.nn.Linear(dims[-2], dims[-1], bias=None))
    # print(layers)

    return torch.nn.Sequential(*layers)


def apply_linear_dynamics(A, B, x, u):
    '''
    computes next state
    
    Arguments:
    ---------
    A: ndarray
        dynamics matrix with shape - (2, 2)
    B: ndarray
        control transformation with shape - (2, 1)
    x: ndarray
        initial state with shape - (2, 1)
    u: ndarray
        initial control with shape - (1, 1)
        
    Returns:
    -------
    x_next: ndarray
        next state with shape - (2, 1)
    '''
    
    return np.matmul(A, x) + np.matmul(B, u)


def apply_linear_dynamics_tensor(A, B, x, u):
    '''
    computes next state
    
    Arguments:
    ---------
    A: ndarray
        dynamics matrix with shape - (2, 2)
    B: ndarray
        control transformation with shape - (2, 1)
    x: ndarray
        initial state with shape - (2, 1)
    u: ndarray
        initial control with shape - (1, 1)
        
    Returns:
    -------
    x_next: ndarray
        next state with shape - (2, 1)
    '''
    return torch.matmul(A, x) + torch.matmul(B, u)


def compute_K(A, B, P, R):
    '''
    computes gain to solve for optimal control input
    K = -(R + B^T P B)^{-1} B^T P A
    
    https://stanford.edu/class/ee363/lectures/dlqr.pdf (slide 21)
    Using the dynamic programming formulation for the value function
    the optimal control can be solved for by going backward
    
    Arguments:
    ---------
    A: ndarray
        dynamics matrix
    B: ndarray
        control transformation matrix
    P: ndarray
        cost-to-go matrix
    R: ndarray
        input cost matrix
    
    Returns:
    -------
    K: ndarray
        gain matrix
    '''
    # intermediate matrix
    Bt_P = np.matmul(B.T, P)
    
    M = R + np.matmul(Bt_P, B)
    N = np.matmul(Bt_P, A)
    
    M_inv = np.linalg.inv(M)
    
    return -np.matmul(M_inv, N)


def compute_P(A, B, P_next, Q, K_cur):
    '''
    computes the cost-to-go and gain matrix
    P_t = Q + A^T P_{t+1} A - A^T P_{t+1} B (R + B^T P_{t+1} B)^{-1} B^T P_{t+1} A
    
    https://stanford.edu/class/ee363/lectures/dlqr.pdf  (slide 22)
    Using the dynamic programming formulation for the value function
    the optimal control can be solved for by going backward
    
    Arguments:
    ---------
    A: ndarray
        dynamics matrix
    B: ndarray
        control transformation matrix
    P_next: ndarray
        cost-to-go matrix for the next step
    Q: ndarray
        output cost matrix
    K_cur: ndarray
        gain matrix
    
    Returns:
    -------
    P_cur: ndarray
        cost-to-go matrix for the current step
    '''
    # intermediate computation
    # A^T P_{t+1} A
    At_P_next = np.matmul(A.T, P_next)
    At_P_next_A = np.matmul(At_P_next, A)
    
    # intermediate computation
    # A^T P_{t+1} B
    At_P_next_B = np.matmul(At_P_next, B)
    
    # P_t = Q + A^T P_{t+1} A - A^T P_{t+1} B (R + B^T P_{t+1} B)^{-1} B^T P_{t+1} A
    P_cur = Q + At_P_next_A + np.matmul(At_P_next_B, K_cur)
    
    return P_cur


def generate_LQR_policy(A, B, R, Q, Q_f, n):
    '''
    generates LQR policy
    
    Arguments:
    ---------
    A: ndarray
        dynamics matrix
    B: ndarray
        control transformation matrix
    Q: ndarray
        output cost matrix
    Q_f: ndarray
        final output cost matrix
    R: ndarray
        input cost matrix
    n: int
        number of steps
    
    Returns:
    -------
    K_t: list
        gains
    '''
    # P_n = Q_f = 0
    P = Q_f
    K_t = []
    
    for i in range(n):
        K = compute_K(A, B, P, R)
        P = compute_P(A, B, P, Q, K)
        K_t.insert(0, K)
        
    return K_t


def apply_LQR_policy(K, x, t):
    '''
    Argumets:
    --------
    K: list
        gains
    x: ndarray
        state with shape - (2, 1)
    t: int
    
    Returns:
    -------
    control input at time t
    '''
    
    return np.matmul(K[t], x)
    
    
def compute_trajectories(A, B, K_t, x0, apply_policy, n):
    '''
    computes trajectories
    
    Arguments:
    ---------
    A: ndarray
        dynamics matrix
    B: ndarray
        control transformation matrix
    K_t: list
        gains
    x0: ndarray
        initial state with shape - (2, 1)
    n: int
        number of steps
    
    Returns:
    -------
    x: list
        states
    u: list
        control
    '''
        
    x = [x0]
    u = []
    
    for t in range(n - 1):
        x_cur = x[-1]
        u_cur = apply_policy(K_t, x_cur, t)
        x_next = apply_linear_dynamics(A, B, x_cur, u_cur)
        
        u.append(u_cur)
        x.append(x_next)
    
    return x, u