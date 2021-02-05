import numpy as np


def compute_dynamical_update(state, g, l):
    '''
    
    Arguments:
    ---------
    
    Returns:
    -------
    '''
    theta, d_theta = np.split(state, indices_or_sections=2, axis=-1)
    d2_theta = -g/l * np.sin(theta)
    # print(np.sin(theta))
    # print(d_theta.shape, d2_theta.shape)
    
    return np.hstack((d_theta, d2_theta))


def sym_euler_update(state, d_state, dt):
    '''
    '''
    theta, d_theta = np.split(state, indices_or_sections=2, axis=-1)
    _, d2_theta = np.split(d_state, indices_or_sections=2, axis=-1)

    d_theta_next = d_theta + dt * d2_theta
    theta_next = theta + dt * d_theta

    next_state = np.hstack((theta_next, d_theta_next))
    
    return next_state


def generate_dataset(num_examples, sequence_len, g, l, dt):
    '''
    '''
    theta_0 = np.random.uniform(-np.pi, np.pi, size=(num_examples, 1))
    d_theta_0 = np.random.uniform(-1., 1., size=(num_examples, 1))
    # print(theta_0.shape, d_theta_0.shape)
    init_cond = np.hstack((theta_0, d_theta_0))
    # print(init_cond.shape)
    
    data = [init_cond]
    for i in range(sequence_len - 1):
        cur_state = data[-1]
        dynamical_update = compute_dynamical_update(cur_state, g, l)
        next_state = sym_euler_update(cur_state, dynamical_update, dt)
        data.append(next_state)
    
    data = np.asarray(data) # shape - sequence_len x num_examples x state_dim
    new_idx_order = [1, 0, 2]
    data = np.transpose(data, new_idx_order) # shape - num_examples x sequence_len x state_dim
    
    return data