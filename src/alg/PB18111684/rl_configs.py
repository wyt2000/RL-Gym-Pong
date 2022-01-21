params = {
        'alg_name': 'DQN',
        'max_iteration': 1000,
        'frame_buffer_size': 4,
        'replay_buffer_size': 10000,
        'eps_init': 1.0,
        'eps_decay': 0.999985,
        'eps_min': 0.02,
        'learning_rate': 1e-4,
        'batch_size': 32,
        'gamma': 0.99,
        'sync_target_network': 1000
}
