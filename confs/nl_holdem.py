{
    'env': 'NlHoldemEnvWithOpponent',
    'rollout_fragment_length': 50,
    'train_batch_size': 4000,
    'num_workers': 8,
    'num_envs_per_env_runner': 1,
    #'broadcast_interval': 5,
    #'max_sample_requests_in_flight_per_worker': 1,
    #'num_data_loader_buffers': 4,
    'num_gpus': 0,
    'gamma': 1,
    'entropy_coeff': 1e-1,
    'lr': 3e-4,
    'model':{
        'custom_model': 'NlHoldemNet',
        'max_seq_len': 20,
        'custom_model_config': {
        },
    },
    "env_config":{
        'custom_options': {
            'weight':"default",
            "cut":[[0,12],[13,25],[26,38],[39,51],[52,53],[53,54]],
            'epsilon': 0.15,
            'tracker_n': 1000,
            'conut_bb_rather_than_winrate': 2,
            'use_history': True,
            'use_cardnum': True,
            'history_len': 20,
            'num_players': 6,
        },
    }
}