def run_bicycle():
    ## Parameters
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)
    import jax
    import jax.numpy as jnp
    import os
    import json

    parameters = json.load(open("figure_specific/bicycle/parameters.json"))
    env_seed = parameters["env_seed"]
    gamma = parameters["gamma"]
    # Sample collection
    n_samples = parameters["n_samples"]
    n_weights = parameters["n_weights"]

    # Trainings
    layers_dimension = parameters["layers_dimension"]
    max_bellman_iterations = parameters["max_bellman_iterations"]
    training_steps = parameters["training_steps"]
    fitting_steps = parameters["fitting_steps_pbo"]
    batch_size_samples = parameters["batch_size_samples"]
    batch_size_weights = parameters["batch_size_weights"]
    initial_weight_std = parameters["initial_weight_std"]
    learning_rate = {
        "first": parameters["starting_lr_pbo"],
        "last": parameters["ending_lr_pbo"],
        "duration": training_steps * fitting_steps * n_samples // batch_size_samples,
    }

    # Visualisation of errors and performances
    n_omegas = parameters["n_omegas"]
    n_thetas = parameters["n_thetas"]
    n_simulations = parameters["n_simulations"]
    horizon = parameters["horizon"]
    max_bellman_iterations_validation = max_bellman_iterations + 10

    # Search for an unused seed
    max_used_seed = 0
    if not os.path.exists("figures/data/PBO_linear_max_linear/"):
        os.makedirs("figures/data/PBO_linear_max_linear/")
    for file in os.listdir("figures/data/PBO_linear_max_linear/"):
        if int(file.split("_")[0]) == max_bellman_iterations and int(file.split("_")[2][:-4]) > max_used_seed:
            max_used_seed = int(file.split("_")[2][:-4])
    max_used_seed

    # keys
    seed = max_used_seed + 1
    env_key = jax.random.PRNGKey(env_seed)
    env_key, sample_key = jax.random.split(env_key)
    key = jax.random.PRNGKey(seed)
    shuffle_key, q_network_key, pbo_network_key = jax.random.split(key, 3)

    ## Environment
    import numpy as np

    from pbo.environments.bicycle import BicycleEnv

    env = BicycleEnv(env_key, gamma)

    from pbo.sample_collection.replay_buffer import ReplayBuffer

    ## Sample collection
    replay_buffer = ReplayBuffer()

    env.reset()
    n_episodes = 0
    n_steps = 0
    positions = [[env.position]]

    for idx_sample in range(n_samples):
        state = env.state

        sample_key, key = jax.random.split(sample_key)
        action = jax.random.choice(key, env.actions_on_max)

        next_state, reward, absorbing, _ = env.step(action)
        n_steps += 1
        positions[n_episodes].append(env.position)

        replay_buffer.add(state, action, reward, next_state, absorbing)

        if absorbing[0] or n_steps >= 20:
            sample_key, key = jax.random.split(sample_key)
            env.reset(
                jax.random.multivariate_normal(
                    key,
                    jnp.zeros(4),
                    jnp.array([[1e-4, -1e-4, 0, 0], [-1e-4, 1e-3, 0, 0], [0, 0, 1e-3, -1e-4], [0, 0, -1e-4, 1e-2]]),
                )
                / 10
            )
            positions[n_episodes] = np.array(positions[n_episodes])
            positions.append([])
            n_episodes += 1
            n_steps = 0

    replay_buffer.cast_to_jax_array()

    ## Weight collection
    from pbo.weights_collection.weights_buffer import WeightsBuffer
    from pbo.networks.learnable_q import FullyConnectedQ

    weights_buffer = WeightsBuffer()

    # Add the validation weights
    q = FullyConnectedQ(
        state_dim=4,
        action_dim=2,
        actions_on_max=env.actions_on_max,
        gamma=gamma,
        network_key=q_network_key,
        layers_dimension=layers_dimension,
        zero_initializer=True,
        learning_rate=learning_rate,
    )
    validation_weights = q.to_weights(q.params)
    weights_buffer.add(validation_weights)

    # Add random weights
    while len(weights_buffer) < n_weights:
        weights = q.random_init_weights()
        weights_buffer.add(weights)

    weights_buffer.cast_to_jax_array()

    ## Train PBO
    from tqdm import tqdm

    from pbo.sample_collection.dataloader import SampleDataLoader
    from pbo.weights_collection.dataloader import WeightsDataLoader
    from pbo.networks.learnable_pbo import DeepPBO

    data_loader_samples = SampleDataLoader(replay_buffer, batch_size_samples, shuffle_key)
    data_loader_weights = WeightsDataLoader(weights_buffer, batch_size_weights, shuffle_key)
    pbo = DeepPBO(
        q=q,
        max_bellman_iterations=max_bellman_iterations,
        network_key=pbo_network_key,
        learning_rate=learning_rate,
        initial_weight_std=initial_weight_std,
    )
    importance_iteration = jnp.ones(max_bellman_iterations + 1)

    for training_step in tqdm(range(training_steps)):
        params_target = pbo.params

        for fitting_step in range(fitting_steps):

            data_loader_weights.shuffle()
            for batch_weights in data_loader_weights:
                data_loader_samples.shuffle()
                for batch_samples in data_loader_samples:
                    pbo.params, pbo.optimizer_state, _ = pbo.learn_on_batch(
                        pbo.params,
                        params_target,
                        pbo.optimizer_state,
                        batch_weights,
                        batch_samples,
                        importance_iteration,
                    )

    ## Metrics

    metrics = np.ones((max_bellman_iterations_validation + 1, n_simulations, 2)) * np.nan
    q_weights = validation_weights

    metrics[0] = env.evaluate(q, q.to_params(q_weights), horizon, n_simulations)

    for bellman_iteration in tqdm(range(1, max_bellman_iterations_validation + 1)):
        q_weights = pbo(pbo.params, q_weights.reshape((1, -1)))[0]

        metrics[bellman_iteration] = env.evaluate(q, q.to_params(q_weights), horizon, n_simulations)

    np.save(
        f"figure_specific/bicycle/figures/data/PBO_linear_max_linear/{max_bellman_iterations}_metrics_{seed}.npy",
        metrics,
    )


if __name__ == "__main__":
    run_bicycle()
