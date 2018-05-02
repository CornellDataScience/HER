import numpy as np
import inspect


def make_sample_her_transitions(replay_strategy, replay_k, reward_fun):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0


    def _sample_her_transitions(episode_batch, batch_size_in_transitions, rollout_worker):
        #episode_batch is {key: array(buffer_size x T x dim_key)}
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        # Select which episodes and time steps to use.
        #length of episode_idxs by default is 256
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        #T by default is 50
        #length of t_samples by default is 256
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}


        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        # batch_size by default here is 256
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)

        #for each transition that we want to substitute a goal
        for t in range(0, len(her_indexes[0])):
            #create probability distribution to sample from
            specific_time = t_samples[her_indexes[0][t]]
            specific_episode = episode_idxs[her_indexes[0][t]]

            future_visits = []
            #iterate through the future time steps for this specific transition
            for m in range(specific_time, T):

                reached_goal = episode_batch['ag'][specific_episode, m]
                reached_hash = rollout_worker.countTracker.compute_hash_code(reached_goal)
                num_visits = rollout_worker.countTracker.hashtable[reached_hash]
                future_visits.append([m, num_visits])


            future_visits = sorted(future_visits, key=lambda x : x[1], reverse=True)
            future_visits = np.array(future_visits)

            time_steps = future_visits[:,0]

            unnormalized_probs = np.arange(1, (T - specific_time) + 1)
            alpha = 1

            unnormalized_probs = np.power(unnormalized_probs, alpha)
            normalizing_constant = sum(unnormalized_probs)
            probs = unnormalized_probs / normalizing_constant

            #sample from the created probability distribution
            selected_goal = np.random.choice(time_steps, p=probs)

            #Substitute in the newly selected goals
            future_ag = episode_batch['ag'][specific_episode, selected_goal]
            transitions['g'][her_indexes[0][t]] = future_ag


        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions

    return _sample_her_transitions
