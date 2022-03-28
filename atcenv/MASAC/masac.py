import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from atcenv.MASAC.buffer import ReplayBuffer
from atcenv.MASAC.networks import ActorNetwork, CriticNetwork, ValueNetwork

import atcenv.units as u
import math

NUMBER_INTRUDERS_STATE = 2
MAX_DISTANCE = 250*u.nm
MAX_BEARING = math.pi

MEANS = [57000,57000,0,0,0,0,0,0]
STDS = [31500,31500,100000,100000,1,1,1,1]

class MASAC:
    def __init__(self, alpha=0.003, beta=0.003, n_agents = 10, state_size = 14,
            input_dims=[14], actionbounds = 1, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
            layer1_size=256, layer2_size=256, batch_size=256, reward_scale=20):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, n_agents, state_size, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.n_agents = n_agents
        self.state_size = state_size

        self.actor = ActorNetwork(n_actions=n_actions, name='actor', max_action=actionbounds)
        self.critic_1 = CriticNetwork(n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(n_actions=n_actions, name='critic_2')
        self.value = ValueNetwork(name='value')
        self.target_value = ValueNetwork(name='target_value')

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.critic_2.compile(optimizer=Adam(learning_rate=beta))
        self.value.compile(optimizer=Adam(learning_rate=beta))
        self.target_value.compile(optimizer=Adam(learning_rate=beta))

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def do_step(self, observation, max_speed, min_speed, test = False, batch = False):
        
        state = tf.convert_to_tensor([observation])
        actions, _ = self.actor.sample_normal(state, reparameterize=False, test=test, batch=batch)

        return actions[0]

    def setResult(self,episode_name, state, new_state, reward, action, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_value.weights
        for i, weight in enumerate(self.value.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_value.set_weights(weights)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic_1.save_weights(self.critic_1.checkpoint_file)
        self.critic_2.save_weights(self.critic_2.checkpoint_file)
        self.value.save_weights(self.value.checkpoint_file)
        self.target_value.save_weights(self.target_value.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic_1.load_weights(self.critic_1.checkpoint_file)
        self.critic_2.load_weights(self.critic_2.checkpoint_file)
        self.value.load_weights(self.value.checkpoint_file)
        self.target_value.load_weights(self.target_value.checkpoint_file)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            print('blabla')
            return

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)
                
        full_state = state.reshape(self.batch_size, self.n_agents*self.state_size)
        full_state_ = new_state.reshape(self.batch_size, self.n_agents*self.state_size)
        full_action = action.reshape(self.batch_size,self.n_agents*self.n_actions)
        full_reward = reward.reshape(self.batch_size,self.n_agents)

        full_state = tf.convert_to_tensor(full_state, dtype=tf.float32)
        full_state_= tf.convert_to_tensor(full_state_, dtype=tf.float32)
        full_action = tf.convert_to_tensor(full_action, dtype=tf.float32)
        full_reward = tf.convert_to_tensor(full_reward, dtype=tf.float32)

        full_reward = tf.reduce_mean(full_reward, axis = 1) 

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            value = tf.squeeze(self.value(full_state), 1)
            value_ = tf.squeeze(self.target_value(full_state_), 1)

            current_policy_actions, log_probs = self.actor.sample_normal(states,
                                                        reparameterize=False)

            current_policy_actions = tf.reshape(current_policy_actions,[self.batch_size,self.n_agents*self.n_actions])
            log_probs = tf.reshape(log_probs,[self.batch_size,self.n_agents] )
            log_probs = tf.reduce_mean(log_probs,axis=1) #take the mean log_prob because of multiple actors and only 1 critic                                            
            #log_probs = tf.squeeze(log_probs,1)
            q1_new_policy = self.critic_1(full_state, current_policy_actions)
            q2_new_policy = self.critic_2(full_state, current_policy_actions)
            critic_value = tf.squeeze(
                                tf.math.minimum(q1_new_policy, q2_new_policy), 1)

            value_target = critic_value - log_probs
            value_loss = 0.5 * keras.losses.MSE(value, value_target)

        value_network_gradient = tape.gradient(value_loss, self.value.trainable_variables)
        self.value.optimizer.apply_gradients(zip( value_network_gradient, self.value.trainable_variables))


        with tf.GradientTape() as tape:
            # in the original paper, they reparameterize here. We don't implement
            # this so it's just the usual action.
            new_policy_actions, log_probs = self.actor.sample_normal(states,
                                                reparameterize=True)
            new_policy_actions = tf.reshape(new_policy_actions,[self.batch_size,self.n_agents*self.n_actions])
            log_probs = tf.reshape(log_probs,[self.batch_size,self.n_agents] )
            log_probs = tf.reduce_mean(log_probs,axis=1) #take the mean log_prob because of multiple actors and only 1 critic
            #log_probs = tf.squeeze(log_probs, 1)
            q1_new_policy = self.critic_1(full_state, new_policy_actions)
            q2_new_policy = self.critic_2(full_state, new_policy_actions)
            critic_value = tf.squeeze(tf.math.minimum(
                                        q1_new_policy, q2_new_policy), 1)
        
            actor_loss = log_probs - critic_value
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, 
                                            self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
                        actor_network_gradient, self.actor.trainable_variables))
        

        with tf.GradientTape(persistent=True) as tape:
            # I didn't know that these context managers shared values?
            reward = np.sum(reward,axis=1)
            q_hat = self.scale*reward + self.gamma*value_ #*(1-done) no terminal flag is used
            q1_old_policy = tf.squeeze(self.critic_1(full_state, full_action), 1)
            q2_old_policy = tf.squeeze(self.critic_2(full_state, full_action), 1)
            critic_1_loss = 0.5 * keras.losses.MSE(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * keras.losses.MSE(q2_old_policy, q_hat)
    
        critic_1_network_gradient = tape.gradient(critic_1_loss,   self.critic_1.trainable_variables)
        critic_2_network_gradient = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)

        self.critic_1.optimizer.apply_gradients(zip( critic_1_network_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(critic_2_network_gradient, self.critic_2.trainable_variables))

        self.update_network_parameters()
    
    def normalizeState(self, s_t, max_speed, min_speed):
         # distance to closest #NUMBER_INTRUDERS_STATE intruders
        for i in range(0, NUMBER_INTRUDERS_STATE):
            s_t[i] = (s_t[i]-MEANS[0])/(STDS[0]*2)

        # relative bearing to closest #NUMBER_INTRUDERS_STATE intruders
        for i in range(NUMBER_INTRUDERS_STATE, NUMBER_INTRUDERS_STATE*2):
            s_t[i] = (s_t[i]-MEANS[1])/(STDS[1]*2)

        for i in range(NUMBER_INTRUDERS_STATE*2, NUMBER_INTRUDERS_STATE*3):
            s_t[i] = (s_t[i]-MEANS[2])/(STDS[2]*2)
        
        for i in range(NUMBER_INTRUDERS_STATE*3, NUMBER_INTRUDERS_STATE*4):
            s_t[i] = (s_t[i]-MEANS[3])/(STDS[3]*2)

        for i in range(NUMBER_INTRUDERS_STATE*4, NUMBER_INTRUDERS_STATE*5):
            s_t[i] = (s_t[i])/(3.1415)

        # current bearing
        
        # current speed
        s_t[NUMBER_INTRUDERS_STATE*5] = ((s_t[NUMBER_INTRUDERS_STATE*5]-min_speed)/(max_speed-min_speed))*2 - 1
        # optimal speed
        s_t[NUMBER_INTRUDERS_STATE*5 + 1] = ((s_t[NUMBER_INTRUDERS_STATE*5 + 1]-min_speed)/(max_speed-min_speed))*2 - 1
        # # distance to target
        # s_t[NUMBER_INTRUDERS_STATE*2 + 2] = s_t[NUMBER_INTRUDERS_STATE*2 + 2]/MAX_DISTANCE
        # # bearing to target
        s_t[NUMBER_INTRUDERS_STATE*5+2] = s_t[NUMBER_INTRUDERS_STATE*5+2]
        s_t[NUMBER_INTRUDERS_STATE*5+3] = s_t[NUMBER_INTRUDERS_STATE*5+3]

        # s_t[0] = s_t[0]/MAX_BEARING

        return s_t
