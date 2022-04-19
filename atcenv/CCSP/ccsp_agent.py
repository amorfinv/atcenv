"""
CCSP Agent Class

Based on MASAC Agent Class, Critic and Value network take the entire set of observations,
and entire set of actions (Critic only), instead of just local observations.
Reward per timestep is the global (summed) reward of the entire set of agents.

Gradient for the actor network updates is taken with respect to the actions of all agents
instead of the individual agents per step

During training try to limit the number of agents in the environment to allow for faster training.
Adjust the other environment parameters such as airspace size accordingly to not trivialize the problem.
When testing the number of Agents can be scaled up as the Critic no longer is called.
"""

from math import gamma
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from atcenv.CCSP.buffer import ReplayBuffer
from atcenv.CCSP.mactor_critic import Actor, CriticQ, CriticV
from torch.nn.utils.clip_grad import clip_grad_norm_


GAMMMA = 0.99
TAU =5e-3
INITIAL_RANDOM_STEPS = 0
POLICY_UPDATE_FREQUENCE = 1

BUFFER_SIZE = 5000000
BATCH_SIZE = 256

LR_A = 6e-5
LR_Q = 6e-4

MEANS = [57000,57000,0,0,0,0,0,0]
STDS = [31500,31500,100000,100000,1,1,1,1]

class CCSPAgent:
    def __init__(self, num_agents, action_dim, state_dim, intruders_state, use_altitude):
        self.statedim = state_dim
        self.num_agents = num_agents
        self.actiondim = action_dim
        self.number_intruders_state = intruders_state
        self.use_altitude = use_altitude

        self.memory = ReplayBuffer(self.statedim, self.actiondim, self.num_agents, BUFFER_SIZE, BATCH_SIZE)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.target_alpha = -np.prod((self.actiondim,)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        self.actor = Actor(self.statedim, self.actiondim).to(self.device)

        self.vf = CriticV(self.statedim * self.num_agents).to(self.device)
        self.vf_target = CriticV(self.statedim * self.num_agents).to(self.device)
        self.vf_target.load_state_dict(self.vf.state_dict())

        self.qf1 = CriticQ(self.statedim * self.num_agents + self.actiondim * self.num_agents).to(self.device)
        self.qf2 = CriticQ(self.statedim * self.num_agents + self.actiondim * self.num_agents).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_A)
        self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=LR_Q)
        self.qf1_optimizer = optim.Adam(self.qf1.parameters(), lr=LR_Q)
        self.qf2_optimizer = optim.Adam(self.qf2.parameters(), lr=LR_Q)

        self.transition = []

        self.total_step = 0

        self.is_test = False
        
        if self.device.type == 'cpu':
            print('DEVICE USED', self.device.type)
        else:
            print('DEVICE USED', torch.cuda.device(torch.cuda.current_device()), torch.cuda.get_device_name(0))
    
    def do_step(self, state, max_speed, min_speed, test = False, batch = False):

        if self.total_step < INITIAL_RANDOM_STEPS and not self.is_test:
            selected_action = np.random.uniform(-1, 1, (len(state), self.actiondim))
        else:
            selected_action = []
            for i in range(len(state)):
                action = self.actor(torch.FloatTensor(state[i]).to(self.device))[0].detach().cpu().numpy()
                selected_action.append(action)
            selected_action = np.array(selected_action)
            selected_action = np.clip(selected_action, -1, 1)

        self.total_step += 1
        return selected_action.tolist()
    
    def setResult(self,episode_name, state, new_state, reward, action, done):       
        if not self.is_test:       
            self.transition = [state, action, reward, new_state, done]
            self.memory.store(*self.transition)

        if (len(self.memory) >  BATCH_SIZE and self.total_step > INITIAL_RANDOM_STEPS):
            self.update_model()
    
    def update_model(self):
        device = self.device

        samples = self.memory.sample_batch()

        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1,1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        new_action, log_prob = self.actor(state)

        # reshape the tensors from 3D (batch,statedim,n_agents) to 2D for centralized Critic and Value Network
        q_state = torch.reshape(state,(BATCH_SIZE,self.statedim*self.num_agents))
        q_next_state = torch.reshape(next_state,(BATCH_SIZE,self.statedim*self.num_agents))
        q_action = torch.reshape(action,(BATCH_SIZE,self.actiondim*self.num_agents))
        q_new_action = torch.reshape(new_action,(BATCH_SIZE,self.actiondim*self.num_agents))
        q_log_prob = torch.mean(log_prob,1)


        alpha_loss = ( -self.log_alpha.exp() * (log_prob + self.target_alpha).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha = self.log_alpha.exp()

        mask = 1 - done
        q1_pred = self.qf1(q_state, q_action)
        q2_pred = self.qf2(q_state, q_action)
        vf_target = self.vf_target(q_next_state)
        q_target = reward + GAMMMA * vf_target * mask
        qf1_loss = F.mse_loss(q_target.detach(), q1_pred)
        qf2_loss = F.mse_loss(q_target.detach(), q2_pred)

        v_pred = self.vf(q_state)
        q_pred = torch.min(
            self.qf1(q_state, q_new_action), self.qf2(q_state, q_new_action)
        )
        v_target = q_pred - alpha * q_log_prob
        v_loss = F.mse_loss(v_pred, v_target.detach())

        if self.total_step % POLICY_UPDATE_FREQUENCE== 0:
            advantage = q_pred - v_pred.detach()
            actor_loss = (alpha * q_log_prob - advantage).mean()

            # because advantage & q_log_prob are calculated with the entire set of actions
            # instead of local, this gradient calculates the gradient for the weights that
            # best influences the entire action space instead of local actions.
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._target_soft_update()
        else:
            actor_loss = torch.zeros(1)
        
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        qf_loss = qf1_loss + qf2_loss

        self.vf_optimizer.zero_grad()
        v_loss.backward()
        self.vf_optimizer.step()

        return actor_loss.data, qf_loss.data, v_loss.data, alpha_loss.data
    
    def load_models(self):
        self.actor.load_state_dict(torch.load("results/ccspactor.pt"))
        self.qf1.load_state_dict(torch.load("results/ccspqf1.pt"))
        self.qf2.load_state_dict(torch.load("results/ccspqf2.pt"))
        self.vf.load_state_dict(torch.load("results/ccspvf.pt"))
        self.vf_target.load_state_dict(torch.load("results/ccspvft.pt"))

    def save_models(self):
        torch.save(self.actor.state_dict(), "results/ccspactor.pt")
        torch.save(self.qf1.state_dict(), "results/ccspqf1.pt")
        torch.save(self.qf2.state_dict(), "results/ccspqf2.pt")
        torch.save(self.vf.state_dict(), "results/ccspvf.pt")       
        torch.save(self.vf_target.state_dict(), "results/ccspvft.pt")
    
    def _target_soft_update(self):
        for t_param, l_param in zip(
            self.vf_target.parameters(), self.vf.parameters()
        ):
            t_param.data.copy_(TAU * l_param.data + (1.0 - TAU) * t_param.data)

    def normalizeState(self, s_t, max_speed, min_speed):
        # distance to closest #NUMBER_INTRUDERS_STATE intruders
        for i in range(0, self.number_intruders_state):
            s_t[i] = (s_t[i]-MEANS[0])/(STDS[0]*2)

        # relative bearing to closest #NUMBER_INTRUDERS_STATE intruders
        for i in range(self.number_intruders_state, self.number_intruders_state*2):
            s_t[i] = (s_t[i]-MEANS[1])/(STDS[1]*2)

        # current dy intruder (from ownship frame of reference)
        for i in range(self.number_intruders_state*2, self.number_intruders_state*3):
            s_t[i] = (s_t[i]-MEANS[2])/(STDS[2]*2)
        
        # current dx intruder (from ownship frame of reference)
        for i in range(self.number_intruders_state*3, self.number_intruders_state*4):
            s_t[i] = (s_t[i]-MEANS[3])/(STDS[3]*2)
        
        # relative track with intruder
        for i in range(self.number_intruders_state*4, self.number_intruders_state*5):
            s_t[i] = (s_t[i])/(3.1415)

        if self.use_altitude:     
            # current speed
            s_t[self.number_intruders_state*6] = ((s_t[self.number_intruders_state*6]-min_speed)/(max_speed-min_speed))*2 - 1
            # optimal speed
            s_t[self.number_intruders_state*6 + 1] = ((s_t[self.number_intruders_state*6 + 1]-min_speed)/(max_speed-min_speed))*2 - 1

            # bearing to target
            s_t[self.number_intruders_state*6+3] = s_t[self.number_intruders_state*6+3]
            s_t[self.number_intruders_state*6+4] = s_t[self.number_intruders_state*6+4]
        else:
             # current speed
            s_t[self.number_intruders_state*5] = ((s_t[self.number_intruders_state*5]-min_speed)/(max_speed-min_speed))*2 - 1
            # optimal speed
            s_t[self.number_intruders_state*5 + 1] = ((s_t[self.number_intruders_state*5 + 1]-min_speed)/(max_speed-min_speed))*2 - 1
            
            # # bearing to target
            s_t[self.number_intruders_state*5+2] = s_t[self.number_intruders_state*5+2]
            s_t[self.number_intruders_state*5+3] = s_t[self.number_intruders_state*5+3]


        return s_t