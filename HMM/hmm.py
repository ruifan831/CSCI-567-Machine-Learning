from __future__ import print_function
import json
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_{t-1} = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: A dictionary mapping each observation symbol to its index 
        - state_dict: A dictionary mapping each state to its index
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array where alpha[i, t-1] = P(Z_t = s_i, X_{1:t}=x_{1:t})
                 (note that this is alpha[i, t-1] instead of alpha[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        alpha = np.zeros([S, L])
        ######################################################
        # TODO: compute and return the forward messages alpha
        ######################################################
        for key, value in self.state_dict.items():
            alpha[value, 0] = self.pi[value] * \
                self.B[value, self.obs_dict[Osequence[0]]]

        for i in range(1, L):
            for key, value in self.state_dict.items():
                alpha[value, i] = self.B[value, self.obs_dict[Osequence[i]]] * sum(
                    [self.A[index, value]*alpha[index, i-1] for index in self.state_dict.values()])
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array where beta[i, t-1] = P(X_{t+1:T}=x_{t+1:T} | Z_t = s_i)
                    (note that this is beta[i, t-1] instead of beta[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        beta = np.zeros([S, L])
        #######################################################
        # TODO: compute and return the backward messages beta
        #######################################################
        for key, value in self.state_dict.items():
            beta[value, L-1] = 1

        for i in range(0, L-1)[::-1]:
            for key, value in self.state_dict.items():
                beta[value, i] = sum([self.A[value, index] * self.B[index, self.obs_dict[Osequence[i+1]]]
                                      * beta[index, i+1] for index in self.state_dict.values()])
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(X_{1:T}=x_{1:T})
        """

        #####################################################
        # TODO: compute and return prob = P(X_{1:T}=x_{1:T})
        #   using the forward/backward messages
        #####################################################
        T = len(Osequence)
        return sum(self.forward(Osequence)[:, T-1]*self.backward(Osequence)[:, T-1])

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - gamma: (num_state*L) A numpy array where gamma[i, t-1] = P(Z_t = s_i | X_{1:T}=x_{1:T})
                           (note that this is gamma[i, t-1] instead of gamma[i, t])
        """
        ######################################################################
        # TODO: compute and return gamma using the forward/backward messages
        ######################################################################
        temp = self.forward(Osequence)*self.backward(Osequence)
        return temp/np.sum(temp, axis=0)

    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array where prob[i, j, t-1] = 
                    P(Z_t = s_i, Z_{t+1} = s_j | X_{1:T}=x_{1:T})
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        #####################################################################
        # TODO: compute and return prob using the forward/backward messages
        #####################################################################
        forward = self.forward(Osequence)
        backward = self.backward(Osequence)
        sequence_prob = self.sequence_prob(Osequence)
        for i in self.state_dict.values():
            for j in self.state_dict.values():
                for t in range(L-1):
                    prob[i, j, t] = forward[i, t] * self.A[i, j] * self.B[j, self.obs_dict[Osequence[t+1]]] * backward[j,t+1] / sequence_prob
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden states (return actual states instead of their indices;
                    you might find the given function self.find_key useful)
        """
        path = []
        ################################################################################
        # TODO: implement the Viterbi algorithm and return the most likely state path
        ################################################################################
        S = len(self.pi)
        L = len(Osequence)
        opt = np.zeros([S, L])
        paths = np.zeros([S, L], dtype="int")
        for s in self.state_dict.values():
            opt[s, 0] = self.pi[s]* self.B[ s,self.obs_dict[Osequence[0]]]

        for t in range(1, L):
            for s in self.state_dict.values():
                temp = [self.A[s_pre, s]*opt[s_pre, t-1] for s_pre in range(S)]
                opt[s, t] = self.B[s, self.obs_dict[Osequence[t]]] * max(temp)
                paths[s, t] = np.argmax(temp)
        temp = np.argmax(opt[:, L-1])
        path.append(self.find_key(self.state_dict, temp))
        for i in range(1, L)[::-1]:
            temp = paths[temp, i]
            path.append(self.find_key(self.state_dict, temp))
        return path[::-1]

    # DO NOT MODIFY CODE BELOW

    def find_key(self, obs_dict, idx):
        for item in obs_dict:
            if obs_dict[item] == idx:
                return item

    def find_item(self, Osequence):
        O = []
        for item in Osequence:
            O.append(self.obs_dict[item])
        return O