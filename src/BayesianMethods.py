"""
Our online learning framework encapsulates our
Bayesian online learning formulation and multi-armed bandit based
sampling algorithm. 
The former builds belief
distributions for span utility pairs, whereas the latter takes
the belief distributions as inputs and decides how
to adjust the sampling probabilities across spans.

Some useful links for mab review and comparison baselines used:
# https://courses.cs.washington.edu/courses/cse599i/18wi/resources/lecture4/lecture4.pdf
------------------------
MEDIAN ELIMINATION
https://jmlr.csail.mit.edu/papers/volume7/evendar06a/evendar06a.pdf
------------------------
HOEFFDING RACES 
https://papers.nips.cc/paper/1993/file/02a32ad2669e6fe298e607fe7cc0e1a0-Paper.pdf
------------------------
Exponential-Gap Elimination
http://proceedings.mlr.press/v28/karnin13.pdf
------------------------
"""
import numpy as np
from collections import deque
class ABE():
    
    """
    Class of ABE objects used for action elimination mechanism of Astraea. 
    Pseudo-code can be found in our paper.
    """

    def __init__(self, algorithm , experiment_id, confidence, reward_field, epsilon = 5, exp_lambda = 0.5, controlled_span = "None", elim_percentile = 0):
        self.name = algorithm
        self.experiment_id = experiment_id
        
        self.alpha_spans = {}
        self.beta_spans = {}
        self.span_estimates = {}
        self.init_set_spans = {}
        self.prev_traf = {}
        
        self.mc_sample = 1000
        self.max_RMEAN = 0
        self.topK = -1
        self.exploration = epsilon
        self.exp_lambda = exp_lambda
        self.confidence = confidence
        self.reward_field = reward_field
        self.controlled_span = controlled_span
        self.reward_moving_average_window = 10 ## last N periods
        self.max_reward_queue = deque([0]*self.reward_moving_average_window,maxlen=self.reward_moving_average_window)
        self.elim_percentile = elim_percentile
        
        
    def set_mc_sample(self, sample_count):
        self.mc_sample = sample_count
        
    def set_topK(self, topK):
        self.topK = topK
        
    def mert_sampling_median_asplos(self, df_traces, epoch):
        """
        Method of ABE elimination used for action elimination mechanism of Astraea. 
        Collects data, update belief distributions, apply monte carlo estimation and eliminate suboptimal spans.
        """
        
        if self.max_RMEAN == 0: #epoch == 0:
            ## estimate max reward to normalize 
#             self.max_RMEAN = df_traces[self.reward_field].max() * 2
            # self.max_RMEAN = df_traces[self.reward_field].max()
            ## update with moving average
            
#             self.max_RMEAN = 10313815
            self.init_set_spans = set(df_traces["Name"])            ## get span names
            ## construct estimates priors for spans
            self.alpha_spans = dict(zip(list(self.init_set_spans), np.repeat(1., len(self.init_set_spans))))
            self.beta_spans = dict(zip(list(self.init_set_spans), np.repeat(1., len(self.init_set_spans))))
            self.span_estimates = dict(zip(list(self.init_set_spans), np.repeat(0., len(self.init_set_spans))))
            print("--- Initialized ABE")
            

        ## maximum reward observed in last N periods 
        if len(df_traces) > 1:

            self.max_reward_queue.appendleft(df_traces[self.reward_field].max())

            print("max_reward_queue: ",self.max_reward_queue )
            max_rew_estimate = [i for i in self.max_reward_queue if i > 0]
            print("Max rew estimate ", max_rew_estimate )
            self.max_RMEAN = np.mean(max_rew_estimate)



        set_spans_now = set(df_traces["Name"])
        ## get difference with orig spans in the first round
        diff_now = set_spans_now.difference(self.init_set_spans)

        if diff_now: ## if there is different span, update our data
            print("----- There is diff: ", diff_now)
            self.init_set_spans.update( diff_now) ## update set with new elements if any
            alpha_new = dict(zip(list(diff_now), np.repeat(1., len(diff_now))))
            beta_new = dict(zip(list(diff_now), np.repeat(1., len(diff_now))))
            span_estimates_new = dict(zip(list(diff_now), np.repeat(0., len(diff_now))))

            self.alpha_spans = {**self.alpha_spans, **alpha_new}
            self.beta_spans = {**self.beta_spans, **beta_new}
            self.span_estimates = {**self.span_estimates, **span_estimates_new}
                
        ## beta updates
        # display(df_traces)
        for key in set_spans_now:
            if key in self.alpha_spans:
                temp_R = df_traces.loc[df_traces['Name'] == key][self.reward_field].iloc[0]
                count = df_traces.loc[df_traces['Name'] == key]["Count"].iloc[0]
#                 print(key, temp_R, self.max_RMEAN,temp_R/self.max_RMEAN ,(self.max_RMEAN - temp_R)/self.max_RMEAN)
                if temp_R < 0:
                    temp_R = 0
                elif temp_R > self.max_RMEAN:
                    temp_R = self.max_RMEAN
                
                ##batched update

                # successes = np.sum(np.random.binomial(1, temp_R/self.max_RMEAN,1)) ## update by 1 no matter the sample size
                successes = np.sum(np.random.binomial(1, temp_R/self.max_RMEAN,int(count/30)+1)) ## for each 30 samples, we update by 1
                failures = int(count/30) + 1 -successes

                print(key, "rew", "{:.2f}".format(temp_R), "rew_rat", "{:.3f}".format(temp_R/self.max_RMEAN), 
                    "cnt", count, "scs", successes, "fail", failures,
                    "alpha", self.alpha_spans[key], "beta", self.beta_spans[key])
#                 successes = np.sum(np.random.binomial(1, temp_R/self.max_RMEAN,count)) ## update by N for N samples
                
                ## exponentially weighte updates
#                 self.alpha_spans[key] = (1-self.exp_lambda)*self.alpha_spans[key]  + self.exp_lambda * successes 
                # (temp_R*count)/self.max_RMEAN              
#                 self.beta_spans[key] = (1-self.exp_lambda)* self.beta_spans[key]  + self.exp_lambda * (count-successes)
                #(self.max_RMEAN*count - temp_R*count)/self.max_RMEAN
    
                ## normal updates with simulating fraction of data
                
                ## update by count
                self.alpha_spans[key] += successes  #* (self.prev_traf.get(key,1)/100)
                self.beta_spans[key] += failures  #* (self.prev_traf.get(key,1)/100)


        ### estimate median, means//
#         print(self.alpha_spans)
        ### we conduct thompson sampling and construct probabilities of spans
        means = []
        sorted_keys ={}
        for key in self.alpha_spans:
            self.span_estimates[key] = np.random.beta(self.alpha_spans[key],self.beta_spans[key], self.mc_sample)
            means.append(np.mean(self.span_estimates[key]))
            sorted_keys[key] = np.mean(self.span_estimates[key])
        ## sort results and return
        sorted_keys = {k: v for k, v in sorted(sorted_keys.items(), key=lambda item: item[1],reverse=True)}
        
        ## If zero, mean
        if self.elim_percentile == 0:
            print("Calculating mean")
            median = np.mean(means)
        else: ## else given percentile!
            print("Calculating percentile ", self.elim_percentile)
            median = np.percentile(means, self.elim_percentile)
        
        ## sort estimates by value
        any_removed = []
        traffic_split = {}
        
        ### calculate the fraction of the times estimates are greater than median
        for key in sorted_keys:
            times_more_med = (self.span_estimates[key]> median).sum() 
            ## ratio of times_more/ total samples
            traffic_split[key] = (times_more_med*100) / self.mc_sample
            if traffic_split[key] < self.exploration:
                traffic_split[key] = self.exploration
        self.prev_traf = traffic_split
                
            
        
        #### derive 2d np-array and find max per each column
        
#         traffic_split_2d = np.array([self.span_estimates[key] for key in self.alpha_spans ])
#         spans = [span_name for span_name in self.span_estimates]
# #         print("Traffic: ", traffic_split_2d)
# #         print(f"Shape: {traffic_split_2d.shape}")
#         row, column = traffic_split_2d.shape
#         epsilon_matrix = (np.random.rand(row,column) - 0.5)*1e-20
#         traffic_split_2d += epsilon_matrix
#         #print(traffic_split_2d)
#         max_indices = np.argmax(traffic_split_2d, axis=0)
# #         print(f"max ind: {max_indices}")
        
#         # get occurences number of max indices to derive traffic split
#         unique, counts = np.unique(max_indices, return_counts=True)
#         split = dict(zip(unique, counts*100/self.mc_sample))
# #         print(f"*harikasplit: {split}")
        
#         for index,value in split.items():
#             traffic_split[spans[index]]= value
#         print(f"*res split: {traffic_split}")    

        return traffic_split, sorted_keys

class ME():

    def __init__(self, algorithm , experiment_id,reward_field, delta = 0.15, epsilon=0.155 ):
        self.name = algorithm
        self.experiment_id = experiment_id
        
        self.arm_mean_estimates = {}
        self.init_set_spans = {}
        
        self.max_RMEAN = 0
        self.topK = -1

        self.reward_field = reward_field
        self.delta = delta/2
        self.epsilon = epsilon/4
        
        self.pull_prev = 0
        self.arm_sample_counts = {} 
        
    def set_topK(self, topK):
        self.topK = topK
        
    def check_number_of_samples(self,delta_l,epsilon_l, arm_sample_counter):
        num_samples = int(np.log(3 / delta_l) * 2 / (epsilon_l ** 2))
        min_num_samples_sofar = min(arm_sample_counter.values())
        
        if min_num_samples_sofar >= num_samples:
            print("-------Min num samples: " , min_num_samples_sofar)
            return True
        else:
            return False
        
    ## k = number of arms
    def median_elimination_asplos(self, df_traces, epoch):
        
        if epoch == 0:
            ### we normalize the variable by max (i.e., sample/max for it to be between 0-1)
            if "R*" in self.reward_field: ### to get maxc value -- specific condition for R
                self.max_RMEAN = df_traces[self.reward_field[2:]].max() * 1 
            else:
                self.max_RMEAN = df_traces[self.reward_field].max() * 2

            self.init_set_spans = set(df_traces["Name"])
            self.arm_mean_estimates = dict(zip(list(self.init_set_spans), np.repeat(0., len(self.init_set_spans))))    
            self.arm_sample_counts = dict(zip(list(self.init_set_spans), np.repeat(0., len(self.init_set_spans))))
        
        
        if self.topK >= len(self.arm_mean_estimates):
            print("*** Not running the algorithm as TOP K found")
            return
            
            
        ## get difference (new spans) with orig spans in the first round
        set_spans_now = set(df_traces["Name"])
        diff_now = set_spans_now.difference(self.init_set_spans)
        
        if diff_now: ## if there is different, update our data
            self.init_set_spans.update( diff_now) ## update set with new elements if any
            span_estimates_new = dict(zip(list(diff_now), np.repeat(0., len(diff_now))))
            span_sample_count_new = dict(zip(list(diff_now), np.repeat(0., len(diff_now))))
            
            self.arm_mean_estimates = {**self.arm_mean_estimates, **span_estimates_new}
            self.arm_sample_counts = {**self.arm_sample_counts, **span_sample_count_new}
        
        
        ## update arm estimates 
#         print(self.arm_mean_estimates)
#         display(df_traces)
        for key in self.arm_mean_estimates:
            temp_R = df_traces.loc[df_traces['Name'] == key][self.reward_field].iloc[0]
            count = df_traces.loc[df_traces['Name'] == key]["Count"].iloc[0]
            ## special conditions to crop
            if temp_R < 0:
                temp_R = 0
            elif temp_R > self.max_RMEAN:
                temp_R = self.max_RMEAN
                
            self.arm_mean_estimates[key] +=  (temp_R*count)/self.max_RMEAN ## update mean estimates
            self.arm_sample_counts[key] += count ## update number of samples per span/arm

        any_removed = []    
        ### if sampled enough, eliminate worst half by median    
        if self.check_number_of_samples(self.delta, self.epsilon, self.arm_sample_counts): ## samples : int(np.log(3 / delta_l) * 2 / (epsilon_l ** 2))
            
            
            median = np.median(list(self.arm_mean_estimates.values())) # finding median of mean estimates
            for key, value in self.arm_mean_estimates.items():
                if value < median:
                    # finding which arm to remove
                    any_removed.append(key)

            ## remove them from estimator
            for item in any_removed:
                self.arm_mean_estimates.pop(item)   
                self.arm_sample_counts.pop(item)


            self.epsilon *= 3 / 4
            self.delta *= 1 / 2
            
            ## reset number of samples for all
            self.arm_sample_counts = dict(zip(self.arm_mean_estimates.keys(),np.repeat(0.,len(self.arm_mean_estimates.keys()))))
        
        
        return any_removed

    
## k = number of arms
def median_elimination(k=10,epsilon=0.9, delta=0.9):
    results = []
    error = 0
    epsilon_l = epsilon / 4
    delta_l = delta / 2

    n_pulls=2000 # total number of times to pull each arm
    
    q_true= np.random.uniform(0, 1,k) # generating the true means q*(a) for each arm for all bandits
    
    true_opt_arms=np.argmax(q_true) # the true optimal arms in each bandit
    print(true_opt_arms)
    eliminated_arms = []
    
    
    arm_mean_estimates = dict(zip(list(set(np.arange(k))), np.repeat(0., len(set(np.arange(k))))))
    
    def check_number_of_samples(delta_l,epsilon_l, pull_sofar):
        num_samples = int(np.log(3 / delta_l) * 2 / (epsilon_l ** 2))
        if pull_sofar >= num_samples:
            return True
        else:
            return False
        
    pull_prev = 0
    
    
    
#     print("qtrue", q_true)
#     print("true_opt_arms", true_opt_arms)
    
    ### steps for the algorithm
    for pull in range(2,n_pulls+1) : 
#         print("+",pull)
        results.append(['ME', pull, len(arm_mean_estimates)])

        ## sample all arms
        for key in arm_mean_estimates:
            temp_R= np.random.binomial(1, q_true[key])
            arm_mean_estimates[key] += temp_R ## update mean estimates
            
        
        ### if sampled enough, eliminate worst half by median    
        if check_number_of_samples(delta_l, epsilon_l, pull-pull_prev):     # # of samples required for each arm ---> int(np.log(3 / delta_l) * 2 / (epsilon_l ** 2))
#             print("checking")
#             print(delta_l, epsilon_l, pull-pull_prev)
            pull_prev = pull
            any_removed = []
            median = np.median(list(arm_mean_estimates.values())) # finding median of mean estimates
            for key, value in arm_mean_estimates.items():
                if value < median:
                    eliminated_arms.append(key) # finding which arm to remove
#                     print("******** eliminated", key)
                    any_removed.append(key)
                    
                    
            for item in any_removed:
                arm_mean_estimates.pop(item)   
                if item == true_opt_arms:
                    error +=1
                

            epsilon_l *= 3 / 4
            delta_l *= 1 / 2

#         print(arm_mean_estimates)
    print(arm_mean_estimates)
    return results

           
'''
------------------------
HOEFFDING RACES 
https://papers.nips.cc/paper/1993/file/02a32ad2669e6fe298e607fe7cc0e1a0-Paper.pdf
------------------------
'''

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import time

def racing_algorithm(k=3, c=2):
    B = 1 ## possible spread of results
    results = []
    error = 0
    n_pulls=2000 # number of times to pull each arm
    q_true= np.random.uniform(0, 1,k) # generating the true means q*(a) for
    # q_true = [0.9,1 ,0]
#     print("qtrue", q_true)
    true_opt_arms=np.argmax(q_true) # the true optimal arms in each bandit
    # print("true_opt_arms", true_opt_arms)
    # Q=np.zeros(k) # reward estimated
    # N=np.ones(k) # number of times each arm was pulled # each arm is pull
    eliminated_arms = []
    
    Q = dict(zip(list(set(np.arange(k))),np.repeat(0., len(set(np.arange(k))))))
    N = dict(zip(list(set(np.arange(k))), np.repeat(0., len(set(np.arange(k))))))
#     print("Q",Q)
#     print("N",N)
    for pull in range(2,n_pulls+1) :
        results.append(['Racing', pull, len(Q)])
        for key in Q:
            temp_R= np.random.binomial(1, q_true[key])
            N[key]=N[key]+1
            Q[key]=Q[key]+(temp_R-Q[key])/N[key]
        ucb_Q = np.array(list(Q.items()))
        ucb_Q_lower = np.array(list(Q.items()))

        
        ### HOEFFDING EQUATION
#         ucb_Q[:,1] += np.sqrt(c*np.log(pull)/np.array(list(N.values())))
#         ucb_Q_lower[:,1] -= np.sqrt(c*np.log(pull)/np.array(list(N.values())))
        
        ## ORIG HOEFFIND
        ucb_Q[:,1] += np.sqrt( (B**2 * np.log(40)) /(2*np.array(list(N.values()) ) ))
        ucb_Q_lower[:,1] -= np.sqrt( (B**2 * np.log(40)) /(2*np.array(list(N.values()) ) ))
        
        
        
#         if pull % 20 == 0:
#             print("ucb_Q",ucb_Q)
#             print("ucb_Q_lower",ucb_Q_lower)
        
        worst_arm_index = np.argmin(ucb_Q[:,1])
        worst_arm = ucb_Q[worst_arm_index]
        
        ucb_Q_lower = np.delete(ucb_Q_lower, worst_arm_index,0)
        second_worst_arm_index = np.argmin(ucb_Q_lower[:,1])
        second_worst_arm = ucb_Q_lower[second_worst_arm_index]
                
        if worst_arm[1] < second_worst_arm[1]:
            print("*********** eliminated")
#             raise NotImplementedError
            eliminated_arms.append(worst_arm[0])
            Q.pop(worst_arm[0])
            N.pop(worst_arm[0])
            if worst_arm[0] == true_opt_arms:
                error +=1
            if len(Q) < 2:
                break
            
    return results


### Exponential-Gap Elimination

'''
------------------------
Exponential-Gap Elimination
http://proceedings.mlr.press/v28/karnin13.pdf
------------------------
'''

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import time


class EGE():
    def __init__(self, algorithm , experiment_id,reward_field, delta = 0.9):
        self.name = algorithm
        self.experiment_id = experiment_id
        
        self.arm_mean_estimates = {}
        self.init_set_spans = {}
        
        self.max_RMEAN = 0
        self.topK = -1

        self.reward_field = reward_field
        
        self.r
        self.delta = delta/(50 * pow(self.r,3))
        self.epsilon  = 1 / (4* pow(2,self.r))
        
        self.pull_prev = 0
        self.arm_sample_counts = {} 
        
    def set_topK(self, topK):
        self.topK = topK
        
    def check_number_of_samples(self,delta_l,epsilon_l, arm_sample_counter):
        num_samples = int(np.log(2 / delta_l) * 2 / (epsilon_l ** 2))
        min_num_samples_sofar = min(arm_sample_counter.values())
        
        if min_num_samples_sofar >= num_samples:
            print("-------Min num samples: " , min_num_samples_sofar)
            return True
        else:
            return False
        
        
    def exponential_gap_elimination_asplos():
        
        if epoch == 0:
            ### we normalize the variable by max (i.e., sample/max for it to be between 0-1)
            if "R*" in self.reward_field: ### to get maxc value -- specific condition for R
                self.max_RMEAN = df_traces[self.reward_field[2:]].max() * 1 
            else:
                self.max_RMEAN = df_traces[self.reward_field].max() * 2

            self.init_set_spans = set(df_traces["Name"])
            self.arm_mean_estimates = dict(zip(list(self.init_set_spans), np.repeat(0., len(self.init_set_spans))))    
            self.arm_sample_counts = dict(zip(list(self.init_set_spans), np.repeat(0., len(self.init_set_spans))))
        
        
        if self.topK >= len(self.arm_mean_estimates):
            print("*** Not running the algorithm as TOP K found")
            return
            
            
        ## get difference (new spans) with orig spans in the first round
        set_spans_now = set(df_traces["Name"])
        diff_now = set_spans_now.difference(self.init_set_spans)
        
        if diff_now: ## if there is different, update our data
            self.init_set_spans.update( diff_now) ## update set with new elements if any
            span_estimates_new = dict(zip(list(diff_now), np.repeat(0., len(diff_now))))
            span_sample_count_new = dict(zip(list(diff_now), np.repeat(0., len(diff_now))))
            
            self.arm_mean_estimates = {**self.arm_mean_estimates, **span_estimates_new}
            self.arm_sample_counts = {**self.arm_sample_counts, **span_sample_count_new}
        
        
        ## update arm estimates 
        for key in self.arm_mean_estimates:
            temp_R = df_traces.loc[df_traces['Name'] == key][self.reward_field].iloc[0]
            count = df_traces.loc[df_traces['Name'] == key]["Count"].iloc[0]
            ## special conditions to crop
            if temp_R < 0:
                temp_R = 0
            elif temp_R > self.max_RMEAN:
                temp_R = self.max_RMEAN
                
            self.arm_mean_estimates[key] +=  (temp_R*count)/self.max_RMEAN ## update mean estimates
            self.arm_sample_counts[key] += count ## update number of samples per span/arm

        any_removed = []    
        ### if sampled enough, eliminate worst half by median    
        if self.check_number_of_samples(self.delta, self.epsilon, self.arm_sample_counts): ## samples : int(np.log(3 / delta_l) * 2 / (epsilon_l ** 2))
            
            
            median = np.median(list(self.arm_mean_estimates.values())) # finding median of mean estimates
            for key, value in self.arm_mean_estimates.items():
                if value < median:
                    # finding which arm to remove
                    any_removed.append(key)

            ## remove them from estimator
            for item in any_removed:
                self.arm_mean_estimates.pop(item)   
                self.arm_sample_counts.pop(item)

                
            self.r = self.r + 1

            self.epsilon = 1 / (4* pow(2,self.r))
            self.delta = self.delta/(50 * pow(self.r,3))
            
            ## reset number of samples for all
            self.arm_sample_counts = dict(zip(self.arm_mean_estimates.keys(),np.repeat(0.,len(self.arm_mean_estimates.keys()))))
        
        
        return any_removed
        
        

    def exponential_gap_elimination(k=3,delta = 0.05):
        results = []
        r = 1
        epsilon = 1 / (4* pow(2,r))
        delta = delta/(50 * pow(r,3))

        n_pulls=2000 # total number of times to pull each arm

        q_true= np.random.uniform(0, 1,k) # generating the true means q*(a) for each arm for all bandits
        true_opt_arms=np.argmax(q_true) # the true optimal arms in each bandit
        eliminated_arms = []

        arm_mean_estimates = dict(zip(list(set(np.arange(k))), np.repeat(0., len(set(np.arange(k))))))

        def check_number_of_samples(delta,epsilon, pull_sofar):
            num_samples = int(np.log(2 / delta) * 2 / (epsilon ** 2))
            if pull_sofar >= num_samples:
                return True
            else:
                return False

        pull_prev = 0


        ### steps for the algorithm
        for pull in range(2,n_pulls+1) : 
    #         print("+",pull)
            results.append(['EGE', pull, len(arm_mean_estimates)])

            ## sample all arms
            for key in arm_mean_estimates:
                temp_R= np.random.binomial(1, q_true[key])
                arm_mean_estimates[key] += temp_R ## update mean estimates


            ### if sampled enough, eliminate worst half by median    
            if check_number_of_samples(delta, epsilon, pull-pull_prev):     # # of samples required for each arm ---> int(np.log(3 / delta_l) * 2 / (epsilon_l ** 2))
    #             print("checking")
    #             print(delta_l, epsilon_l, pull-pull_prev)
                pull_prev = pull
                any_removed = []
                median = np.median(list(arm_mean_estimates.values())) # finding median of mean estimates
                for key, value in arm_mean_estimates.items():
                    if value < median:
                        eliminated_arms.append(key) # finding which arm to remove
    #                     print("******** eliminated", key)
                        any_removed.append(key)


                for item in any_removed:
                    arm_mean_estimates.pop(item)   
                    if item == true_opt_arms:
                        error +=1

                epsilon = 1 / (4* pow(2,r))
                delta = delta/(50 * pow(r,3))
                r

    #         print(arm_mean_estimates)
        return results
