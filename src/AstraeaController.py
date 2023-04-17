"""
Astraea main program. 
Astraea works in a continuous loop. As the application receives user traffic, trace observations are collected and stored in a database. 
Via TraceManager.py, Astraea periodically queries the database (or Jaeger API) to build Bayesian belief distributions and to estimate span-utility pairs using Monte Carlo sampling. 
Utility (e.g., Variance) represents the usefulness of a span for performance diagnosis; as more data is available, the belief distributions converge to the true values of the utility. 
At each iteration, via BayesianMethods.py, Astraea adjusts the sampling probabilities of spans using its multi-armed bandit algorithm, taking into account the belief distributions. 
The fraction of spans with low utility decreases over time as the fraction of the most rewarding spans increases.
 At the end of each iteration, span sampling probabilities are issued to the instrumented cloud application through AstraeaOrchestrator.py/.
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from matplotlib.pyplot import figure
import glob
import random
import time
from IPython.display import display
from configparser import SafeConfigParser
import random


import BayesianMethods as banditalg
import TraceManager as traceManager
import AstraeaOrchestrator as ao
from csv import writer


pd.set_option('display.max_colwidth', None)
pd.set_option("precision", 1)
pd.options.display.float_format = '{:.1f}'.format
sns.set(style='whitegrid', rc={"grid.linewidth": 0.1})
sns.set_context("paper", font_scale=2.2)                                                  
color = sns.color_palette("Set2", 2)


## Astraea parameters
reward_field = "Var_sum"
confidence = 0.95

parser = SafeConfigParser()
parser.read('../conf/astraea-config.ini')
period = int(parser.get('application_plane', 'Period'))
span_states_file_txt = parser.get('application_plane', 'SpanStatesFileTxt')

  
  
print("***** Welcome to Astraea!")
  
# Defining main function
def main():

    print("---- Astraea started!")

    time.sleep(period)

    # Astraea framework Initialized
    bandit = banditalg.ABE("ABE", "Experiment-id1", confidence=confidence, reward_field = reward_field)
    astraeaOrc = ao.AstraeaOrc()
    astraeaMan = traceManager.TraceManager()

    
    ## peridically run and 1) read traces
    epoch = 0


    while epoch < 100:
        epoch += 1
        print("\n\n\n\n\n---- runninng epoch: ", epoch)

        all_traces = astraeaMan.get_traces_jaeger_api(service = "compose-post-service")
        print("collected the batch with len: ", len(all_traces["data"]))

        ## parse traces and extract span units
        trace_parsed = astraeaMan.traces_to_df_asplos_experimental(all_traces["data"],application_name="SocialNetwork")
        df_traces = trace_parsed[0]

        display(df_traces.sort_values(by=reward_field, ascending=False))

        ## apply bayesian methods and get new sampling policy
        splits, sorted_spans = bandit.mert_sampling_median_asplos(df_traces, epoch)
        print("Check new sampling \n", splits)
        print("Check sorted spans\n",sorted_spans)

        # astraeaOrc.issue_sampling_policy(splits)
        astraeaOrc.issue_sampling_policy_txt(splits)

        print("Finished epoch ", epoch)
        time.sleep(period)
  
# __name__
if __name__=="__main__":
    main()
