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
from statistics import mean
import numpy as np
import os
import logging
logger = logging.getLogger(__name__)


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

app = parser.get('application_plane', 'Application')
service = parser.get('application_plane', 'Endpoint')
  
  
# print("***** Welcome to Astraea!")

class AstraeaControllerEval():
    def __init__(self):
        logger.info("Init!")

    def append_to_csv(self, resultDir, file_name, list_data):

        CHECK_FOLDER = os.path.isdir(resultDir)
        # If folder doesn't exist, then create it.
        if not CHECK_FOLDER:
            os.makedirs(resultDir)
            logger.info(str(("created folder : ", resultDir)))

        # The data assigned to the list e.g., list_data=[['03','Smith','Science'], ...]
        # First, open the old CSV file in append mode, hence mentioned as 'a'
        # Then, for the CSV file, create a file object
        with open(file_name, 'a+', newline='') as f_object:  
            # Pass the CSV  file object to the writer() function
            writer_object = writer(f_object)
            # Pass the data in the list as an argument into the writerow() function
            for line in list_data:
                writer_object.writerow(line)  
            # Close the file object
            f_object.close()

    def run_with_evaluator(self, problem_now,totalExpDuration,resultDir="result", elim_percentile = 0):
        

        time.sleep(period)

        # Astraea framework Initialized
        experimentID = "Experiment-{}-{}".format(problem_now, random.randint(0,99))
        logger.info(str(("---- Astraea {} evaluator started!".format(experimentID))   ))

        bandit = banditalg.ABE("ABE", experimentID, confidence=confidence, reward_field = reward_field, elim_percentile = elim_percentile)
        astraeaOrc = ao.AstraeaOrc()
        astraeaMan = traceManager.TraceManager()

        
        ## peridically run and 1) read traces, 2) collect stats
        epoch = 0
        samples_so_far = 0

        totalEpoch = int(totalExpDuration/period)
        logger.info(str(("---- Will run for total of epoch : ", totalEpoch)))

        while epoch < totalEpoch:
            epoch += 1
            logger.info(str(("---- runninng epoch: ", epoch)))

            all_traces = astraeaMan.get_traces_jaeger_api(service = service) #"compose-post-service"
            logger.info(str(("collected the batch with len: ", len(all_traces["data"]))))


            ## parse traces and extract span units
            # trace_parsed = astraeaMan.traces_to_df_asplos_experimental(all_traces["data"],application_name="SocialNetwork")
            trace_parsed = astraeaMan.traces_to_df_with_self(all_traces["data"],application_name=app, all_enabled=False)
            
            df_traces = trace_parsed

            samples_so_far += len(all_traces["data"])

            display(df_traces.sort_values(by=reward_field, ascending=False))
                

            ## apply bayesian methods and get new sampling policy
            splits, sorted_spans = bandit.mert_sampling_median_asplos(df_traces, epoch)
            # print("Check new sampling \n", splits)
            # print("Check sorted spans\n",sorted_spans)

            # astraeaOrc.issue_sampling_policy(splits)
            astraeaOrc.issue_sampling_policy_txt(splits)

            logger.info(str(("Finished epoch ", epoch)))


            ### collect stats
            ### trace sizes
            span_counts = []
            for trace in all_traces["data"]:
                span_counts.append([len(trace["spans"]), epoch])

            self.append_to_csv(resultDir, "{}/{}-tracesizes.csv".format(resultDir, experimentID.replace("/","%2F")), span_counts)
            logger.info(str(("******* Saved trace sizes with mean ", np.mean(span_counts, axis=0))))


            ### sampling policy
            sampling_policies = []
            with open(span_states_file_txt) as samplingPolicy:
                for line in samplingPolicy:
                    name, var = line.partition(" ")[::2]
                    sampling_policies.append([name.strip(), float(var), epoch, samples_so_far])
                    if name.strip() == problem_now:
                        logger.info(str(("******* Problem's sampling policy ",name.strip(), " : ", var)))
            

            self.append_to_csv(resultDir, "{}/{}-probability.csv".format(resultDir, experimentID.replace("/","%2F")), sampling_policies)
            logger.info("Saved sampling probabilities")

            time.sleep(period)
