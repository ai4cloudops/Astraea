"""
Astraea evaluator program. 
This program runs demo experiments. 
1) Run continous workload (hey or work), 2) inject problems, 3) runs Astraea, 4) collects evaluation data.
Evaluation data:
a) Traffic split of spans, b) traces sizes after each period
"""

import sys
sys.path.append('../src/')

import AstraeaControllerEval as ace
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
import os
import subprocess
import atexit
import signal
import logging
import sys
import fileinput

logging.basicConfig(
                stream = sys.stdout,
                level = logging.INFO)

logger = logging.getLogger()

pd.set_option('display.max_colwidth', None)
pd.set_option("precision", 1)
pd.options.display.float_format = '{:.1f}'.format
sns.set(style='whitegrid', rc={"grid.linewidth": 0.1})
sns.set_context("paper", font_scale=2.2)                                                  
color = sns.color_palette("Set2", 2)


parser = SafeConfigParser()
parser.read('../conf/eval-config.ini')

period = int(parser.get('experimentation_plane', 'Period'))
totalExpDuration = int(parser.get('experimentation_plane', 'ExperimentDuration'))

samplingPolicy = parser.get('experimentation_plane', 'SpanStatesFileTxt')
samplingPolicyDefault = parser.get('experimentation_plane', 'SpanStatesOrigFileTxt')
sleepPath = parser.get('experimentation_plane', 'SleepInjectionPath')


qps = int(parser.get('experimentation_plane', 'QPS'))
workers = int(parser.get('experimentation_plane', 'Workers'))


resultDir = parser.get('experimentation_plane', 'ResultDir')
elimPercentile = int(parser.get('experimentation_plane', 'EliminationPercentile'))

epsilon = int(parser.get('experimentation_plane', 'Epsilon'))

app = parser.get('experimentation_plane', 'Application')

## app specific parameter
if app == "SocialNetwork":
    all_spans = open("../data/{}".format(parser.get('experimentation_plane', 'AllSpansSN')), "r")
    workloadPath = parser.get('experimentation_plane', 'WorkloadGeneratorPathSN')
elif app == "Media":
    all_spans = open("../data/{}".format(parser.get('experimentation_plane', 'AllSpansMedia')), "r")
    workloadPath = parser.get('experimentation_plane', 'WorkloadGeneratorPathMedia')
else:
    all_spans = open("../data/{}".format(parser.get('experimentation_plane', 'AllSpansTT')), "r")
    workloadPath = parser.get('experimentation_plane', 'WorkloadGeneratorPathTT')


all_spans_list = all_spans.read().split("\n")
logger.debug(str(all_spans_list))


logger.info("***** Welcome to Astraea evaluator!")

  
process = None
def exit_handler():
    logger.info('ASTRAEA is ending!')
    os.killpg(0, signal.SIGKILL)
 
    logger.warning("Killed")


def epsilon_modifier_spans(problem):
    for line in fileinput.input([samplingPolicy], inplace=True):
        if line.strip().startswith(problem):
            line = "{} {}\n".format(problem,epsilon) 
        sys.stdout.write(line)

def workloadGenerator():
    if app == "SocialNetwork":
        cmd_wrk = "{}/wrk -D exp -t {} -c {} -d {} -L -s {}/scripts/social-network/compose-post.lua http://localhost:8080/wrk2-api/post/compose -R {}".format(workloadPath, workers,workers, totalExpDuration+100, workloadPath, qps)
        logger.info(str(("Sending req in background: ", cmd_wrk)))
        process = subprocess.Popen(cmd_wrk, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        ### Inject problem to deathstar
        problem_now = random.choice(all_spans_list)
        cmd_inject = "echo {} > {}".format(problem_now,sleepPath)
        logger.info(str(("Injecting problem: ", cmd_inject)))
        os.system(cmd_inject)
        logger.info(str(("Check content now: ", os.system("head -n 3 {}".format(sleepPath)))))
        return problem_now

    elif app == "TrainTicket":
        ## generate problem
        problem_now = random.choice(all_spans_list)
        problem_now = problem_now.split(" ")[0]

        ## run epsilon experiment (remove this later)   
        epsilon_modifier_spans(problem_now)
        logger.info(str(("Epsilon check ", os.system("cat {} | grep {}".format(samplingPolicy, problem_now)))))


        ## start running workload
        cmd_wrk = "workload/tt-heyworkload.sh {} {} {} {}".format(workloadPath,totalExpDuration+100,qps//5,workers)
        logger.info(str(("Sending tt req in background: ", cmd_wrk)))
        process = subprocess.Popen(cmd_wrk, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        ### Inject problem to tt
        cmd_inject = "echo {} > {}".format(problem_now,sleepPath)
        logger.info(str(("Injecting problem: ", cmd_inject)))
        os.system(cmd_inject)
        logger.info(str(("Check content now: ", os.system("head -n 3 {}".format(sleepPath)))))
        return problem_now
        

# Defining main function
def main():
    
    atexit.register(exit_handler)

    logger.info(str(("---- Astraea evaluation started! Parameters are a) period: ", period, " b) total: ", totalExpDuration)))

    # ## first make sure sampling policy is revert to default (100 per span)
    cmd_cp = "cp {} {}".format(samplingPolicyDefault, samplingPolicy)
    logger.info(str(("sampling policy is revert to default: ", cmd_cp)))
    os.system(cmd_cp)
    logger.info(str(("Check it out ", os.system("head -n 3 {}".format(samplingPolicy)))))

    ## start sending requests
    problem_now = workloadGenerator()

    ## sleep for a bit
    time.sleep(period + 15)

    logger.info("Woke up and running Astraea Controller")

    ## run Astraea and collect stats with problem_now
    astraeaCont = ace.AstraeaControllerEval()
    astraeaCont.run_with_evaluator(problem_now, totalExpDuration,resultDir,elimPercentile)


# __name__
if __name__=="__main__":



    os.setpgrp() # create new process group, become its leader
    main()
    # try:
    #     main()
    # finally:
    #     os.killpg(0, signal.SIGKILL) # kill all processes in my group