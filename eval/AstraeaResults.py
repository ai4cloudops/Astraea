import requests
import json
import time;
import requests 
import concurrent.futures
from IPython.display import display, HTML
from collections import defaultdict
from scipy import stats
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
# from termcolor import colored
from matplotlib.pyplot import figure
import random

sns.set(style='whitegrid', rc={"grid.linewidth": 0.1})
sns.set_context("paper", font_scale=2)                                                  
color = sns.color_palette("Set2", 2)

#### Astreaea percentile evaluation

files_list= ["/Users/merttoslali/Desktop/fall21/IBM/tech/notebooks/Astraea-2022/AstraeaPaper/Experiments/new-experiments-e2e/socialNetwork/"] 

for file_outer in files_list:
    for file in glob.glob(file_outer+ '*'):
        print(file)

# df_sampling_tt = pd.read_csv("Data/astraea-runnig-cost/Astraea-eval-probabilistic-median-tt.csv")
