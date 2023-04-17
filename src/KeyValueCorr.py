"""
Demonstration of pearson correlation analysis of key value attributes.
Distributed tracing helps pinpoint the code regions, which
initiate the performance issues. Traces additionally
include tags, such as RPC parameters, resource configurations,
etc., making it more powerful tool for debugging
tasks such as detecting resource contentions.

Astraea analyzes tag pairs via Pearson
product-moment correlation, measuring the strength of
relationships between pairs and end-to-end latency, denoted by
r. 

In this python file, you can find a demo case study on how tag pairs can be used for diagnosis. 
We include trace data that has a performance problem due to a congested host (compose-delay-hostname.json). 
Specifically, we use 2 hosts (containers) for compose-post service with one of them is congested via DTC tool. 
Docker Traffic Control (DTC): Emulate conditions like delay, packet loss, .. in given hosts.
Using the hostname tag available in traces, correlation analysis reveal that problem is due to specific host of a service (compose-post).
"""
import json
import time
from scipy.stats.stats import pearsonr   
import pandas as pd
from sklearn.preprocessing import LabelEncoder


### reading traces
def get_traces_jaeger_file(file_name):
    print("------")
    data = {}
    with open(file_name) as f:
        data = json.load(f)

    print(len(data["data"]))
    return data


time_nanosec_start = time.time_ns()

data_path = "data/"
ds = data_path+ "compose-delay-hostname.json"

unique_keys = set()
unique_key_values = {}
unique_key_values_processes = {}

data = [ds]
e2e_list = []
host_list =[]

key_att = {}
nuff_attr = []

for item in data:

    print("*** item, ", item)
    tt_traces = get_traces_jaeger_file(item)
    
    for trace in tt_traces["data"]:
        e2e_collected = False
        visited_duplicate_root = False
                
        for span in trace["spans"]:
            if span["operationName"] == "/wrk2-api/post/compose" and not(e2e_collected):
                e2e = span["duration"]
                e2e_list.append(e2e/1000)
                e2e_collected = True
#                 print("e2e: ", e2e)
                
                
        for span in trace["spans"]:  
            if span["operationName"] != "/wrk2-api/post/compose" or (span["operationName"] == "/wrk2-api/post/compose" and not(visited_duplicate_root)):
                if span["operationName"] == "/wrk2-api/post/compose":
                    visited_duplicate_root = True
                
                
                for tag in span["tags"]:
                    key_now = tag['key']
                    value_now = tag['value']
                    
                    if not value_now:
                        continue

                    op_name= span["operationName"]
                    processID = span["processID"]
                    svc_name = trace["processes"][processID]["serviceName"]
                   

                    if "sampler" not in key_now and "format" not in key_now:

                        key = svc_name + ";" + op_name + "_" + key_now + ":" + str(value_now)
                        nuff_attr.append((key,e2e))
#                      
                        if (svc_name + ";" +span["operationName"]+"_"+tag['key']) not in key_att:
                            key_att[svc_name + ";" +span["operationName"]+"_"+tag['key']] = []

                        key_att[svc_name + ";" +span["operationName"]+"_"+tag['key']].append(value_now)

                
#         break

                    
                
        for process in trace["processes"]:
#             print(process)
          
            for tag in trace["processes"][process]["tags"]:
                key_now = tag['key']
                value_now = tag['value']
                if (trace["processes"][process]["serviceName"]+"_"+tag['key']) not in key_att:
                    key_att[trace["processes"][process]["serviceName"]+"_"+tag['key']] = []

                key_att[trace["processes"][process]["serviceName"]+"_"+tag['key']].append(value_now)
                nuff_attr.append((trace["processes"][process]["serviceName"]+"_"+tag['key'] + ":" + value_now,e2e))
                
#         break
#                    

## merge them into a dataframe
data_pd= pd.DataFrame(e2e_list,columns=['e2e'])
for item in key_att:
    if "sampler" not in item and "format" not in item:
        data_pd[item] = key_att[item]
    
    
data_pd = data_pd.apply(LabelEncoder().fit_transform)

res = data_pd.drop("e2e", axis=1).apply(lambda x: x.corr(data_pd.e2e,method ='pearson'))
time_nanosec_end = time.time_ns()
print("Conducted in ", (time_nanosec_end-time_nanosec_start)/1000000)

# data
# print(data_pd)
print(res)
