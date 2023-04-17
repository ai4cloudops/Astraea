## READ traces

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
from sklearn.metrics import explained_variance_score
import random
from scipy.stats import norm, kurtosis, pearsonr
from kneed import KneeLocator, DataGenerator as dg
import random
from configparser import SafeConfigParser
from collections import deque 
import logging
logger = logging.getLogger(__name__)

class TraceManager():
    def __init__(self):
        parser = SafeConfigParser()
        parser.read('../conf/astraea-config.ini')
        self.JaegerAPIEndpoint = parser.get('application_plane', 'JaegerAPIEndpoint')
        self.period = parser.get('application_plane', 'Period')

        self.concurrent_children = {} ### key and list of children names e.g., "spanA" : [{children: ("spanB", "spanC", "spanD"), max = [12,11,10,11,11]
        self.children_moving_window = 10
        self.immediate_parent = {} ## "spanChild" : ["spanParent", "spanParen2"]

    ## get traces from API, given service and lookback period
    def get_traces_jaeger_api(self,service = "compose-post-service"):
        """
        Method for fetching traces from JAEGER API.  
        """
        logger.info("------ JAeger api called")

        # JaegerAPIEndpoint = "http://localhost:16686/api/traces?end={}&maxDuration&minDuration&service={}&start={}&prettyPrint=true"
        
        ## get current time in jaeger format
        current_milli_time = lambda: int(round(time.time() * 1000000))    
        
        end=current_milli_time()
        ## adjust time -- delayed batch by 10 sec
        delayed_sec = 15
        end = end - delayed_sec * 1000000

        ## period for lookback
        start = end - int(self.period) * 1000000

        logger.debug(str(("start: ", start, " end: ",end)))

        formatted_endpoint = str(self.JaegerAPIEndpoint.format(end, service, start)).replace('"','')
        logger.debug(str(("formatted endpiont now: ", formatted_endpoint)))

        response_batch = requests.get(url = formatted_endpoint)

        data = response_batch.json()
        logger.info(str(("# of traces in this batch: ",len(data["data"]))))
        return data

    def traces_to_df_with_self(self,all_traces_data, application_name = "SocialNetwork", all_enabled = False):
        """
        Method for mapping traces to astraea data structure. 
        Input: trace["data"] and boolean flag to indicate application. Boolean is for identifying unique spans (url is used in tt).
        Output: DataFrame w/ columns = ['Name','Count', 'Mean','Var','Mean_sum','Var_sum','m/std']
        """
        
        dict_spans = {}
        dict_traces = {} ## traceID, Graph, lat
        span_stats = {} ## global span stats
        span_stats_e2e = {} ## e2e latencies per span
        span_stats_summed = {} ### sum repeating spans
        end_to_end_lats = []
        end_to_end_lats_dict ={} ## traceId, lat

        corrupted_traces =0
        for trace in all_traces_data:
            
            G = nx.DiGraph()
            traceID = trace["traceID"]
            logger.debug(str(("Working on TraceID " ,traceID)))
            span_ids = []
        
            for span in trace["spans"]:
                is_server = False
                ## e2e latency
                span_ids.append(span["spanID"])

                if application_name == "TrainTicket": #is_train_ticket:
                    # ****** get http url and append it to span name - unique
                    for tag in span["tags"]:
                        if tag["key"] == "http.url":
                            url = tag["value"]
                        if tag["key"] == "span.kind" and tag["value"] == "server":
                            is_server = True

                    urlLastPart = url.split("/")[-1]
                    while (  any(ele.isupper() for ele in urlLastPart) or any(ele.isdigit() for ele in urlLastPart)):
                        url = url[0:url.rindex(urlLastPart)-1]
                        urlLastPart = url.split("/")[-1]

                    if not is_server: # spans can be uniquely identified via svc+opName+url
                        key_now = trace["processes"][span["processID"]]["serviceName"] + ":" + span["operationName"] + ":" + url
                    else: # spans can be uniquely identified via svc+opName
                        key_now = trace["processes"][span["processID"]]["serviceName"] + ":" + span["operationName"] 
                ## if deathstar or uber traces -> keynow does not include url
                else:
                    if application_name == "SocialNetwork": ## spans can be uniquely identified via operation name
                        key_now = span["operationName"] 
                    else: ## Media -- spans can be uniquely identified via svc+opName
                        key_now = trace["processes"][span["processID"]]["serviceName"] + ":" + span["operationName"]                     
                
                if span["references"]:
                    parent = span["references"][0]["spanID"]
                else:
                    parent = ""

                ## create node from span 
                node = Node(key_now,parents=parent, iden=span["spanID"], latency=span["duration"], start=span["startTime"], end=span["startTime"]+span["duration"])
                dict_spans[span["spanID"]] = span
                G.add_node(span["spanID"], node=node)

            ## construct edges
            for item in span_ids:
                span = dict_spans[item]
                if span["references"]:
                    G.add_edge(span["references"][0]["spanID"],span["spanID"])

            dict_traces[traceID] = G.copy()        
            ## get spans for analysis
            local_span_stats = {}
            local_span_max = {}
            local_span_min = {}
            local_span_count = {}
            e2e_lat = 0
            
            if nx.number_of_nodes(G) != nx.number_of_edges(G) + 1:
                logger.warning(str(("mertiko problematic trace",traceID)))
                dict_traces.pop(traceID)
                corrupted_traces += 1
                continue
            
            for x in G.nodes():
                if not G.nodes[x]:
                    logger.warning(str(("!!!!!!!! Problem",traceID, 0/0)))
                    
                span_now = G.nodes[x]['node'].name   
                logger.debug(str(("\n---Main span now: ", span_now, " , duration", G.nodes[x]['node'].latency, " id ", G.nodes[x]['node'].id)))

                ## update immediate parent list if not root!
                if all_enabled and G.in_degree(x) != 0:
                    span_parent =nx.dfs_predecessors(G, source=x, depth_limit=1)
                    key, values = span_parents.items()
                    span_parent_name = G.nodes[values[0]]['node'].name 
                    if span_now not in self.immediate_parent:
                        self.immediate_parent[span_now] = set()
                    ## add it to set anyway, no repeating is allowed
                    self.immediate_parent[span_now].add(span_parent_name)
                    logger.debug(str(("---Immediate parent updated for ",span_now,  self.immediate_parent[span_now])))
                
                if G.in_degree(x) == 0: ## root
                    end_to_end_lats.append(G.nodes[x]['node'].latency)
                    end_to_end_lats_dict[traceID] = G.nodes[x]['node'].latency
                    e2e_lat = G.nodes[x]['node'].latency

                elif G.out_degree(x)==0: ###leaves
                    ## this was not leaf before (disabled now); so extract the children estimate
                    if span_now in self.concurrent_children: 
                        child_lat_before = 0
                        for elem in self.concurrent_children[span_now]:
                            estimates_before = elem["max"]
                            max_estimate = [i for i in estimates_before if i > 0]
                            child_lat_before += np.mean(max_estimate)

                        logger.debug(str(("local span update", span_now, G.nodes[x]['node'].latency, G.nodes[x]['node'].latency - child_lat_before, self.concurrent_children[span_now])))
                        local_span_stats[span_now] = local_span_stats.get(span_now,0) +  max(0,G.nodes[x]['node'].latency - child_lat_before)
                        local_span_count[span_now] = local_span_count.get(span_now,0) + 1
                        logger.debug(str(("***** This leaf used to have child", span_now, " lat: ", G.nodes[x]['node'].latency, " self:" , G.nodes[x]['node'].latency - child_lat_before, " see child ",  self.concurrent_children[span_now])))

                    else: ## this is leaf, and was always a leaf
                        ## sum local observations
                        logger.debug(str(("local span update", span_now, G.nodes[x]['node'].latency)))
                        local_span_stats[span_now] = local_span_stats.get(span_now,0) + G.nodes[x]['node'].latency
                        local_span_count[span_now] = local_span_count.get(span_now,0) + 1
                    
                    local_span_max[span_now] = G.nodes[x]['node'].latency if G.nodes[x]['node'].latency > local_span_max.get(span_now,0) else local_span_max.get(span_now,0)
                    local_span_min[span_now] = G.nodes[x]['node'].latency if G.nodes[x]['node'].latency < local_span_min.get(span_now,0) else local_span_min.get(span_now,0)

                else: ## intermediate spans
                    ### Hard-coded check for Social Network repeating span.. The root and the very first child are identical so ignore it.
                    if span_now == "/wrk2-api/post/compose":
                        logger.debug(str(("Hard coded for Social Network repeating span")))
                        continue
                    
                    span_child =nx.dfs_successors(G, source=x, depth_limit=1) ## get all immediate children
                    child_lat = 0
                    child_dict = {} ## for concurrency
                    
                    for key, values in span_child.items():
                        # It has multiple children
                        if len(values) > 1:
                            for val in values: ## concurrency logic to calculate duration
                                child_dict[G.nodes[val]['node'].name + "_start"] = G.nodes[val]['node'].start
                                child_dict[G.nodes[val]['node'].name + "_end"] = G.nodes[val]['node'].end

                        else: ## either always had 1 child, or other children are disabled 
                            
                            child_now  = G.nodes[values[0]]['node'].name
                            logger.debug(str(("-=-= Tek child: ", child_now, " duration: ", G.nodes[values[0]]['node'].latency)))

                            ## first time observing parent span!!!, let's update our oracle concurrent_children with single child
                            # child_lat should be extracted later in this code
                            if span_now not in self.concurrent_children:
                                self.concurrent_children[span_now] = [{"children":set([child_now]), "max":deque([0]*self.children_moving_window,maxlen=self.children_moving_window)}]
                                child_lat = G.nodes[values[0]]['node'].latency ## get current child's latency
                                self.concurrent_children[span_now][0]["max"].appendleft(child_lat)

                                logger.debug(str(("--=-=-Added this sppan with one child for first time: ", span_now, self.concurrent_children[span_now])))
                            
                            else: ## 1 child now but could have other children
                                child_found_before = False
                                for item in self.concurrent_children[span_now]:
                                    if child_now in item["children"]: ## concurrent
                                        # child_found_before = True
                                        item["max"].appendleft(G.nodes[values[0]]['node'].latency) ## update its latency estimator
                                    ### there are other sequential children so self segment should be estimated    
                                    estimates_before = item["max"]
                                    max_estimate = [i for i in estimates_before if i > 0]
                                    child_lat += np.mean(max_estimate)

                                logger.debug(str(("***** This span with one child", span_now, " lat: ", G.nodes[x]['node'].latency, " self:" , G.nodes[x]['node'].latency - child_lat, " see child ",  self.concurrent_children[span_now])))
                        
                        ### it has multipl children so more complex analysis; concurrent and sequential breakdown of children for self_segment analysis
                        if child_dict:
                            parent_seen_first_time = False
                            ### if span_now exists in our estimator let's measure previous observations and extract
                            if span_now in self.concurrent_children:
                                for item in self.concurrent_children[span_now]:
                                    estimates_before = item["max"]
                                    max_estimate = [i for i in estimates_before if i > 0]
                                    child_lat += np.mean(max_estimate)
                                    logger.debug(str(("-=-=- Estimating children contribution from previous rounds ",item["children"], np.mean(max_estimate))))
                                logger.debug(str(("Done estimation, total contribution is: ", child_lat)))

                            spans_processes = []
                            most_start = 0
                            ## sort children dict by values
                            child_dict = {k: v for k, v in sorted(child_dict.items(), key=lambda item: item[1])}
                            for key,value in child_dict.items():

                                ## if first span then set the most start time
                                if len(spans_processes) == 0:
                                    most_start = value
                                    active_children = []
                                
                                ## if process started add it to the list
                                if "_start" in key:
                                    spans_processes.append(key)
                                    active_children.append(key.split("_start")[0])                                                

                                    
                                else:
                                    ## remove the operation's start tracepoint
                                    spans_processes.remove(key.split("_end")[0]+"_start")
                                    ## if we do not have any elements in the list, then set most end
                                    if len(spans_processes) == 0: 
                                        # child_lat = child_lat + (value - most_start) ## value = end_timestamp
                                        logger.debug(str(("*** Check child : ", key, " , duration: ", (value - most_start), ' ,total dur: ' , child_lat)))

                                        ## if this is the first time for parent span!!!, let's update our oracle_child_map
                                        if span_now not in self.concurrent_children:
                                            self.concurrent_children[span_now] = [{"children":set(active_children), "max":deque([0]*self.children_moving_window,maxlen=self.children_moving_window)}]
                                            self.concurrent_children[span_now][0]["max"].appendleft(value - most_start)

                                            child_lat += value - most_start
                                            logger.debug(str(("First time added parent span check children ", self.concurrent_children[span_now])))
                                            parent_seen_first_time = True

                                        else:
                                            if parent_seen_first_time: ## still first trace we see parent span; but new set of sequential
                                                obj = {"children":set(active_children), "max":deque([0]*self.children_moving_window,maxlen=self.children_moving_window)}
                                                obj["max"].appendleft(value - most_start)
                                                self.concurrent_children[span_now].append(obj)
                                                child_lat += value - most_start
                                                logger.debug(str(("First/sec time added parent span check children ", self.concurrent_children[span_now])))

                                            ## iterate children and see if it is there concurrently!
                                            for item in self.concurrent_children[span_now]:
                                                ## yes we have some common elements for active children -- so adding to concurrent list
                                                if not set(active_children).isdisjoint(item["children"]): 
                                                    logger.debug(str(("Yes we have some common elements for active children and previous children from map")))
                                                    # child_found_before = True
                                                    item["max"].appendleft(value - most_start) ## update its latency estimator
                                        most_start = 0
                                        
                    logger.debug(str(("local span update", span_now, G.nodes[x]['node'].latency, G.nodes[x]['node'].latency - child_lat, self.concurrent_children[span_now])))
                    local_span_stats[span_now] = local_span_stats.get(span_now,0) + max(0,G.nodes[x]['node'].latency - child_lat)
                    local_span_count[span_now] = local_span_count.get(span_now,0) + 1

            ### sum repeating spans and add to span_stats
            for item in local_span_stats:
                if item not in span_stats:
                    span_stats[item] = []
                    span_stats_e2e[item] = []
                    
                span_stats[item].append(local_span_stats[item]/local_span_count[item])
                span_stats_e2e[item].append(e2e_lat) ## e2e latency per span

                if item not in span_stats_summed:
                    span_stats_summed[item] = []
                span_stats_summed[item].append(local_span_stats[item])         
            
        ### final dataframe populated below
        datalar = []
        pd.set_option("precision", 2)

        for item in span_stats:
            datalar.append([
            item, 
            len(span_stats[item]),
            np.mean(span_stats[item]), 
            np.var(span_stats[item]),
            np.std(span_stats[item]),
            np.max(span_stats[item]),
            ## summed variance, mean, std for repeating spans
            np.mean(span_stats_summed[item]), 
            np.var(span_stats_summed[item]), 
            np.std(span_stats_summed[item]), 
            np.max(span_stats_summed[item]),
    #         np.mean(span_stats_summed[item])/ np.std(span_stats_summed[item]), ### log2 == mean/std              
            np.percentile(span_stats_summed[item], 50),
            np.percentile(span_stats_summed[item], 99),         
    #         corr_matrix.loc['e2e'][item],
    #             corr_matrix_span[item],
    #             corr_matrix_span[item]*(np.percentile(span_stats_summed[item], 99)/np.percentile(span_stats_summed[item], 50))                            
                        ])

        df_traces = pd.DataFrame(datalar, columns = ['Name','Count', 'Mean','Var','Std','Max', 'Mean_sum','Var_sum','Std_sum', 'Max_sum', '50_sum', '99_sum']) 
        #,'R','R*99/50']) # ,'m/std','Max','Min','Range'])
        df_traces['Var_sum_cum'] = df_traces['Var_sum']/(df_traces['Var_sum'].sum()/100)
 
        return df_traces

  
    ### experimental method to calculate all stats for utility
    def traces_to_df_asplos_experimental(self,all_traces_data, application_name = "SocialNetwork", all_enabled = False):
        """
        Method for mapping traces to astraea data structure. 
        Input: trace["data"] and boolean flag to indicate application. Boolean is for identifying unique spans (url is used in tt).
        Output: DataFrame w/ columns = ['Name','Count', 'Mean','Var','Mean_sum','Var_sum','m/std']
        """
        
        dict_spans = {}
        dict_traces = {} ## traceID, Graph, lat
        span_stats = {} ## global span stats
        span_stats_e2e = {} ## e2e latencies per span
        span_stats_summed = {} ### sum repeating spans
        end_to_end_lats = []
        end_to_end_lats_dict ={} ## traceId, lat
        span_max = {}
        span_min = {}
        span_range = {}

        i = 0
        corrupted_traces =0
        print("traces_to_df_asplos trace len now: " , len(all_traces_data))
        for trace in all_traces_data:
            
            G = nx.DiGraph()
            G_spannames = nx.DiGraph()
            traceID = trace["traceID"]
            print("Working on TraceID " ,traceID)
            span_ids = []
        
            for span in trace["spans"]:
                is_server = False
                ## e2e latency
                span_ids.append(span["spanID"])

                
                if application_name == "TrainTicket": #is_train_ticket:
                    # ****** get http url and append it to span name - unique
                    for tag in span["tags"]:
                        if tag["key"] == "http.url":
                            url = tag["value"]
                        if tag["key"] == "span.kind" and tag["value"] == "server":
                            is_server = True

                    urlLastPart = url.split("/")[-1]
                    while (  any(ele.isupper() for ele in urlLastPart) or any(ele.isdigit() for ele in urlLastPart)):
                        url = url[0:url.rindex(urlLastPart)-1]
        #                 print("url; " + url)
                        urlLastPart = url.split("/")[-1]

                    if not is_server: # spans can be uniquely identified via svc+opName+url
                        key_now = trace["processes"][span["processID"]]["serviceName"] + ":" + span["operationName"] + ":" + url
                    else: # spans can be uniquely identified via svc+opName
                        key_now = trace["processes"][span["processID"]]["serviceName"] + ":" + span["operationName"] 
                ## if deathstar or uber traces -> keynow does not include url
                else:
                    if application_name == "SocialNetwork": ## spans can be uniquely identified via operation name
                        key_now = span["operationName"] 
                    else: ## Media -- spans can be uniquely identified via svc+opName
                        key_now = trace["processes"][span["processID"]]["serviceName"] + ":" + span["operationName"] 
    #                 key_now = span["operationName"] 
                    
                
                if span["references"]:
                    parent = span["references"][0]["spanID"]
                else:
                    parent = ""

                ## create node from span 
                node = Node(key_now,parents=parent, iden=span["spanID"], latency=span["duration"], start=span["startTime"], end=span["startTime"]+span["duration"])
                dict_spans[span["spanID"]] = span
                G.add_node(span["spanID"], node=node)

            ## construct edges
            for item in span_ids:
                span = dict_spans[item]
                if span["references"]:
                    G.add_edge(span["references"][0]["spanID"],span["spanID"])

            dict_traces[traceID] = G.copy()        
            ## get spans for analysis
            local_span_stats = {}
            local_span_max = {}
            local_span_min = {}
            local_span_count = {}
            e2e_lat = 0
            
            if nx.number_of_nodes(G) != nx.number_of_edges(G) + 1:
                print("mertiko problematic trace",traceID)
                dict_traces.pop(traceID)
                corrupted_traces += 1
                continue
            
            for x in G.nodes():
                if not G.nodes[x]:
                    print("!!!!!!!! Problem",traceID, 0/0)
                    break
                    
                span_now = G.nodes[x]['node'].name   
                print("\n---Main span now: ", span_now, " , duration", G.nodes[x]['node'].latency, " id ", G.nodes[x]['node'].id)

                ## update immediate parent list if not root!
                if all_enabled and not G.in_degree(x) == 0:
                    span_parent =nx.dfs_predecessors(G, source=x, depth_limit=1)
                    key, values = span_parents.items()
                    span_parent_name = G.nodes[values[0]]['node'].name 
                    if span_now not in self.immediate_parent:
                        self.immediate_parent[span_now] = set()
                    ## add it to set anyway, no repeating is allowed
                    self.immediate_parent[span_now].add(span_parent_name)
                    print("---Immediate parent updated for ",span_now,  self.immediate_parent[span_now])
                
                if G.in_degree(x) == 0: ## root
                    end_to_end_lats.append(G.nodes[x]['node'].latency)
                    end_to_end_lats_dict[traceID] = G.nodes[x]['node'].latency
                    e2e_lat = G.nodes[x]['node'].latency

                elif G.out_degree(x)==0: ###leaves

 
                    ## if it had any children before (i.e., disabled children)
                    if span_now in self.concurrent_children: ## it had children before so extract the children estimate
                        child_lat_before = 0
                        for elem in self.concurrent_children[span_now]:
                            estimates_before = elem["max"]

                            max_estimate = [i for i in estimates_before if i > 0]
                            child_lat_before += np.mean(max_estimate)

                        print("local span update", span_now, G.nodes[x]['node'].latency, G.nodes[x]['node'].latency - child_lat_before, self.concurrent_children[span_now])
                        local_span_stats[span_now] = local_span_stats.get(span_now,0) +  G.nodes[x]['node'].latency - child_lat_before
                        local_span_count[span_now] = local_span_count.get(span_now,0) + 1
                        print("***** This leaf used to have child", span_now, " lat: ", G.nodes[x]['node'].latency, " self:" , G.nodes[x]['node'].latency - child_lat_before, " see child ",  self.concurrent_children[span_now])

                    else: ## this leaf was always a leaf
                        ## sum local observations
                        print("local span update", span_now, G.nodes[x]['node'].latency)
                        local_span_stats[span_now] = local_span_stats.get(span_now,0) + G.nodes[x]['node'].latency
                        local_span_count[span_now] = local_span_count.get(span_now,0) + 1
                    

                    local_span_max[span_now] = G.nodes[x]['node'].latency if G.nodes[x]['node'].latency > local_span_max.get(span_now,0) else local_span_max.get(span_now,0)
                    local_span_min[span_now] = G.nodes[x]['node'].latency if G.nodes[x]['node'].latency < local_span_min.get(span_now,0) else local_span_min.get(span_now,0)

                else: ## intermediate spans
                    ### Hard-coded check for Social Network repeating span.. The root and the very first child are identical so ignore it.
                    if span_now == "/wrk2-api/post/compose":
                        print("Hard coded for Social Network repeating span")
                        continue
                    
                    span_child =nx.dfs_successors(G, source=x, depth_limit=1) ## get all immediate children
                    
                    child_lat = 0
                    ## for concurrency
                    child_dict = {}
                    
                    for key, values in span_child.items():
                        # It has multiple children
                        if len(values) > 1:
                            for val in values:
                                child_dict[G.nodes[val]['node'].name + "_start"] = G.nodes[val]['node'].start
                                child_dict[G.nodes[val]['node'].name + "_end"] = G.nodes[val]['node'].end

                        else: ## either always had 1 child, or other children are disabled 
                            child_lat = G.nodes[values[0]]['node'].latency ## get current child's latency
                            child_now  = G.nodes[values[0]]['node'].name
                            print("-=-= Tek child: ", child_now, " duration: ", child_lat)

                            ## first time observing parent span!!!, let's update our oracle concurrent_children with single child
                            # child_lat should be extracted later in this code
                            if span_now not in self.concurrent_children:
                                self.concurrent_children[span_now] = [{"children":set([child_now]), "max":deque([0]*self.children_moving_window,maxlen=self.children_moving_window)}]
                                # child_lat = child_lat + G.nodes[values[0]]['node'].latency
                                self.concurrent_children[span_now][0]["max"].appendleft(child_lat)
                            
                                print(span_now, "     Added this sppan for first time, ", self.concurrent_children[span_now])

                            
                            else: ## 1 child now but could have other children
                                # child_found_before = False
                                ## iterate children and see if it is there!
                                for item in self.concurrent_children[span_now]:
                                    if child_now in item["children"]: ## concurren
                                        # child_found_before = True
                                        item["max"].appendleft(child_lat) ## update its latency estimator
                                    else: ### there are other sequential children so self segment should be estimated
                                        
                                        estimates_before = item["max"]
                                        max_estimate = [i for i in estimates_before if i > 0]
                                        child_lat += np.mean(max_estimate)

                                        print("-=-=- there were other squential children disabled with ",item["children"], np.mean(max_estimate))

                                # if not child_found_before:
                                #     print("We did not see this single child before ", 0/0)
                                    
                                #     obj = {"children":set([G.nodes[values[0]]['node'].name]), "max":deque([0]*self.children_moving_window,maxlen=self.children_moving_window)}
                                #     obj["max"].appendleft(child_lat)
                                #     self.concurrent_children[span_now].append(obj)

                                # local_span_stats[span_now] = local_span_stats.get(span_now,0) +  G.nodes[x]['node'].latency - child_lat
                                # local_span_count[span_now] = local_span_count.get(span_now,0) + 1

                                print("***** This span with one child", span_now, " lat: ", G.nodes[x]['node'].latency, " self:" , G.nodes[x]['node'].latency - child_lat, " see child ",  self.concurrent_children[span_now])
                        ### it has multipl children so more complex analysis;
                        ## concurrent and sequential breakdown of children for self_segment analysis
                        if child_dict:
                            parent_seen_first_time = False
                            ### if span_now exists in our estimator let's measure previous observations and extract
                            ### if not, it is the first time seeing it so measure and extract
                            if span_now in self.concurrent_children:
                                for item in self.concurrent_children[span_now]:
                                    estimates_before = item["max"]
                                    max_estimate = [i for i in estimates_before if i > 0]
                                    child_lat += np.mean(max_estimate)
                                    print("-=-=- Estimating children contribution from previous rounds ",item["children"], np.mean(max_estimate))
                                print("Done estimation, total contribution is: ", child_lat)

                            spans_processes = []
                            most_start = 0
                            ## sort children dict by values
                            child_dict = {k: v for k, v in sorted(child_dict.items(), key=lambda item: item[1])}
                            for key,value in child_dict.items():

                                ## if first span then set the most start time
                                if len(spans_processes) == 0:
                                    most_start = value
                                    active_children = []
                                
                                ## if process started add it to the list
                                if "_start" in key:
                                    spans_processes.append(key)
                                    active_children.append(key.split("_start")[0])                                                

                                    
                                else:
                                    ## remove the operation's start tracepoint
                                    spans_processes.remove(key.split("_end")[0]+"_start")
                                    ## if we do not have any elements in the list, then set most end
                                    if len(spans_processes) == 0: 
                                        # child_lat = child_lat + (value - most_start) ## value = end_timestamp
                                        print("*** Check child : ", key, " , duration: ", (value - most_start), ' ,total dur: ' , child_lat)


                                        ## if this is the first time for parent span!!!, let's update our oracle_child_map
                                        if span_now not in self.concurrent_children:
                                            self.concurrent_children[span_now] = [{"children":set(active_children), "max":deque([0]*self.children_moving_window,maxlen=self.children_moving_window)}]
                                            self.concurrent_children[span_now][0]["max"].appendleft(value - most_start)

                                            child_lat += value - most_start
                                            print("First time added parent span check children ", self.concurrent_children[span_now])
                                            parent_seen_first_time = True

                                        ## if we saw parent span before, then try to find its children
                                        else:
                                            if parent_seen_first_time: ## still first trace we see parent span; but new set of sequential
                                                obj = {"children":set(active_children), "max":deque([0]*self.children_moving_window,maxlen=self.children_moving_window)}
                                                obj["max"].appendleft(value - most_start)
                                                self.concurrent_children[span_now].append(obj)
                                                child_lat += value - most_start
                                                print("First/sec time added parent span check children ", self.concurrent_children[span_now])

                                            # child_found_before = False
                                            ## iterate children and see if it is there concurrently!
                                            for item in self.concurrent_children[span_now]:
                                                ## yes we have some common elements for active children -- so adding to concurrent list
                                                if not set(active_children).isdisjoint(item["children"]): 
                                                    print("Yes we have some common elements for active children and previous children from map")
                                                    # child_found_before = True
                                                    item["max"].appendleft(value - most_start) ## update its latency estimator

                                                    # Following piece was breaking the code, adding grand children to immediate children
                                                    # for active_diff in list(set(active_children) - item["children"]): ## add mnissing elements if any
                                                    #     print("-=-=-=-=-!!!Active diff now, ", active_diff)
                                                    #     item["children"].add(active_diff)



                                            # ### We have observed this parent before but no child like this :/
                                            # if not child_found_before:
                                            #     ## TODO! problem!!!!!
                                            #     print("-=-=-=-We do not have any commons, so creating sequential child", active_children, 0/0)
                                            #     obj = {"children":set(active_children), "max":deque([0]*self.children_moving_window,maxlen=self.children_moving_window)}
                                            #     obj["max"].appendleft(value - most_start)
                                            #     self.concurrent_children[span_now].append(obj)
                                            #     child_lat += value - most_start

                                        most_start = 0
                                        
                    print("local span update", span_now, G.nodes[x]['node'].latency, G.nodes[x]['node'].latency - child_lat, self.concurrent_children[span_now])
                    local_span_stats[span_now] = local_span_stats.get(span_now,0) + G.nodes[x]['node'].latency - child_lat
                    local_span_count[span_now] = local_span_count.get(span_now,0) + 1
    #                 print("Parent now: ", span_now, " new duration: ", local_span_stats[span_now])
            # print("******** debug concurrent children " , self.concurrent_children)
            ### sum repeating spans and add to span_stats
            for item in local_span_stats:
                if item not in span_stats:
                    span_stats[item] = []
                    span_stats_e2e[item] = []
                    
                span_stats[item].append(local_span_stats[item]/local_span_count[item])
                span_stats_e2e[item].append(e2e_lat) ## e2e latency per span

                if item not in span_stats_summed:
                    span_stats_summed[item] = []
                span_stats_summed[item].append(local_span_stats[item])

            i = i + 1 
    # print("+Parsed traces", i)

        ### e2e correlation and eta for utility evaluation purposes
    #     span_single = {}
    #     for item in span_stats:
    #         if item not in span_single:
    #             span_single[item] = []
    #         span_single[item].extend(span_stats[item])

    #     # span_single
    #     span_single["e2e"] = end_to_end_lats
    #     print(span_single)
    #     single_ladies = pd.DataFrame.from_dict(span_single)
    #     # single_ladies
    #     corr_matrix = single_ladies.corr()

    ## new e2e correlation for different sizes
    #     corr_matrix_span ={}
    #     for item in span_stats:
    #         corr, _ = pearsonr(span_stats[item], span_stats_e2e[item])
    #         corr_matrix_span[item]= corr
            
            
            
        ### final dataframe populated below
        datalar = []
        pd.set_option("precision", 2)

        for item in span_stats:

            datalar.append([
            item, 
            len(span_stats[item]),
            np.mean(span_stats[item]), 
            np.var(span_stats[item]),
            np.std(span_stats[item]),
            np.max(span_stats[item]),
            ## summed variance, mean, std for repeating spans
            np.mean(span_stats_summed[item]), 
            np.var(span_stats_summed[item]), 
            np.std(span_stats_summed[item]), 
            np.max(span_stats_summed[item]),
                
    #         np.mean(span_stats_summed[item])/ np.std(span_stats_summed[item]), ### log2 == mean/std
                            
            np.percentile(span_stats_summed[item], 50),
            np.percentile(span_stats_summed[item], 99),
            
    #         corr_matrix.loc['e2e'][item],
    #             corr_matrix_span[item],
    #             corr_matrix_span[item]*(np.percentile(span_stats_summed[item], 99)/np.percentile(span_stats_summed[item], 50))
                
                            
                            
                        ])

        df_traces = pd.DataFrame(datalar, columns = ['Name','Count', 'Mean','Var','Std','Max', 'Mean_sum','Var_sum','Std_sum', 'Max_sum', '50_sum', '99_sum']) #,'R','R*99/50']) # ,'m/std'
    #                                                 'Max','Min','Range'])

        df_traces['Var_sum_cum'] = df_traces['Var_sum']/(df_traces['Var_sum'].sum()/100)
    #     df_traces['R*99/50_cum'] = df_traces['R*99/50']/(df_traces['R*99/50'].sum()/100)
    #     df_traces['50_sum_cum'] = df_traces['50_sum']/(df_traces['50_sum'].sum()/100)
    #     return df_traces
        return df_traces, end_to_end_lats_dict, span_stats, dict_traces,end_to_end_lats,span_stats_summed



### experimental method to calculate all stats for utility -- LOG2
def traces_to_df_asplos_experimental_log2(all_traces_data, is_train_ticket = True):
    """
    Method for mapping traces to astraea data structure. 
    Input: trace["data"] and boolean flag to indicate application. Boolean is for identifying unique spans (url is used in tt).
    Output: DataFrame w/ columns = ['Name','Count', 'Mean','Var','Mean_sum','Var_sum','m/std']
    """
    
    dict_spans = {}
    dict_traces = {} ## traceID, Graph, lat
    span_stats = {} ## global span stats
    span_stats_e2e = {} ## e2e latencies per span
    span_stats_summed = {} ### sum repeating spans
    end_to_end_lats = []
    end_to_end_lats_dict ={} ## traceId, lat
    span_max = {}
    span_min = {}
    span_range = {}
    


    i = 0
    corrupted_traces =0
    print("traces_to_df_asplos trace len now: " , len(all_traces_data))
    for trace in all_traces_data:
        
        local_span_stats = {}
        local_span_count = {}
        
        traceID = trace["traceID"]
#         print("Now working on : " , traceID, " len: ", len(trace["spans"]))
        span_ids = []
     
        for span in trace["spans"]:
            is_server = False
            ## e2e latency
            span_ids.append(span["spanID"])
            
            if is_train_ticket:
                # ****** get http url and append it to span name - unique
                for tag in span["tags"]:
                    if tag["key"] == "http.url":
                        url = tag["value"]
                    if tag["key"] == "span.kind" and tag["value"] == "server":
                        is_server = True

                urlLastPart = url.split("/")[-1]
                while (  any(ele.isupper() for ele in urlLastPart) or any(ele.isdigit() for ele in urlLastPart)):
                    url = url[0:url.rindex(urlLastPart)-1]
    #                 print("url; " + url)
                    urlLastPart = url.split("/")[-1]

                if not is_server:
                    key_now = trace["processes"][span["processID"]]["serviceName"] + ":" + span["operationName"] + ":" + url
                else:
                    key_now = trace["processes"][span["processID"]]["serviceName"] + ":" + span["operationName"] 
            ## if deathstar or uber traces -> keynow does not include url
            else:
                #key_now = trace["processes"][span["processID"]]["serviceName"] + ":" + span["operationName"] 
                key_now = span["operationName"] 
                
            
            if span["references"]:
                parent = span["references"][0]["spanID"]
            else:
                parent = ""

            ## create node from span 
            node = Node(key_now,parents=parent, iden=span["spanID"], latency=span["duration"], start=span["startTime"], end=span["startTime"]+span["duration"])
            
            ## we do not need graph in log2 -- so just get duration
            local_span_stats[key_now] = local_span_stats.get(key_now,0) + span["duration"]
            local_span_count[key_now] = local_span_count.get(key_now,0) + 1
            
            span_max[key_now] = span["duration"] if span["duration"] > span_max.get(key_now,0) else span_max.get(key_now,0)
                
            

        ## now calculate mean for single trace w/ multiple spans
        for item in local_span_stats:
            if item not in span_stats:
                span_stats[item] = []
                span_stats_e2e[item] = []
                
            span_stats[item].append(local_span_stats[item]/local_span_count[item])
        
    ### final dataframe populated below
    datalar = []
    pd.set_option("precision", 2)
    
#     print(local_span_count)
#     print(span_stats)
#     print(span_max)

    for item in span_stats:     
        datalar.append([
                        item, 
                        local_span_count[item] if item in local_span_count else 1,
                        np.mean(span_stats[item]), 
                        span_max[item]
                       ])

    df_traces = pd.DataFrame(datalar, columns = ['Name','Count', 'Mean','Max']) # ,'m/std'
#                                                 'Max','Min','Range'])
    df_traces['Mean_cum'] = df_traces['Mean']/(df_traces['Mean'].sum()/100)
    df_traces['Max_cum'] = df_traces['Max']/(df_traces['Max'].sum()/100)

    return df_traces
    

class Node(object):
    """
    Class of node objects in a doubly linked list to
    represent DOT edges in a treelike form. source
    nodes are parents and destination nodes are children.
    info from edge labels resides in destination nodes/
    children.
    """
    def __init__(self, name='root',  parents="", iden = "", latency = 0, start=0, end=0):
        self.name = name
        self.parents = parents
        self.id = iden
        self.latency = latency
        self.start = start
        self.end = end
        
def get_traces_jaeger_file(file_name):
    """
    Method for reading traces from provided file. 
    Used when replaying traces from files. 
    """
    print("------")
    data = {}
    with open(file_name) as f:
        data = json.load(f)

    # Output: {'name': 'Bob', 'languages': ['English', 'Fench']}
    print("# of traces: ",len(data["data"]))
    return data
