"""
Astraea main program. 
Astraea works in a continuous loop. As the application receives user traffic, trace observations are collected and stored in a database. 
Via TraceManager.py, Astraea periodically queries the database (or Jaeger API) to build Bayesian belief distributions and to estimate span-utility pairs using Monte Carlo sampling. 
Utility (e.g., Variance) represents the usefulness of a span for performance diagnosis; as more data is available, the belief distributions converge to the true values of the utility. 
At each iteration, via BayesianMethods.py, Astraea adjusts the sampling probabilities of spans using its multi-armed bandit algorithm, taking into account the belief distributions. 
The fraction of spans with low utility decreases over time as the fraction of the most rewarding spans increases.
 At the end of each iteration, span sampling probabilities are issued to the instrumented cloud application through AstraeaOrchestrator.py/.
"""
import time
import boto3
from IPython.display import display
import AstraeaOrchestrator as ao
import BayesianMethods as banditalg
import TraceManager as traceManager
from config import load_config
from logger import setup_logging

print("***** Welcome to Astraea!")

# Defining main function
def main():
    print("---- Astraea started!")
    config = load_config('../conf/astraea-config.ini')
    logger = setup_logging()
    s3 = boto3.client(
        's3',
        endpoint_url=config['s3_url'],
        aws_access_key_id=config['aws_access_key_id'],
        aws_secret_access_key=config['aws_secret_access_key']
    )
    time.sleep(config['period'])

    # Astraea framework Initialized
    bandit = banditalg.ABE("ABE", "Experiment-id1", confidence=config['confidence'],
                           reward_field=config['reward_field'])
    orchestrator = ao.AstraeaOrc(config['app'], config['span_states_file_txt'], s3)
    trace_manager = traceManager.TraceManager()

    ## peridically run and 1) read traces
    epoch = 0
    samples_so_far = 0
    while epoch < 100:
        epoch += 1
        logger.info(str(("---- runninng epoch: ", epoch)))
        all_traces = trace_manager.get_traces_jaeger_api(service=config['service'])  # "compose-post-service"
        logger.info(str(("collected the batch with len: ", len(all_traces["data"]))))
        start_time = time.time()
        trace_parsed = trace_manager.traces_to_df_with_self(all_traces["data"], application_name=config['app'],
                                                            all_enabled=False)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")
        df_traces = trace_parsed
        samples_so_far += len(all_traces["data"])
        display(df_traces.sort_values(by=config['reward_field'], ascending=False))

        ## apply bayesian methods and get new sampling policy
        splits, sorted_spans = bandit.mert_sampling_median_asplos(df_traces, epoch)
        orchestrator.issue_sampling_policy_txt(splits)

        logger.info(str(("Finished epoch ", epoch)))

        ### collect stats
        ### trace sizes
        span_counts = []
        for trace in all_traces["data"]:
            span_counts.append([len(trace["spans"]), epoch])

        trace_manager.append_to_csv(config['result_dir'], "tracesizes.csv", span_counts)

        logger.info(str((" Astraea Controller with params: epsilon:", (config['epsilon']), "Percentile:",
                         (config['elim_percentile']))))
        ### sampling policy
        sampling_policies = []
        with open(config['span_states_file_txt']) as samplingPolicy:
            for line in samplingPolicy:
                name, var = line.partition(" ")[::2]
                sampling_policies.append([name.strip(), float(var), epoch, samples_so_far])

        trace_manager.append_to_csv(config['result_dir'],
                                    "probability.csv", sampling_policies)
        logger.info("Saved sampling probabilities")

        time.sleep(config['period'])


if __name__ == "__main__":
    main()
