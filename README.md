# Astraea
The demo repo of Astraea implementation. <br />
Astraea is an online distributed tracing system that tackles the trade-off between tracing cost and granularity. It achieves this by utilizing our mathematical formulation that combines online Bayesian learning and multi-armed bandit frameworks. With Astraea, tracing is directed towards the critical instrumentation necessary for performance diagnosis. <br />
This repository contains guidelines for configuring Astraea with either a Social Network or Train Ticket application. It includes a demonstration of an experiment where a delay is deliberately introduced into the source code of an application. Astraea then adjusts span decisions in a manner that tries to maximize the sampling ratio of the faulty span responsible for capturing the injected delay. The ultimate goal of this process is to enhance the efficiency of tracing.

# Content
This repo includes /src <br />
AstraeaController: Periodically run; fetch traces, extract span stats, apply Bayesian methods, and issue sampling decisions to application. <br />
TraceManager: Read and parse traces. Convert trace to DAG, and extract span utilities and statistics.<br />
BayesianFramework: Implementation of Astraea Bayesian online learning and ABS algorithm. Initialize and update beta distributions, apply ABS algorithm.<br />


# Install requirements of Astraea 
sudo apt-get --yes install python3-pip <br />
sudo apt install libjpeg-dev zlib1g-dev <br />
pip3 install Pillow <br />
cd src <br />
pip3 install -r requirements.txt <br />

# Setup the application (Modified tracing to deliver selective enable/disable)
First, you need to change the specified application from the config files (under /conf). Default is social network, to be able to use train ticket, just comment-out social network, and -in the train ticket related configs.

## Setup environment
We use Cloudlab to setup benchmark applications (Social network and Train ticket). <br />
Cloudlab profile for Train ticket is at https://github.com/mtoslalibu/astraea-tt-cloudlab <br />
Cloudlab profile for Social network is at https://github.com/mtoslalibu/astraea-ds-cloudlab <br />
You can follow instructions on how to create a profile at https://docs.cloudlab.us/creating-profiles.html and use one of the repositories above. <br />

In those repos, setup.sh includes instructions on how to setup docker, create extrafs (a cloudlab script that can be run to quickly create a new filesystem using the remaining space on the system disk -- docker runtime needs to be updated accordingly using --data-root), install benchmark application and requirements, and build docker images. When you boot an experiment using a profile, applications are under /local (e.g., /local/train-ticket or /local/DeathStarBench/socialNetwork). Then, you can use the yaml files to boot the application (see below). 

## Setup train ticket
cd /local/train-ticket <br />
sudo docker-compose up -d <br />

## Setup social network
cd /local/DeathStarBench/socialNetwork <br />
sudo docker-compose up -d <br />

## make spans policy directory accessible by Astraea 
sudo touch /local/astraea-spans/sleeps <br />
sudo chmod ugo+rwx -R /local/astraea-spans/ <br />

## Prepare default sampling policy (for experimental purposes)
### Social network
cd /local/astraea/src
cat ../data/all-spans-socialNetwork-probabilities > /local/astraea-spans/spans-orig <br />
### Train ticket
cat ../data/all-spans-tt > /local/astraea-spans/spans-orig <br />
sudo chmod ugo+rwx -R /local/astraea-spans/


## Populate data for Social network app
cd /local/DeathStarBench/socialNetwork
pip3 install --upgrade setuptools <br />
pip3 install aiohttp[full] <br />
sudo python3 scripts/init_social_graph.py --graph=socfb-Reed98 <br />


## Run an experiment

### Use hey workload  (only for train ticket)
chmod +x ../eval/workload/tt-heyworkload.sh <br />
chmod +x ../eval/workload/hey_linux_amd64 <br />


cd ../eval/ <br />
mkdir -p astraea-results
python3 AstraeaEvaluator.py <br />

### Results are populated under 
ResultDir = astraea/eval/astraea-results <br />
That directory includes probability recordings (corresponding to sampling policy) as well as trace sizes recorded.


# Notes
All scripts are tested on  Operating System: Ubuntu 18.04.1 LTS, Kernel: Linux 4.15.0-169-generic, Architecture: x86-64. <br />
A demo case study on Key-value correlation analysis can be found at src/KeyValueCorr.py  <br />


# License
<!-- License -->
------- <br />

Astraea is licensed under the [BSD 3-Clause license](https://github.com/mtoslalibu/astraea/blob/master/LICENSE).

