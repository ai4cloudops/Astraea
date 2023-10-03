# Astraea
The demo repo of Astraea implementation. 

Astraea is an online distributed tracing system that tackles the trade-off between tracing cost and granularity. It achieves this by utilizing our mathematical formulation that combines online Bayesian learning and multi-armed bandit frameworks. With Astraea, tracing is directed towards the critical instrumentation necessary for performance diagnosis. 

This repository contains guidelines for configuring Astraea with either a Social Network or Train Ticket application. It includes a demonstration of an experiment where a delay is deliberately introduced into the source code of an application. Astraea then adjusts span decisions in a manner that tries to maximize the sampling ratio of the faulty span responsible for capturing the injected delay. The ultimate goal of this process is to enhance the efficiency of tracing.

# Content
This repo includes /src 
- AstraeaController: Periodically run; fetch traces, extract span stats, apply Bayesian methods, and issue sampling decisions to application. 
- TraceManager: Read and parse traces. Convert trace to DAG, and extract span utilities and statistics. 
- BayesianFramework: Implementation of Astraea Bayesian online learning and ABS algorithm. Initialize and update beta distributions, apply ABS algorithm. 



# Setup 

## Setup social network
This Code assumes that the application is already setup with astraea-compatible tracing client. i.e the tracing client
on the application is syncying span-sampling probabilities from a s3 file.
You may see example of the C++ Jaeger tracing agent here [Socil Media application](https://github.com/syedmohdqasim/DeathStarBench/blob/master/socialNetwork/docker/thrift-microservice-deps/cpp/Tracer.cpp)
```console
instructions..
## create a virtual env using Python 3.11.9
##clone and cd into repo.
python -m venv venv
source venv/bin/activate
pip install -r src/requirements.txt
## populate jaeger(JaegerAPIEndpoint) , and other aws config in conf/astraea-config.ini
cd src
## START ASTRAEA
python3 AstraeaController.py

## you will see tables with span sampling probabilities and variations on console.
```

## Results are populated under 
ResultDir = src/results <br />
That directory includes probability recordings (corresponding to sampling policy) as well as trace sizes recorded.


# Notes

- A demo case study on Key-value correlation analysis can be found at src/KeyValueCorr.py  <br />


# License
<!-- License -->
------- <br />

Astraea is licensed under the [BSD 3-Clause license](https://github.com/mtoslalibu/astraea/blob/master/LICENSE).
