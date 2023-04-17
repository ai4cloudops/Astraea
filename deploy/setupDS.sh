## Pre-requirements
# Docker
# Docker-compose
# Python 3.5+ (with asyncio and aiohttp)

# scripts tested for "Ubuntu 18.04.1 LTS"

## install requirements of Astraea 
sudo apt-get --yes install python3-pip
sudo apt --yes install libjpeg-dev zlib1g-dev
pip3 install Pillow
pip3 install -r ../src/requirements.txt
echo "Requirements installed for Astraea" 

## install docker and docker-compose
sudo apt update
sudo apt --yes install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
apt-cache policy docker-ce
sudo apt --yes install docker-ce
sudo curl -L "https://github.com/docker/compose/releases/download/1.27.4/docker-compose-Linux-x86_64" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo docker-compose –version


## install DS specific requirements
sudo apt-get --yes install luarocks
sudo luarocks install luasocket
sudo apt --yes install python3.8
sudo apt-get --yes install libssl-dev
sudo apt-get --yes install libz-dev
sudo apt-get --yes install python3-pip
pip3 install --upgrade setuptools
pip3 install aiohttp[full]
# pip3 install aiohttp

echo "Requirements installed for DS" 

## clone modified DS repo
git clone https://github.com/mtoslalibu/DeathStarBench.git
sudo chmod -R ugo+rwx .

## build your own version 
sudo docker build -t mert/thrift-microservice-deps:xenial DeathStarBench/socialNetwork/docker/thrift-microservice-deps/cpp/
sudo docker build -t mert/social-network-microservices:latest DeathStarBench/socialNetwork/
echo "Built social network from source" 

sudo docker-compose -f DeathStarBench/socialNetwork/docker-compose.yml up -d
echo "Deployed DS social network app"

cd DeathStarBench/socialNetwork/
sudo python3 scripts/init_social_graph.py --graph=socfb-Reed98
cd ../../
echo "Registered users and construct social graphs" 

## build workload generator
cd DeathStarBench/socialNetwork/wrk2
make
cd ../../../
echo "Built wrk2"

## make spans policy directory accessible by Astraea
sudo mkdir -p /local/astraea-spans/
# sudo chmod ugo+rwx -R /local/astraea-spans/
touch /local/astraea-spans/sleeps
sudo chmod ugo+rwx -R /local/astraea-spans/


## prepare default sampling policy (for experimental purposes)
## deathstar
cat ../data/all-spans-socialNetwork-probabilities > /local/astraea-spans/spans-orig

## run an experiment (don't forget to modify conf files for the target application)
cd ../eval/
mkdir -p astraea-results
python3 AstraeaEvaluator.py