## Pre-requirements
# Docker
# Docker-compose

# scripts tested for "Ubuntu 18.04.1 LTS"

## install requirements of Astraea 
sudo apt-get --yes install python3-pip
sudo apt --yes install libjpeg-dev zlib1g-dev
pip3 install Pillow
pip3 install -r ../src/requirements.txt
echo "Requirements installed for Astraea" 

# install openjdk and maven for train ticket
sudo apt update
sudo apt-get --yes install openjdk-8-jdk
sudo apt --yes install maven

## install docker and docker-compose
sudo apt --yes install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
apt-cache policy docker-ce
sudo apt --yes install docker-ce
sudo curl -L "https://github.com/docker/compose/releases/download/1.27.4/docker-compose-Linux-x86_64" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo docker-compose --version

## fork repo of java client
git clone https://github.com/mtoslalibu/jaeger-client-java.git    
cd jaeger-client-java
git checkout --track origin/v0.30.6-astraea
git submodule init
git submodule update
sudo ./gradlew clean install
cd ..
echo "Checkpoint jaeger-client-java" 

## servlet mert’s version
git clone https://github.com/mtoslalibu/java-web-servlet-filter.git
cd java-web-servlet-filter
git checkout --track origin/v0.1.1-astraea
sudo ./mvnw clean install -Dlicense.skip=true -Dcheckstyle.skip -DskipTests=true
cd ..
echo "Checkpoint java-web-servlet-filter" 

## java spring web mert’s version
git clone https://github.com/mtoslalibu/java-spring-web.git
cd java-spring-web
git checkout --track origin/v-0.3.4-astraea
sudo ./mvnw clean install -Dlicense.skip=true -Dcheckstyle.skip -DskipTests=true
cd ..
echo "Checkpoint java-spring-web.git"


## git clone fork repo of java spring jaeger
git clone https://github.com/mtoslalibu/java-spring-jaeger.git
cd java-spring-jaeger
git checkout --track origin/v0.2.2-astraea
sudo ./mvnw clean install -Dlicense.skip=true -Dcheckstyle.skip -DskipTests=true
cd ..
echo "Checkpoint java spring jaeger"

## go trainticket + switch to jaeger branch + then change java-jaeger-spring version to snapshot
git clone https://github.com/mtoslalibu/train-ticket.git
cd train-ticket
git checkout --track origin/astraea
## change version under ts-common to snapshot
sudo mvn clean package -Dmaven.test.skip=true
cd ..

## astraea spans and sleep injection
touch /local/astraea-spans/sleeps
mkdir -p /local/astraea-spans
sudo chmod ugo+rwx -R /local/astraea-spans
cp ../data/all-spans-tt /local/astraea-spans/spans

cd train-ticket
sudo docker-compose build
sudo docker-compose up -d
echo "Built tt and booted"
cd ../../

## workload generators
chmod +x eval/workload/tt-heyworkload.sh
chmod +x eval/workload/hey_linux_amd64

## span policy reset for experimental purposes
cat data/all-spans-tt > /local/astraea-spans/spans-orig

## run an experiment
cd eval/
mkdir -p astraea-results
python3 AstraeaEvaluator.py