echo "Running tt hey workload"
echo "workload dir: $1";
echo "duration: $2";
echo "qps: $3";
echo "workers: $4";

$1/hey_linux_amd64 -z $2s -q $3 -c $4 -m POST -H "Content-Type: application/json" -d '{"startingPlace": "Shang Hai", "endPlace": "Su Zhou"}' http://localhost:8080/api/v1/travelservice/trips/left