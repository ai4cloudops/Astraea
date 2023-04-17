## remove previously injected delays
cat ../../data/all-spans-tt > /local/astraea-spans/states

sleep 2; 

echo "check line number afterwards should be empty"
cat /local/astraea-spans/states | grep 'inject'

## randomly generate number
randomNumber=$((1 + $RANDOM % 30))
# randomNumber=1
randomNumber=${entries[$x]}
if [[ "$randomNumber" -eq 5 ]]; then
randomNumber=1 
echo "queryinfo should not be there" 
fi

echo "Line number now $randomNumber"

## remove 1
sed -i "${randomNumber} s/1//" /local/astraea-spans/states
sed -i "${randomNumber} s/./inject-&/" /local/astraea-spans/states-sleeps
sleep 11

#echo "check line number"
#cat /local/astraea-spans/states | grep 'inject'
## get line to var e.g., inject-ts-order-service:calculateSoldTicket
svc_line=$(cat /local/astraea-spans/states | grep 'inject')
# echo "svcline $svc_line"
echo "delay injected to to $svc_line"