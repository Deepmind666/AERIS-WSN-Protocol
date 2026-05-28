#!/bin/bash
cd /home/ns3user/ns-allinone-3.40/ns-3.40
BIN="./build/scratch/ns3.40-aeris-validation-standalone-default"
export LD_LIBRARY_PATH="/home/ns3user/ns-allinone-3.40/ns-3.40/build/lib:$LD_LIBRARY_PATH"
ENVS=(indoor_office indoor_factory outdoor_urban outdoor_suburban)
echo "=== PEGASIS fix rerun ===" > /home/ns3user/pegasis_master.log
echo "Start: $(date)" >> /home/ns3user/pegasis_master.log
for ENV in "${ENVS[@]}"; do
  OUT="/home/ns3user/shard_PEGASIS_${ENV}.json"
  LOG="/home/ns3user/pegasis_${ENV}.log"
  echo "Starting PEGASIS $ENV" >> /home/ns3user/pegasis_master.log
  $BIN --runShard --protocol=PEGASIS --env=$ENV --nodes=50,100,200,300,500,800,1000 --output=$OUT > $LOG 2>&1 &
done
echo "Launched 4 PEGASIS shards at $(date)" >> /home/ns3user/pegasis_master.log
wait
echo "All 4 PEGASIS shards done at $(date)" >> /home/ns3user/pegasis_master.log
