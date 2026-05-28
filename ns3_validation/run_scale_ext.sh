#!/bin/bash
cd /home/ns3user/ns-allinone-3.40/ns-3.40
echo "START: $(date)" > /home/ns3user/ns3_scale_ext_run.log
echo "Running 5 node counts (50,100,200,300,500) x 4 envs x 30 seeds"
./ns3 run "aeris-validation-standalone --runMultiEnv --output=/home/ns3user/ns3_scale_extension_20260211.json" >> /home/ns3user/ns3_scale_ext_run.log 2>&1
echo "END: $(date)" >> /home/ns3user/ns3_scale_ext_run.log
echo "DONE" >> /home/ns3user/ns3_scale_ext_run.log
