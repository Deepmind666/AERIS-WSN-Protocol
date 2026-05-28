#!/bin/bash
# NS-3 Multi-Environment Publication Run
# 4 envs x 3 node counts x 30 seeds x 2 protocols + 4 envs x 30 seeds x 4 ablation = 1200 experiments
cd /home/ns3user/ns-allinone-3.40/ns-3.40
echo "START: $(date)" > /home/ns3user/ns3_multienv_run.log
./ns3 run "aeris-validation-standalone --runMultiEnv --output=/home/ns3user/ns3_multienv_publication_20260211.json" >> /home/ns3user/ns3_multienv_run.log 2>&1
echo "END: $(date)" >> /home/ns3user/ns3_multienv_run.log
echo "DONE" >> /home/ns3user/ns3_multienv_run.log
