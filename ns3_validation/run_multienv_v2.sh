#!/bin/bash
cd /home/ns3user/ns-allinone-3.40/ns-3.40
echo "START: $(date)" > /home/ns3user/ns3_multienv_run_v2.log
./ns3 run "aeris-validation-standalone --runMultiEnv --output=/home/ns3user/ns3_multienv_publication_v2_20260211.json" >> /home/ns3user/ns3_multienv_run_v2.log 2>&1
echo "END: $(date)" >> /home/ns3user/ns3_multienv_run_v2.log
echo "DONE" >> /home/ns3user/ns3_multienv_run_v2.log
