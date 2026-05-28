#\!/bin/bash
cd /home/ns3user/ns-allinone-3.40/ns-3.40
date > /home/ns3user/ns3_scale_ext_1000_20260211.log
./ns3 run "aeris-validation-standalone --runMultiEnv --output=/home/ns3user/ns3_scale_ext_1000_20260211.json" >> /home/ns3user/ns3_scale_ext_1000_20260211.log 2>&1
date >> /home/ns3user/ns3_scale_ext_1000_20260211.log
echo DONE >> /home/ns3user/ns3_scale_ext_1000_20260211.log
