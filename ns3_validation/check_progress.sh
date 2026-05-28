#\!/bin/bash
ps aux | grep aeris | grep -v grep
tail -5 /home/ns3user/ns3_scale_ext_1000_20260211.log
wc -l /home/ns3user/ns3_scale_ext_1000_20260211.log
