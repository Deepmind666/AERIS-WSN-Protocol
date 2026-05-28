#\!/bin/bash
cd /home/ns3user/ns-allinone-3.40/ns-3.40
echo BUILD START
./ns3 build 2>&1 | tail -30
echo BUILD DONE
