set -euo pipefail
cd '/mnt/c/Users/sshuser/AERIS-WSN-Protocol'
export NS3_ROOT=/home/ns3user/ns-allinone-3.40/ns-3.40
export BIN=/home/ns3user/ns-allinone-3.40/ns-3.40/build/scratch/ns3.40-aeris-validation-standalone-default
export PROTOCOLS='ABLATION'
export ENVS='outdoor_suburban outdoor_urban'
export NODES='50,100,200,300,500,800,1000'
bash ns3_validation/run_lcn26_focused_matrix.sh 2 '/mnt/c/Users/sshuser/AERIS-WSN-Protocol/ns3_validation/results/lcn26_ns3_ablation_fat_suburban_urban_20260501_011001'
mkdir -p '/mnt/c/Users/sshuser/AERIS-WSN-Protocol/ns3_validation/results/lcn26_ns3_ablation_fat_suburban_urban_20260501_011001/summary'
python3 ns3_validation/merge_lcn26_focused_results.py --input-dir '/mnt/c/Users/sshuser/AERIS-WSN-Protocol/ns3_validation/results/lcn26_ns3_ablation_fat_suburban_urban_20260501_011001/raw' --output-dir '/mnt/c/Users/sshuser/AERIS-WSN-Protocol/ns3_validation/results/lcn26_ns3_ablation_fat_suburban_urban_20260501_011001/summary'
echo "[DONE] FatMachine ablation split complete"