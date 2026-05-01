set -euo pipefail
cd '/mnt/c/AERIS-WSN-Protocol'
cp ns3_validation/aeris-validation-standalone.cc /home/lkr/ns-allinone-3.40/ns-3.40/scratch/aeris-validation-standalone.cc
cd /home/lkr/ns-allinone-3.40/ns-3.40
./ns3 build
cd '/mnt/c/AERIS-WSN-Protocol'
export NS3_ROOT=/home/lkr/ns-allinone-3.40/ns-3.40
export BIN=/home/lkr/ns-allinone-3.40/ns-3.40/build/scratch/ns3.40-aeris-validation-standalone-default
export PROTOCOLS='ABLATION'
export ENVS='indoor_office indoor_factory'
export NODES='50,100,200,300,500,800,1000'
bash ns3_validation/run_lcn26_focused_matrix.sh 2 '/mnt/c/AERIS-WSN-Protocol/ns3_validation/results/lcn26_ns3_ablation_local_office_factory_20260501_010355'
mkdir -p '/mnt/c/AERIS-WSN-Protocol/ns3_validation/results/lcn26_ns3_ablation_local_office_factory_20260501_010355/summary'
python3 ns3_validation/merge_lcn26_focused_results.py --input-dir '/mnt/c/AERIS-WSN-Protocol/ns3_validation/results/lcn26_ns3_ablation_local_office_factory_20260501_010355/raw' --output-dir '/mnt/c/AERIS-WSN-Protocol/ns3_validation/results/lcn26_ns3_ablation_local_office_factory_20260501_010355/summary'
echo "[DONE] local ablation split complete"