/**
 * Comprehensive NS-3 Cross-Validation Example with Realistic Channel Model
 *
 * Runs AERIS and LEACH protocols across multiple configurations
 * for rigorous cross-validation with Python simulation results.
 *
 * Features:
 * - Physics-based channel model (path loss, shadow fading, multi-path)
 * - SNR-based packet error rate
 * - Link quality indicator tracking
 *
 * Usage:
 *   ./ns3 run "aeris-validation --protocol=AERIS --numNodes=100 --numRounds=200 --seed=42001"
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/aeris-protocol-full.h"
#include "ns3/leach-protocol-ns3.h"
#include <fstream>
#include <iomanip>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("AerisValidation");

// Results structure with channel quality metrics
struct ExperimentResult {
    std::string protocol;
    uint32_t numNodes;
    uint32_t numRounds;
    uint32_t seed;
    double pdr;
    double totalEnergy;
    double avgEnergyPerNode;
    uint32_t aliveNodes;
    uint32_t deadNodes;
    uint32_t packetsSent;
    uint32_t packetsDelivered;
    // Channel quality metrics
    double avgSnr;
    double avgRssi;
};

// Run single experiment
ExperimentResult RunExperiment (std::string protocol, uint32_t numNodes, uint32_t numRounds,
                                 uint32_t seed, double areaWidth, double areaHeight,
                                 bool enableCas, bool enableFairness, bool enableGateway)
{
    ExperimentResult result;
    result.protocol = protocol;
    result.numNodes = numNodes;
    result.numRounds = numRounds;
    result.seed = seed;

    // Set random seed
    RngSeedManager::SetSeed (seed);
    RngSeedManager::SetRun (1);

    // Check if using AERIS variant (AERIS, AERIS-FULL, AERIS-noCAS, etc.) or LEACH
    bool useAeris = (protocol.find("AERIS") == 0);

    // Reset global stats
    if (useAeris)
    {
        AerisProtocolFull::ResetGlobalStats ();
    }
    else
    {
        LeachProtocolNs3::ResetGlobalStats ();
    }

    // Create nodes
    NodeContainer sensorNodes;
    sensorNodes.Create (numNodes);

    // Setup mobility
    MobilityHelper mobility;
    mobility.SetPositionAllocator ("ns3::RandomRectanglePositionAllocator",
                                   "X", StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string (areaWidth) + "]"),
                                   "Y", StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string (areaHeight) + "]"));
    mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
    mobility.Install (sensorNodes);

    // BS position
    Vector bsPos (areaWidth / 2, areaHeight, 0);

    // Install Internet stack for socket support
    InternetStackHelper internet;
    internet.Install (sensorNodes);

    // Install protocol
    double simTime = numRounds + 10;

    if (useAeris)
    {
        for (uint32_t i = 0; i < sensorNodes.GetN (); ++i)
        {
            Ptr<AerisProtocolFull> app = CreateObject<AerisProtocolFull> ();
            app->SetBaseStationPosition (bsPos);
            app->SetClusterHeadProbability (0.1);
            app->SetInitialEnergy (0.5);
            app->SetDataPacketSize (512);
            app->SetNumRounds (numRounds);
            app->SetEnableCas (enableCas);
            app->SetEnableFairness (enableFairness);
            app->SetEnableGateway (enableGateway);
            sensorNodes.Get (i)->AddApplication (app);
            app->SetStartTime (Seconds (1.0));
            app->SetStopTime (Seconds (simTime));
        }
    }
    else  // LEACH
    {
        for (uint32_t i = 0; i < sensorNodes.GetN (); ++i)
        {
            Ptr<LeachProtocolNs3> app = CreateObject<LeachProtocolNs3> ();
            app->SetBaseStationPosition (bsPos);
            app->SetClusterHeadProbability (0.1);
            app->SetInitialEnergy (0.5);
            app->SetDataPacketSize (512);
            app->SetNumRounds (numRounds);
            sensorNodes.Get (i)->AddApplication (app);
            app->SetStartTime (Seconds (1.0));
            app->SetStopTime (Seconds (simTime));
        }
    }

    // Run simulation
    Simulator::Stop (Seconds (simTime + 1));
    Simulator::Run ();

    // Collect results
    double totalEnergy = 0;
    uint32_t aliveNodes = 0;
    uint32_t deadNodes = 0;

    for (uint32_t i = 0; i < sensorNodes.GetN (); ++i)
    {
        if (useAeris)
        {
            Ptr<AerisProtocolFull> app = DynamicCast<AerisProtocolFull> (sensorNodes.Get(i)->GetApplication(0));
            if (app)
            {
                totalEnergy += app->GetTotalEnergyConsumed ();
                if (app->IsAlive ()) aliveNodes++; else deadNodes++;
            }
        }
        else
        {
            Ptr<LeachProtocolNs3> app = DynamicCast<LeachProtocolNs3> (sensorNodes.Get(i)->GetApplication(0));
            if (app)
            {
                totalEnergy += app->GetTotalEnergyConsumed ();
                if (app->IsAlive ()) aliveNodes++; else deadNodes++;
            }
        }
    }

    // Get global PDR and channel quality statistics
    if (useAeris)
    {
        result.pdr = AerisProtocolFull::GetGlobalPdr ();
        result.packetsSent = AerisProtocolFull::GetGlobalPacketsSent ();
        result.packetsDelivered = AerisProtocolFull::GetGlobalPacketsDelivered ();
        result.avgSnr = AerisProtocolFull::GetGlobalAvgSnr ();
        result.avgRssi = AerisProtocolFull::GetGlobalAvgRssi ();
    }
    else
    {
        result.pdr = LeachProtocolNs3::GetGlobalPdr ();
        result.packetsSent = LeachProtocolNs3::GetGlobalPacketsSent ();
        result.packetsDelivered = LeachProtocolNs3::GetGlobalPacketsDelivered ();
        result.avgSnr = LeachProtocolNs3::GetGlobalAvgSnr ();
        result.avgRssi = LeachProtocolNs3::GetGlobalAvgRssi ();
    }

    result.totalEnergy = totalEnergy;
    result.avgEnergyPerNode = totalEnergy / numNodes;
    result.aliveNodes = aliveNodes;
    result.deadNodes = deadNodes;

    Simulator::Destroy ();

    return result;
}

void PrintResult (const ExperimentResult& r)
{
    std::cout << std::fixed << std::setprecision (4);
    std::cout << r.protocol << " | Nodes=" << r.numNodes
              << " | PDR=" << (r.pdr * 100) << "%"
              << " | Energy=" << (r.totalEnergy * 1000) << "mJ"
              << " | Alive=" << r.aliveNodes << "/" << r.numNodes
              << " | SNR=" << r.avgSnr << "dB"
              << " | RSSI=" << r.avgRssi << "dBm"
              << std::endl;
}

void ExportResults (const std::vector<ExperimentResult>& results, const std::string& filename)
{
    std::ofstream ofs (filename);
    ofs << "{\n  \"channel_model\": {\n";
    ofs << "    \"type\": \"realistic_physics_based\",\n";
    ofs << "    \"path_loss_model\": \"log_distance\",\n";
    ofs << "    \"shadow_fading\": \"log_normal\",\n";
    ofs << "    \"multipath_fading\": \"rayleigh_rician\",\n";
    ofs << "    \"modulation\": \"O-QPSK (IEEE 802.15.4)\",\n";
    ofs << "    \"radio\": \"CC2420\",\n";
    ofs << "    \"environment\": \"INDOOR_NLOS\",\n";
    ofs << "    \"path_loss_exponent\": 3.5,\n";
    ofs << "    \"shadow_fading_std_db\": 6.0,\n";
    ofs << "    \"tx_power_dbm\": 0.0,\n";
    ofs << "    \"rx_sensitivity_dbm\": -95.0\n";
    ofs << "  },\n";
    ofs << "  \"experiments\": [\n";

    for (size_t i = 0; i < results.size (); ++i)
    {
        const auto& r = results[i];
        ofs << "    {\n";
        ofs << "      \"protocol\": \"" << r.protocol << "\",\n";
        ofs << "      \"num_nodes\": " << r.numNodes << ",\n";
        ofs << "      \"num_rounds\": " << r.numRounds << ",\n";
        ofs << "      \"seed\": " << r.seed << ",\n";
        ofs << "      \"pdr\": " << std::fixed << std::setprecision (6) << r.pdr << ",\n";
        ofs << "      \"total_energy_mj\": " << std::setprecision (2) << (r.totalEnergy * 1000) << ",\n";
        ofs << "      \"avg_energy_mj\": " << (r.avgEnergyPerNode * 1000) << ",\n";
        ofs << "      \"alive_nodes\": " << r.aliveNodes << ",\n";
        ofs << "      \"dead_nodes\": " << r.deadNodes << ",\n";
        ofs << "      \"packets_sent\": " << r.packetsSent << ",\n";
        ofs << "      \"packets_delivered\": " << r.packetsDelivered << ",\n";
        ofs << "      \"avg_snr_db\": " << std::setprecision (2) << r.avgSnr << ",\n";
        ofs << "      \"avg_rssi_dbm\": " << r.avgRssi << "\n";
        ofs << "    }";
        if (i < results.size () - 1) ofs << ",";
        ofs << "\n";
    }

    ofs << "  ]\n}\n";
    ofs.close ();
}

int main (int argc, char *argv[])
{
    // Parameters
    std::string protocol = "AERIS";
    uint32_t numNodes = 100;
    uint32_t numRounds = 200;
    uint32_t seed = 42001;
    double areaWidth = 200.0;
    double areaHeight = 200.0;
    bool runAll = false;
    std::string outputFile = "ns3_validation_results.json";

    CommandLine cmd (__FILE__);
    cmd.AddValue ("protocol", "Protocol (AERIS or LEACH)", protocol);
    cmd.AddValue ("numNodes", "Number of nodes", numNodes);
    cmd.AddValue ("numRounds", "Number of rounds", numRounds);
    cmd.AddValue ("seed", "Random seed", seed);
    cmd.AddValue ("areaWidth", "Area width", areaWidth);
    cmd.AddValue ("areaHeight", "Area height", areaHeight);
    cmd.AddValue ("runAll", "Run comprehensive validation", runAll);
    cmd.AddValue ("output", "Output file", outputFile);
    cmd.Parse (argc, argv);

    std::cout << "================================================" << std::endl;
    std::cout << "NS-3 Cross-Validation with Realistic Channel Model" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "Channel Model: Log-distance path loss + Shadow fading + Multi-path" << std::endl;
    std::cout << "Environment: Indoor NLOS (n=3.5, σ=6dB)" << std::endl;
    std::cout << "Radio: CC2420 (0 dBm TX, -95 dBm sensitivity)" << std::endl;
    std::cout << "================================================" << std::endl;

    std::vector<ExperimentResult> allResults;

    if (runAll)
    {
        // Comprehensive validation suite
        std::vector<uint32_t> nodeCounts = {50, 100, 200, 300};
        std::vector<uint32_t> seeds = {42001, 42002, 42003, 42004, 42005};

        std::cout << "\n--- Scalability Comparison ---\n" << std::endl;

        for (uint32_t nodes : nodeCounts)
        {
            for (uint32_t s : seeds)
            {
                // AERIS
                auto r1 = RunExperiment ("AERIS", nodes, numRounds, s, areaWidth, areaHeight, true, true, true);
                allResults.push_back (r1);
                PrintResult (r1);

                // LEACH
                auto r2 = RunExperiment ("LEACH", nodes, numRounds, s, areaWidth, areaHeight, false, false, false);
                allResults.push_back (r2);
                PrintResult (r2);
            }
            std::cout << std::endl;
        }

        // Ablation study
        std::cout << "\n--- Ablation Study (100 nodes) ---\n" << std::endl;

        auto full = RunExperiment ("AERIS-FULL", 100, 200, 42001, 200, 200, true, true, true);
        allResults.push_back (full);
        PrintResult (full);

        auto noCas = RunExperiment ("AERIS-noCAS", 100, 200, 42001, 200, 200, false, true, true);
        allResults.push_back (noCas);
        PrintResult (noCas);

        auto noFairness = RunExperiment ("AERIS-noFair", 100, 200, 42001, 200, 200, true, false, true);
        allResults.push_back (noFairness);
        PrintResult (noFairness);

        auto noGateway = RunExperiment ("AERIS-noGW", 100, 200, 42001, 200, 200, true, true, false);
        allResults.push_back (noGateway);
        PrintResult (noGateway);
    }
    else
    {
        // Single experiment
        auto result = RunExperiment (protocol, numNodes, numRounds, seed, areaWidth, areaHeight,
                                      protocol == "AERIS", protocol == "AERIS", protocol == "AERIS");
        allResults.push_back (result);
        PrintResult (result);
    }

    // Export results
    ExportResults (allResults, outputFile);
    std::cout << "\nResults saved to: " << outputFile << std::endl;

    // Summary statistics
    if (runAll)
    {
        std::cout << "\n================================================" << std::endl;
        std::cout << "Summary Statistics" << std::endl;
        std::cout << "================================================" << std::endl;

        // Calculate averages for each protocol
        double aerisPdrSum = 0, leachPdrSum = 0;
        double aerisSnrSum = 0, leachSnrSum = 0;
        double aerisRssiSum = 0, leachRssiSum = 0;
        int aerisCount = 0, leachCount = 0;

        for (const auto& r : allResults)
        {
            if (r.protocol == "AERIS")
            {
                aerisPdrSum += r.pdr;
                aerisSnrSum += r.avgSnr;
                aerisRssiSum += r.avgRssi;
                aerisCount++;
            }
            else if (r.protocol == "LEACH")
            {
                leachPdrSum += r.pdr;
                leachSnrSum += r.avgSnr;
                leachRssiSum += r.avgRssi;
                leachCount++;
            }
        }

        if (aerisCount > 0 && leachCount > 0)
        {
            double aerisPdrAvg = aerisPdrSum / aerisCount;
            double leachPdrAvg = leachPdrSum / leachCount;
            double aerisSnrAvg = aerisSnrSum / aerisCount;
            double leachSnrAvg = leachSnrSum / leachCount;
            double aerisRssiAvg = aerisRssiSum / aerisCount;
            double leachRssiAvg = leachRssiSum / leachCount;
            double pdrImprovement = (aerisPdrAvg - leachPdrAvg) / leachPdrAvg * 100;

            std::cout << std::fixed << std::setprecision (2);
            std::cout << "\nPacket Delivery Ratio:" << std::endl;
            std::cout << "  AERIS avg PDR: " << (aerisPdrAvg * 100) << "%" << std::endl;
            std::cout << "  LEACH avg PDR: " << (leachPdrAvg * 100) << "%" << std::endl;
            std::cout << "  Improvement:   " << pdrImprovement << "%" << std::endl;

            std::cout << "\nChannel Quality (SNR):" << std::endl;
            std::cout << "  AERIS avg SNR: " << aerisSnrAvg << " dB" << std::endl;
            std::cout << "  LEACH avg SNR: " << leachSnrAvg << " dB" << std::endl;

            std::cout << "\nSignal Strength (RSSI):" << std::endl;
            std::cout << "  AERIS avg RSSI: " << aerisRssiAvg << " dBm" << std::endl;
            std::cout << "  LEACH avg RSSI: " << leachRssiAvg << " dBm" << std::endl;
        }
    }

    return 0;
}
