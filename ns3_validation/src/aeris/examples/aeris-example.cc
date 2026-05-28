/**
 * AERIS Protocol Example for NS-3 3.40
 *
 * Simplified example for cross-validation
 *
 * Usage:
 *   ./ns3 run "aeris-example --numNodes=100 --simTime=200 --seed=42001"
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/energy-module.h"
#include "ns3/aeris-protocol.h"
#include "ns3/aeris-helper.h"
#include <fstream>
#include <iomanip>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("AerisExample");

int
main (int argc, char *argv[])
{
    // Default parameters
    uint32_t numNodes = 100;
    double simTime = 200.0;
    uint32_t seed = 42001;
    double areaWidth = 200.0;
    double areaHeight = 200.0;
    std::string outputFile = "aeris_ns3_results.json";
    bool enableCas = true;
    bool enableFairness = true;
    bool enableGateway = true;

    // Command line arguments
    CommandLine cmd (__FILE__);
    cmd.AddValue ("numNodes", "Number of sensor nodes", numNodes);
    cmd.AddValue ("simTime", "Simulation time in seconds", simTime);
    cmd.AddValue ("seed", "Random seed", seed);
    cmd.AddValue ("areaWidth", "Network area width (m)", areaWidth);
    cmd.AddValue ("areaHeight", "Network area height (m)", areaHeight);
    cmd.AddValue ("outputFile", "Output JSON file", outputFile);
    cmd.AddValue ("enableCas", "Enable CAS module", enableCas);
    cmd.AddValue ("enableFairness", "Enable fairness module", enableFairness);
    cmd.AddValue ("enableGateway", "Enable gateway module", enableGateway);
    cmd.Parse (argc, argv);

    // Set random seed
    RngSeedManager::SetSeed (seed);
    RngSeedManager::SetRun (1);

    std::cout << "========================================" << std::endl;
    std::cout << "AERIS NS-3 Cross-Validation Experiment" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Nodes:      " << numNodes << std::endl;
    std::cout << "Area:       " << areaWidth << "x" << areaHeight << " m" << std::endl;
    std::cout << "Sim Time:   " << simTime << " s" << std::endl;
    std::cout << "Seed:       " << seed << std::endl;
    std::cout << "========================================" << std::endl;

    // Create nodes
    NodeContainer sensorNodes;
    sensorNodes.Create (numNodes);

    // Create base station
    NodeContainer baseStation;
    baseStation.Create (1);

    // Setup mobility (random uniform distribution)
    MobilityHelper mobility;

    // Sensor nodes: random position in area
    mobility.SetPositionAllocator ("ns3::RandomRectanglePositionAllocator",
                                   "X", StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string (areaWidth) + "]"),
                                   "Y", StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string (areaHeight) + "]"));
    mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
    mobility.Install (sensorNodes);

    // Base station: center top
    Ptr<ListPositionAllocator> bsPosition = CreateObject<ListPositionAllocator> ();
    bsPosition->Add (Vector (areaWidth / 2, areaHeight, 0.0));
    mobility.SetPositionAllocator (bsPosition);
    mobility.Install (baseStation);

    // Setup Internet stack
    InternetStackHelper internet;
    NodeContainer allNodes;
    allNodes.Add (sensorNodes);
    allNodes.Add (baseStation);
    internet.Install (allNodes);

    // Create point-to-point links (simplified network)
    // In a full implementation, this would use LR-WPAN or WiFi

    // Get base station address (simplified)
    Ipv4Address bsAddress ("10.1.1.1");

    // Setup AERIS protocol
    AerisHelper aerisHelper;
    aerisHelper.SetBaseStationAddress (bsAddress);
    aerisHelper.EnableCas (enableCas);
    aerisHelper.EnableFairness (enableFairness);
    aerisHelper.EnableGateway (enableGateway);
    aerisHelper.SetClusterHeadProbability (0.1);
    aerisHelper.SetInitialEnergy (0.5);

    // Install on sensor nodes
    ApplicationContainer aerisApps = aerisHelper.Install (sensorNodes);
    aerisApps.Start (Seconds (1.0));
    aerisApps.Stop (Seconds (simTime));

    // Run simulation
    std::cout << "\nRunning simulation..." << std::endl;
    Simulator::Stop (Seconds (simTime + 1.0));
    Simulator::Run ();

    // Collect statistics
    AerisStatsCollector stats;
    stats.Collect (aerisApps);

    // Print results
    std::cout << "\n";
    stats.PrintSummary ();

    // Export results
    stats.ExportJson (outputFile);
    std::cout << "\nResults saved to: " << outputFile << std::endl;

    // Cleanup
    Simulator::Destroy ();

    return 0;
}
