/**
 * AERIS Protocol Helper for NS-3
 *
 * Helper class for easy installation of AERIS protocol on NS-3 nodes
 */

#ifndef AERIS_HELPER_H
#define AERIS_HELPER_H

#include "ns3/object-factory.h"
#include "ns3/ipv4-address.h"
#include "ns3/node-container.h"
#include "ns3/application-container.h"
#include "ns3/aeris-protocol.h"

namespace ns3 {

/**
 * Helper class for AERIS protocol installation
 */
class AerisHelper
{
public:
    /**
     * Create an AerisHelper
     */
    AerisHelper ();

    /**
     * Set an attribute for the protocol
     * @param name Attribute name
     * @param value Attribute value
     */
    void SetAttribute (std::string name, const AttributeValue &value);

    /**
     * Install AERIS protocol on a single node
     * @param node The node to install on
     * @return ApplicationContainer with the installed application
     */
    ApplicationContainer Install (Ptr<Node> node) const;

    /**
     * Install AERIS protocol on multiple nodes
     * @param nodes The nodes to install on
     * @return ApplicationContainer with all installed applications
     */
    ApplicationContainer Install (NodeContainer nodes) const;

    /**
     * Set the base station node
     * @param bsNode The base station node
     */
    void SetBaseStation (Ptr<Node> bsNode);

    /**
     * Set the base station address
     * @param addr IPv4 address of base station
     */
    void SetBaseStationAddress (Ipv4Address addr);

    /**
     * Enable/disable CAS module
     * @param enable True to enable CAS
     */
    void EnableCas (bool enable);

    /**
     * Enable/disable fairness module
     * @param enable True to enable fairness
     */
    void EnableFairness (bool enable);

    /**
     * Enable/disable gateway module
     * @param enable True to enable gateway routing
     */
    void EnableGateway (bool enable);

    /**
     * Set safety profile
     * @param profile Safety profile (ROBUST, BALANCED, ENERGY_SAVING)
     */
    void SetSafetyProfile (AerisSafetyProfile profile);

    /**
     * Set cluster head probability
     * @param prob Probability [0, 1]
     */
    void SetClusterHeadProbability (double prob);

    /**
     * Set initial energy for all nodes
     * @param energy Initial energy in Joules
     */
    void SetInitialEnergy (double energy);

    /**
     * Set simulation duration
     * @param duration Total simulation time
     */
    void SetSimulationDuration (Time duration);

private:
    Ptr<Application> InstallPriv (Ptr<Node> node) const;
    ObjectFactory m_factory;
    Ipv4Address m_baseStationAddr;
    Ptr<Node> m_baseStation;
};

/**
 * Statistics collector for AERIS experiments
 */
class AerisStatsCollector
{
public:
    AerisStatsCollector ();

    /**
     * Collect statistics from all AERIS applications
     * @param apps ApplicationContainer with AERIS apps
     */
    void Collect (ApplicationContainer apps);

    /**
     * Get overall PDR
     * @return Packet Delivery Ratio
     */
    double GetOverallPdr () const;

    /**
     * Get total energy consumed
     * @return Total energy in Joules
     */
    double GetTotalEnergy () const;

    /**
     * Get average energy per node
     * @return Average energy in Joules
     */
    double GetAverageEnergy () const;

    /**
     * Get network lifetime (rounds until first death)
     * @return Number of rounds
     */
    uint32_t GetNetworkLifetime () const;

    /**
     * Get number of alive nodes
     * @return Number of nodes with residual energy > 0
     */
    uint32_t GetAliveNodes () const;

    /**
     * Get number of dead nodes
     * @return Number of nodes with residual energy <= 0
     */
    uint32_t GetDeadNodes () const;

    /**
     * Export statistics to JSON
     * @param filename Output filename
     */
    void ExportJson (std::string filename) const;

    /**
     * Export statistics to CSV
     * @param filename Output filename
     */
    void ExportCsv (std::string filename) const;

    /**
     * Print summary to stdout
     */
    void PrintSummary () const;

private:
    uint32_t m_totalPacketsSent;
    uint32_t m_totalPacketsReceived;
    double m_totalEnergy;
    uint32_t m_aliveNodes;
    uint32_t m_deadNodes;
    uint32_t m_networkLifetime;
    std::vector<double> m_nodeEnergies;
    std::vector<double> m_nodePdrs;
};

} // namespace ns3

#endif /* AERIS_HELPER_H */
