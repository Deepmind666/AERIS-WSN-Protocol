/**
 * AERIS Protocol Implementation for NS-3
 *
 * Adaptive Environment-aware Routing for IoT Sensors
 * Cross-validation module for verifying Python simulation results
 *
 * Author: AERIS Research Team
 * Date: 2026
 */

#ifndef AERIS_PROTOCOL_H
#define AERIS_PROTOCOL_H

#include "ns3/application.h"
#include "ns3/event-id.h"
#include "ns3/ptr.h"
#include "ns3/ipv4-address.h"
#include "ns3/traced-callback.h"
#include "ns3/random-variable-stream.h"
#include "ns3/energy-module.h"
#include "ns3/vector.h"
#include <vector>
#include <map>
#include <set>

namespace ns3 {

class Socket;
class Packet;

/**
 * Node state enumeration
 */
enum class AerisNodeState {
    NORMAL,           // Regular sensor node
    CLUSTER_HEAD,     // Selected as cluster head
    GATEWAY,          // Gateway node for multi-hop
    DEAD              // Energy depleted
};

/**
 * Safety profile enumeration
 */
enum class AerisSafetyProfile {
    ROBUST,           // High reliability, more energy
    BALANCED,         // Balanced tradeoff
    ENERGY_SAVING     // Lower reliability, less energy
};

/**
 * CAS (Cluster-head Adaptive Selection) scoring structure
 */
struct CasScore {
    double residualEnergy;    // Normalized residual energy [0,1]
    double linkQuality;       // Average link quality to neighbors
    double nodeDistance;      // Distance to base station
    double historicalLoad;    // Historical cluster head load
    double combinedScore;     // Weighted combination
};

/**
 * Cluster information structure
 */
struct ClusterInfo {
    uint32_t headId;
    std::vector<uint32_t> members;
    double avgEnergy;
    double avgLinkQuality;
};

/**
 * AERIS Protocol Application
 *
 * Implements the AERIS routing protocol with:
 * - CAS (Cluster-head Adaptive Selection)
 * - Multi-hop gateway coordination
 * - Fairness-aware load balancing
 * - Safety-margin transmission control
 */
class AerisProtocol : public Application
{
public:
    /**
     * Get the type ID
     */
    static TypeId GetTypeId (void);

    AerisProtocol ();
    virtual ~AerisProtocol ();

    // Configuration setters
    void SetClusterHeadProbability (double p);
    void SetSafetyProfile (AerisSafetyProfile profile);
    void SetEnableCas (bool enable);
    void SetEnableFairness (bool enable);
    void SetEnableGateway (bool enable);
    void SetBaseStationAddress (Ipv4Address addr);
    void SetMaxGatewayHops (uint32_t hops);

    // State getters
    AerisNodeState GetNodeState () const;
    double GetResidualEnergy () const;
    uint32_t GetPacketsSent () const;
    uint32_t GetPacketsReceived () const;
    double GetPdr () const;

    // Cluster operations
    bool IsClusterHead () const;
    uint32_t GetClusterId () const;
    std::vector<uint32_t> GetClusterMembers () const;

    // Statistics
    double GetTotalEnergyConsumed () const;
    uint32_t GetClusterHeadRotations () const;

protected:
    virtual void DoDispose (void);

private:
    virtual void StartApplication (void);
    virtual void StopApplication (void);

    // ==================== CAS Module ====================
    /**
     * Calculate CAS score for cluster head selection
     */
    CasScore CalculateCasScore ();

    /**
     * Perform cluster head election using CAS
     */
    void PerformClusterHeadElection ();

    /**
     * Announce cluster head status to neighbors
     */
    void AnnounceClusterHead ();

    /**
     * Join nearest cluster head
     */
    void JoinCluster ();

    // ==================== Gateway Module ====================
    /**
     * Select gateway nodes for multi-hop routing
     */
    void SelectGatewayNodes ();

    /**
     * Calculate gateway score based on position and energy
     */
    double CalculateGatewayScore (uint32_t nodeId);

    /**
     * Route data through gateway nodes
     */
    void RouteViaGateway (Ptr<Packet> packet);

    // ==================== Fairness Module ====================
    /**
     * Update historical load for fairness
     */
    void UpdateHistoricalLoad ();

    /**
     * Calculate fairness penalty for repeated CH selection
     */
    double CalculateFairnessPenalty ();

    /**
     * Balance cluster sizes
     */
    void BalanceClusterSizes ();

    // ==================== Safety Module ====================
    /**
     * Calculate safety margin for transmission
     */
    double CalculateSafetyMargin (double linkQuality);

    /**
     * Determine transmission power based on safety profile
     */
    double GetTransmissionPower (double distance, double linkQuality);

    /**
     * Check if transmission is safe
     */
    bool IsTransmissionSafe (double snr);

    // ==================== Communication ====================
    /**
     * Send sensor data to cluster head
     */
    void SendDataToClusterHead ();

    /**
     * Aggregate and forward data (for cluster heads)
     */
    void AggregateAndForward ();

    /**
     * Handle received packet
     */
    void HandleRead (Ptr<Socket> socket);

    /**
     * Schedule next round
     */
    void ScheduleNextRound ();

    /**
     * Start new round
     */
    void StartNewRound ();

    // ==================== Energy Model ====================
    /**
     * Consume energy for transmission
     */
    void ConsumeTransmissionEnergy (double distance, uint32_t packetSize);

    /**
     * Consume energy for reception
     */
    void ConsumeReceptionEnergy (uint32_t packetSize);

    /**
     * Consume energy for aggregation
     */
    void ConsumeAggregationEnergy (uint32_t numPackets);

    /**
     * Check if node is alive
     */
    bool IsAlive () const;

    // ==================== Link Quality ====================
    /**
     * Estimate link quality using RSSI/SNR
     */
    double EstimateLinkQuality (uint32_t neighborId);

    /**
     * Update neighbor table
     */
    void UpdateNeighborTable ();

    /**
     * Get distance to base station
     */
    double GetDistanceToBaseStation () const;

    // Member variables
    Ptr<Socket> m_socket;
    Ipv4Address m_baseStationAddr;
    uint16_t m_port;

    // Node state
    AerisNodeState m_state;
    uint32_t m_nodeId;
    double m_residualEnergy;
    double m_initialEnergy;
    Vector m_position;

    // Protocol parameters
    double m_clusterHeadProb;
    AerisSafetyProfile m_safetyProfile;
    bool m_enableCas;
    bool m_enableFairness;
    bool m_enableGateway;
    uint32_t m_maxGatewayHops;

    // Clustering
    uint32_t m_clusterId;
    uint32_t m_clusterHeadId;
    std::vector<uint32_t> m_clusterMembers;
    std::map<uint32_t, double> m_neighborLinkQuality;

    // Gateway routing
    std::vector<uint32_t> m_gatewayPath;
    bool m_isGateway;

    // Fairness tracking
    uint32_t m_clusterHeadCount;
    double m_historicalLoad;

    // Statistics
    uint32_t m_packetsSent;
    uint32_t m_packetsReceived;
    uint32_t m_packetsDropped;
    double m_totalEnergyConsumed;
    uint32_t m_currentRound;

    // Energy model parameters (CC2420 based)
    static const double E_ELEC;      // Electronics energy (50 nJ/bit)
    static const double E_FS;        // Free space (10 pJ/bit/m^2)
    static const double E_MP;        // Multi-path (0.0013 pJ/bit/m^4)
    static const double E_DA;        // Data aggregation (5 nJ/bit)
    static const double D_CROSSOVER; // Crossover distance

    // Timing
    Time m_roundDuration;
    Time m_dataInterval;
    EventId m_roundEvent;
    EventId m_dataEvent;

    // Random variables
    Ptr<UniformRandomVariable> m_random;

    // Tracing
    TracedCallback<uint32_t, uint32_t, double> m_pdrTrace;
    TracedCallback<uint32_t, double> m_energyTrace;
};

} // namespace ns3

#endif /* AERIS_PROTOCOL_H */
