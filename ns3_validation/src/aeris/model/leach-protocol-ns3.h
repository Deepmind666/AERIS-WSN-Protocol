/**
 * LEACH Protocol - NS-3 Implementation for Baseline Comparison
 *
 * Standard LEACH implementation with realistic channel model
 * for fair cross-validation comparison with AERIS
 *
 * Uses physics-based channel model:
 * - Log-distance path loss
 * - Shadow fading (log-normal)
 * - Multi-path fading (Rayleigh/Rician)
 * - SNR-based packet error rate
 */

#ifndef LEACH_PROTOCOL_NS3_H
#define LEACH_PROTOCOL_NS3_H

#include "ns3/application.h"
#include "ns3/event-id.h"
#include "ns3/ptr.h"
#include "ns3/socket.h"
#include "ns3/random-variable-stream.h"
#include "ns3/vector.h"
#include "ns3/mobility-model.h"
#include "ns3/node.h"
#include "ns3/simulator.h"
#include "ns3/log.h"
#include "realistic-channel-model.h"
#include <vector>
#include <map>
#include <cmath>

namespace ns3 {

/**
 * Standard LEACH Protocol Implementation with Realistic Channel Model
 */
class LeachProtocolNs3 : public Application
{
public:
    static TypeId GetTypeId (void);

    LeachProtocolNs3 ();
    virtual ~LeachProtocolNs3 ();

    void SetBaseStationPosition (Vector pos) { m_bsPosition = pos; }
    void SetClusterHeadProbability (double p) { m_chProb = p; }
    void SetInitialEnergy (double e) { m_initialEnergy = e; m_residualEnergy = e; }
    void SetDataPacketSize (uint32_t size) { m_dataPacketSize = size; }
    void SetNumRounds (uint32_t rounds) { m_totalRounds = rounds; }

    /**
     * Set the radio environment for channel model
     */
    void SetRadioEnvironment (RadioEnvironment env) { m_channelModel.SetEnvironment (env); }

    uint32_t GetPacketsSent () const { return m_packetsSent; }
    uint32_t GetPacketsDelivered () const { return m_packetsDelivered; }
    double GetTotalEnergyConsumed () const { return m_totalEnergyConsumed; }
    double GetResidualEnergy () const { return m_residualEnergy; }
    bool IsAlive () const { return m_residualEnergy > 0; }

    static void ResetGlobalStats ();
    static uint32_t GetGlobalPacketsDelivered () { return s_globalPacketsDelivered; }
    static uint32_t GetGlobalPacketsSent () { return s_globalPacketsSent; }
    static double GetGlobalPdr ();

    // Statistics for channel quality analysis
    static double GetGlobalAvgSnr ();
    static double GetGlobalAvgRssi ();

protected:
    virtual void DoDispose (void) override;

private:
    virtual void StartApplication (void) override;
    virtual void StopApplication (void) override;

    void StartRound ();
    bool ElectClusterHead ();  // Standard LEACH election
    void DataTransmission ();
    void Aggregation ();

    void ConsumeEnergy (double energy);
    double CalculateTxEnergy (double distance, uint32_t bits);
    double CalculateRxEnergy (uint32_t bits);
    double GetDistanceToBS ();
    Vector GetNodePosition ();

    /**
     * Estimate link quality using realistic channel model
     * @param targetPos Target position (CH or BS)
     * @return Success probability based on channel conditions
     */
    double EstimateLinkQuality (Vector targetPos);

    /**
     * Attempt packet transmission using realistic channel model
     * @param targetPos Target position
     * @return true if packet delivered successfully
     */
    bool AttemptTransmission (Vector targetPos);

    Ptr<Socket> m_socket;
    uint32_t m_nodeId;
    Vector m_position;
    Vector m_bsPosition;

    bool m_isClusterHead;
    double m_chProb;
    double m_initialEnergy;
    double m_residualEnergy;
    uint32_t m_dataPacketSize;
    uint32_t m_totalRounds;
    uint32_t m_currentRound;

    uint32_t m_packetsSent;
    uint32_t m_packetsDelivered;
    double m_totalEnergyConsumed;
    uint32_t m_roundsNotCH;  // For G set tracking

    // Energy model parameters (standard values from LEACH paper)
    static constexpr double E_ELEC = 50e-9;      // Electronics energy (J/bit)
    static constexpr double E_FS = 10e-12;       // Free space model (J/bit/m^2)
    static constexpr double E_MP = 0.0013e-12;   // Multi-path model (J/bit/m^4)
    static constexpr double E_DA = 5e-9;         // Data aggregation (J/bit)
    static constexpr double D_CROSSOVER = 87.7;  // Crossover distance (m)

    Ptr<UniformRandomVariable> m_random;

    // Realistic channel model
    RealisticChannelModel m_channelModel;

    // Global statistics
    static uint32_t s_globalPacketsSent;
    static uint32_t s_globalPacketsDelivered;
    static uint32_t s_totalNodes;
    static uint32_t s_currentRoundCHs;
    static uint32_t s_currentRoundNumber;

    // Channel quality statistics
    static double s_globalTotalSnr;
    static double s_globalTotalRssi;
    static uint32_t s_globalSnrSamples;
    static uint32_t s_globalRssiSamples;
};

// Static member initialization
uint32_t LeachProtocolNs3::s_globalPacketsSent = 0;
uint32_t LeachProtocolNs3::s_globalPacketsDelivered = 0;
uint32_t LeachProtocolNs3::s_totalNodes = 0;
uint32_t LeachProtocolNs3::s_currentRoundCHs = 0;
uint32_t LeachProtocolNs3::s_currentRoundNumber = 0;
double LeachProtocolNs3::s_globalTotalSnr = 0;
double LeachProtocolNs3::s_globalTotalRssi = 0;
uint32_t LeachProtocolNs3::s_globalSnrSamples = 0;
uint32_t LeachProtocolNs3::s_globalRssiSamples = 0;

NS_OBJECT_ENSURE_REGISTERED (LeachProtocolNs3);

TypeId
LeachProtocolNs3::GetTypeId (void)
{
    static TypeId tid = TypeId ("ns3::LeachProtocolNs3")
        .SetParent<Application> ()
        .SetGroupName ("Aeris")
        .AddConstructor<LeachProtocolNs3> ()
    ;
    return tid;
}

LeachProtocolNs3::LeachProtocolNs3 ()
    : m_socket (nullptr),
      m_nodeId (0),
      m_bsPosition (100, 200, 0),
      m_isClusterHead (false),
      m_chProb (0.1),
      m_initialEnergy (0.5),
      m_residualEnergy (0.5),
      m_dataPacketSize (512),
      m_totalRounds (200),
      m_currentRound (0),
      m_packetsSent (0),
      m_packetsDelivered (0),
      m_totalEnergyConsumed (0),
      m_roundsNotCH (0)
{
    m_random = CreateObject<UniformRandomVariable> ();

    // Configure channel model for indoor environment (typical WSN deployment)
    m_channelModel.SetEnvironment (RadioEnvironment::INDOOR_NLOS);
    m_channelModel.SetTxPower (0.0);  // 0 dBm (1 mW) - standard CC2420 setting
}

LeachProtocolNs3::~LeachProtocolNs3 ()
{
}

void
LeachProtocolNs3::DoDispose (void)
{
    m_socket = nullptr;
    Application::DoDispose ();
}

void
LeachProtocolNs3::ResetGlobalStats ()
{
    s_globalPacketsSent = 0;
    s_globalPacketsDelivered = 0;
    s_totalNodes = 0;
    s_currentRoundCHs = 0;
    s_currentRoundNumber = 0;
    s_globalTotalSnr = 0;
    s_globalTotalRssi = 0;
    s_globalSnrSamples = 0;
    s_globalRssiSamples = 0;
}

double
LeachProtocolNs3::GetGlobalPdr ()
{
    if (s_globalPacketsSent == 0) return 0.0;
    return static_cast<double>(s_globalPacketsDelivered) / s_globalPacketsSent;
}

double
LeachProtocolNs3::GetGlobalAvgSnr ()
{
    if (s_globalSnrSamples == 0) return 0.0;
    return s_globalTotalSnr / s_globalSnrSamples;
}

double
LeachProtocolNs3::GetGlobalAvgRssi ()
{
    if (s_globalRssiSamples == 0) return 0.0;
    return s_globalTotalRssi / s_globalRssiSamples;
}

void
LeachProtocolNs3::StartApplication (void)
{
    m_nodeId = GetNode ()->GetId ();
    m_residualEnergy = m_initialEnergy;
    m_position = GetNodePosition ();
    s_totalNodes++;

    double offset = m_random->GetValue (0, 0.1);
    Simulator::Schedule (Seconds (offset), &LeachProtocolNs3::StartRound, this);
}

void
LeachProtocolNs3::StopApplication (void)
{
    if (m_socket)
    {
        m_socket->Close ();
        m_socket = nullptr;
    }
}

double
LeachProtocolNs3::EstimateLinkQuality (Vector targetPos)
{
    // Use realistic channel model to estimate link quality
    LinkQualityIndicator lqi = m_channelModel.CalculateLinkQuality (
        m_position, targetPos, m_dataPacketSize);

    // Track statistics
    s_globalTotalSnr += lqi.snr;
    s_globalSnrSamples++;
    s_globalTotalRssi += lqi.rssi;
    s_globalRssiSamples++;

    return lqi.successProb;
}

bool
LeachProtocolNs3::AttemptTransmission (Vector targetPos)
{
    // Use realistic channel model for transmission decision
    return m_channelModel.TransmitPacket (m_position, targetPos, m_dataPacketSize);
}

bool
LeachProtocolNs3::ElectClusterHead ()
{
    // Standard LEACH threshold calculation
    // T(n) = p / (1 - p * (r mod (1/p)))
    // Only nodes in G (not CH in last 1/p rounds) can become CH

    if (!IsAlive ()) return false;

    int roundMod = static_cast<int>(1.0 / m_chProb);
    int r = m_currentRound % roundMod;

    // Check if node is in G (eligible set)
    if (m_roundsNotCH < roundMod)
    {
        if (r == 0) r = roundMod;
        double threshold = m_chProb / (1.0 - m_chProb * (r - 1));

        double rand = m_random->GetValue (0.0, 1.0);
        if (rand < threshold)
        {
            m_roundsNotCH = 0;  // Reset counter
            return true;
        }
    }

    m_roundsNotCH++;
    return false;
}

void
LeachProtocolNs3::StartRound ()
{
    if (!IsAlive () || m_currentRound >= m_totalRounds)
    {
        return;
    }

    m_currentRound++;

    // Track round progression and reset per-round CH count
    if (m_currentRound > s_currentRoundNumber)
    {
        s_currentRoundNumber = m_currentRound;
        s_currentRoundCHs = 0;
    }

    m_isClusterHead = ElectClusterHead ();

    if (m_isClusterHead)
    {
        s_currentRoundCHs++;  // Track global CH count
        // Broadcast CH announcement (50m range)
        ConsumeEnergy (CalculateTxEnergy (50.0, 256));
    }

    // Schedule data transmission
    Simulator::Schedule (MilliSeconds (300), &LeachProtocolNs3::DataTransmission, this);

    // Schedule aggregation (for CHs)
    Simulator::Schedule (MilliSeconds (700), &LeachProtocolNs3::Aggregation, this);

    // Schedule next round
    Simulator::Schedule (Seconds (1.0), &LeachProtocolNs3::StartRound, this);
}

void
LeachProtocolNs3::DataTransmission ()
{
    if (!IsAlive ()) return;

    // Every node sends one data packet per round
    m_packetsSent++;
    s_globalPacketsSent++;

    double distToDest;
    if (m_isClusterHead)
    {
        // CH sends directly to BS
        distToDest = GetDistanceToBS ();
    }
    else
    {
        // Regular node sends to CH (assume avg distance 30m)
        distToDest = 30.0;
    }

    ConsumeEnergy (CalculateTxEnergy (distToDest, m_dataPacketSize * 8));

    // NOTE: Packet delivery to BS is tracked in Aggregation phase
    // Individual node->CH transmission uses channel model
}

void
LeachProtocolNs3::Aggregation ()
{
    if (!IsAlive () || !m_isClusterHead) return;

    // Calculate actual cluster size based on current CHs in network
    uint32_t numCHs = (s_currentRoundCHs > 0) ? s_currentRoundCHs : 1;
    uint32_t clusterSize = s_totalNodes / numCHs;
    if (clusterSize < 1) clusterSize = 1;

    // Receive from cluster members
    if (clusterSize > 1)
    {
        ConsumeEnergy (CalculateRxEnergy (m_dataPacketSize * 8 * (clusterSize - 1)));
    }

    // Aggregation
    ConsumeEnergy (E_DA * clusterSize * m_dataPacketSize * 8);

    // Send to BS
    double distToBS = GetDistanceToBS ();
    uint32_t aggregatedSize = m_dataPacketSize + clusterSize * 64;
    ConsumeEnergy (CalculateTxEnergy (distToBS, aggregatedSize * 8));

    // =====================================================================
    // REALISTIC CHANNEL MODEL FOR PACKET DELIVERY
    // =====================================================================
    // LEACH disadvantages compared to AERIS:
    // 1. No CAS optimization - random CH may have poor link quality
    // 2. No gateway relay - long distance transmissions suffer more loss
    // 3. No link quality aware CH selection - CH may be in bad location
    // =====================================================================

    uint32_t deliveredCount = 0;

    for (uint32_t i = 0; i < clusterSize; i++)
    {
        // Phase 1: Internal cluster transmission (node -> CH)
        // Assume average distance of 30m for cluster members
        Vector memberPos = m_position;  // Approximate, actual positions vary
        memberPos.x += m_random->GetValue (-30, 30);
        memberPos.y += m_random->GetValue (-30, 30);

        // For LEACH, internal transmissions have typical indoor reliability
        // Using channel model for node->CH link
        bool internalSuccess = m_channelModel.TransmitPacket (
            memberPos, m_position, m_dataPacketSize);

        if (!internalSuccess)
        {
            // Packet lost in intra-cluster transmission
            continue;
        }

        // Phase 2: CH -> BS transmission using realistic channel model
        // LEACH has no gateway relay, so long-distance transmissions
        // directly use the channel model which accounts for:
        // - Path loss increases with d^n (n=3.5 for NLOS)
        // - Shadow fading (σ=6dB for indoor NLOS)
        // - Multi-path Rayleigh fading for NLOS
        // - SNR-based PER calculation

        bool chToBsSuccess = m_channelModel.TransmitPacket (
            m_position, m_bsPosition, aggregatedSize);

        if (chToBsSuccess)
        {
            deliveredCount++;
        }
    }

    s_globalPacketsDelivered += deliveredCount;
    m_packetsDelivered += deliveredCount;
}

void
LeachProtocolNs3::ConsumeEnergy (double energy)
{
    m_residualEnergy -= energy;
    m_totalEnergyConsumed += energy;
    if (m_residualEnergy < 0) m_residualEnergy = 0;
}

double
LeachProtocolNs3::CalculateTxEnergy (double distance, uint32_t bits)
{
    // Standard first-order radio energy model
    if (distance < D_CROSSOVER)
        return E_ELEC * bits + E_FS * bits * distance * distance;
    else
        return E_ELEC * bits + E_MP * bits * std::pow(distance, 4);
}

double
LeachProtocolNs3::CalculateRxEnergy (uint32_t bits)
{
    return E_ELEC * bits;
}

double
LeachProtocolNs3::GetDistanceToBS ()
{
    double dx = m_position.x - m_bsPosition.x;
    double dy = m_position.y - m_bsPosition.y;
    return std::sqrt (dx * dx + dy * dy);
}

Vector
LeachProtocolNs3::GetNodePosition ()
{
    Ptr<MobilityModel> mobility = GetNode ()->GetObject<MobilityModel> ();
    if (mobility)
        return mobility->GetPosition ();
    return Vector (0, 0, 0);
}

} // namespace ns3

#endif /* LEACH_PROTOCOL_NS3_H */
