/**
 * AERIS Protocol Implementation for NS-3
 *
 * Adaptive Environment-aware Routing for IoT Sensors
 * Cross-validation module for verifying Python simulation results
 */

#include "aeris-protocol.h"
#include "ns3/log.h"
#include "ns3/ipv4-address.h"
#include "ns3/nstime.h"
#include "ns3/inet-socket-address.h"
#include "ns3/socket.h"
#include "ns3/simulator.h"
#include "ns3/socket-factory.h"
#include "ns3/packet.h"
#include "ns3/uinteger.h"
#include "ns3/double.h"
#include "ns3/boolean.h"
#include "ns3/enum.h"
#include "ns3/mobility-model.h"
#include "ns3/node.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("AerisProtocol");
NS_OBJECT_ENSURE_REGISTERED (AerisProtocol);

// Energy model constants (based on CC2420 radio)
const double AerisProtocol::E_ELEC = 50e-9;       // 50 nJ/bit
const double AerisProtocol::E_FS = 10e-12;        // 10 pJ/bit/m^2
const double AerisProtocol::E_MP = 0.0013e-12;    // 0.0013 pJ/bit/m^4
const double AerisProtocol::E_DA = 5e-9;          // 5 nJ/bit
const double AerisProtocol::D_CROSSOVER = 87.7;   // Crossover distance in meters

TypeId
AerisProtocol::GetTypeId (void)
{
    static TypeId tid = TypeId ("ns3::AerisProtocol")
        .SetParent<Application> ()
        .SetGroupName ("Aeris")
        .AddConstructor<AerisProtocol> ()
        .AddAttribute ("ClusterHeadProbability",
                       "Probability of becoming cluster head",
                       DoubleValue (0.1),
                       MakeDoubleAccessor (&AerisProtocol::m_clusterHeadProb),
                       MakeDoubleChecker<double> (0.0, 1.0))
        .AddAttribute ("EnableCAS",
                       "Enable Cluster-head Adaptive Selection",
                       BooleanValue (true),
                       MakeBooleanAccessor (&AerisProtocol::m_enableCas),
                       MakeBooleanChecker ())
        .AddAttribute ("EnableFairness",
                       "Enable fairness-aware load balancing",
                       BooleanValue (true),
                       MakeBooleanAccessor (&AerisProtocol::m_enableFairness),
                       MakeBooleanChecker ())
        .AddAttribute ("EnableGateway",
                       "Enable multi-hop gateway routing",
                       BooleanValue (true),
                       MakeBooleanAccessor (&AerisProtocol::m_enableGateway),
                       MakeBooleanChecker ())
        .AddAttribute ("InitialEnergy",
                       "Initial energy in Joules",
                       DoubleValue (0.5),
                       MakeDoubleAccessor (&AerisProtocol::m_initialEnergy),
                       MakeDoubleChecker<double> ())
        .AddAttribute ("MaxGatewayHops",
                       "Maximum gateway hops to base station",
                       UintegerValue (3),
                       MakeUintegerAccessor (&AerisProtocol::m_maxGatewayHops),
                       MakeUintegerChecker<uint32_t> ())
        .AddAttribute ("RoundDuration",
                       "Duration of each round",
                       TimeValue (Seconds (1.0)),
                       MakeTimeAccessor (&AerisProtocol::m_roundDuration),
                       MakeTimeChecker ())
        .AddTraceSource ("Pdr",
                         "Packet delivery ratio trace",
                         MakeTraceSourceAccessor (&AerisProtocol::m_pdrTrace),
                         "ns3::AerisProtocol::PdrTracedCallback")
        .AddTraceSource ("Energy",
                         "Energy consumption trace",
                         MakeTraceSourceAccessor (&AerisProtocol::m_energyTrace),
                         "ns3::AerisProtocol::EnergyTracedCallback")
    ;
    return tid;
}

AerisProtocol::AerisProtocol ()
    : m_socket (0),
      m_port (9),
      m_state (AerisNodeState::NORMAL),
      m_nodeId (0),
      m_clusterHeadProb (0.1),
      m_safetyProfile (AerisSafetyProfile::ROBUST),
      m_enableCas (true),
      m_enableFairness (true),
      m_enableGateway (true),
      m_maxGatewayHops (3),
      m_clusterId (0),
      m_clusterHeadId (0),
      m_isGateway (false),
      m_clusterHeadCount (0),
      m_historicalLoad (0.0),
      m_packetsSent (0),
      m_packetsReceived (0),
      m_packetsDropped (0),
      m_totalEnergyConsumed (0.0),
      m_currentRound (0),
      m_roundDuration (Seconds (1.0)),
      m_dataInterval (MilliSeconds (100))
{
    NS_LOG_FUNCTION (this);
    m_random = CreateObject<UniformRandomVariable> ();
    m_residualEnergy = m_initialEnergy;
}

AerisProtocol::~AerisProtocol ()
{
    NS_LOG_FUNCTION (this);
    m_socket = 0;
}

void
AerisProtocol::DoDispose (void)
{
    NS_LOG_FUNCTION (this);
    Application::DoDispose ();
}

void
AerisProtocol::StartApplication (void)
{
    NS_LOG_FUNCTION (this);

    m_nodeId = GetNode ()->GetId ();
    m_residualEnergy = m_initialEnergy;

    // Get node position
    Ptr<MobilityModel> mobility = GetNode ()->GetObject<MobilityModel> ();
    if (mobility)
    {
        m_position = mobility->GetPosition ();
    }

    // Create socket for communication
    if (!m_socket)
    {
        TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
        m_socket = Socket::CreateSocket (GetNode (), tid);
        InetSocketAddress local = InetSocketAddress (Ipv4Address::GetAny (), m_port);
        m_socket->Bind (local);
    }

    m_socket->SetRecvCallback (MakeCallback (&AerisProtocol::HandleRead, this));

    // Start first round
    ScheduleNextRound ();
}

void
AerisProtocol::StopApplication (void)
{
    NS_LOG_FUNCTION (this);

    if (m_socket)
    {
        m_socket->Close ();
        m_socket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket>> ());
    }

    Simulator::Cancel (m_roundEvent);
    Simulator::Cancel (m_dataEvent);
}

// ==================== CAS Module Implementation ====================

CasScore
AerisProtocol::CalculateCasScore ()
{
    CasScore score;

    // 1. Residual energy factor (normalized)
    score.residualEnergy = m_residualEnergy / m_initialEnergy;

    // 2. Link quality factor (average to neighbors)
    double totalLinkQuality = 0.0;
    for (const auto& neighbor : m_neighborLinkQuality)
    {
        totalLinkQuality += neighbor.second;
    }
    score.linkQuality = m_neighborLinkQuality.empty() ? 0.5 :
                        totalLinkQuality / m_neighborLinkQuality.size();

    // 3. Distance factor (inverse normalized)
    double distToBS = GetDistanceToBaseStation ();
    double maxDist = 200.0;  // Assuming 200m max distance
    score.nodeDistance = 1.0 - std::min(distToBS / maxDist, 1.0);

    // 4. Historical load factor (fairness)
    if (m_enableFairness)
    {
        score.historicalLoad = CalculateFairnessPenalty ();
    }
    else
    {
        score.historicalLoad = 1.0;
    }

    // Combined score with weights
    // w1=0.4 (energy), w2=0.25 (link), w3=0.2 (distance), w4=0.15 (fairness)
    score.combinedScore = 0.4 * score.residualEnergy +
                          0.25 * score.linkQuality +
                          0.2 * score.nodeDistance +
                          0.15 * score.historicalLoad;

    return score;
}

void
AerisProtocol::PerformClusterHeadElection ()
{
    NS_LOG_FUNCTION (this);

    if (!IsAlive ())
    {
        m_state = AerisNodeState::DEAD;
        return;
    }

    double threshold;

    if (m_enableCas)
    {
        // CAS-based election
        CasScore score = CalculateCasScore ();

        // Adaptive threshold based on CAS score
        double baseThreshold = m_clusterHeadProb /
                              (1.0 - m_clusterHeadProb * (m_currentRound % (int)(1.0 / m_clusterHeadProb)));
        threshold = baseThreshold * score.combinedScore;
    }
    else
    {
        // Standard LEACH-style election
        threshold = m_clusterHeadProb /
                   (1.0 - m_clusterHeadProb * (m_currentRound % (int)(1.0 / m_clusterHeadProb)));
    }

    double randomValue = m_random->GetValue (0.0, 1.0);

    if (randomValue < threshold)
    {
        m_state = AerisNodeState::CLUSTER_HEAD;
        m_clusterHeadId = m_nodeId;
        m_clusterHeadCount++;
        AnnounceClusterHead ();

        NS_LOG_INFO ("Node " << m_nodeId << " elected as cluster head in round " << m_currentRound);
    }
    else
    {
        m_state = AerisNodeState::NORMAL;
    }
}

void
AerisProtocol::AnnounceClusterHead ()
{
    NS_LOG_FUNCTION (this);

    // Create cluster head announcement packet
    Ptr<Packet> packet = Create<Packet> (32);  // 32 bytes for CH announcement

    // Broadcast to all neighbors
    InetSocketAddress remote = InetSocketAddress (Ipv4Address::GetBroadcast (), m_port);
    m_socket->SendTo (packet, 0, remote);

    // Energy consumption for broadcast
    ConsumeTransmissionEnergy (50.0, 32);  // 50m broadcast range
}

void
AerisProtocol::JoinCluster ()
{
    NS_LOG_FUNCTION (this);

    if (m_state == AerisNodeState::CLUSTER_HEAD)
    {
        return;  // Already a cluster head
    }

    // Find nearest cluster head (would be populated from announcements)
    // For now, simplified implementation

    // Send join request to cluster head
    Ptr<Packet> packet = Create<Packet> (16);  // 16 bytes for join request

    // Energy consumption
    double distToCH = 30.0;  // Estimated distance to cluster head
    ConsumeTransmissionEnergy (distToCH, 16);
}

// ==================== Gateway Module Implementation ====================

void
AerisProtocol::SelectGatewayNodes ()
{
    NS_LOG_FUNCTION (this);

    if (!m_enableGateway || m_state != AerisNodeState::CLUSTER_HEAD)
    {
        return;
    }

    // Gateway selection based on position and energy
    // Nodes between CH and BS with good link quality

    double distToBS = GetDistanceToBaseStation ();

    if (distToBS > D_CROSSOVER)
    {
        // Need gateway nodes for long-distance transmission
        m_isGateway = true;

        NS_LOG_INFO ("Node " << m_nodeId << " selected as gateway, dist to BS: " << distToBS);
    }
}

double
AerisProtocol::CalculateGatewayScore (uint32_t nodeId)
{
    // Score based on:
    // - Position (closer to direct path to BS)
    // - Residual energy
    // - Link quality

    double positionScore = 0.5;  // Simplified
    double energyScore = m_residualEnergy / m_initialEnergy;

    return 0.6 * energyScore + 0.4 * positionScore;
}

void
AerisProtocol::RouteViaGateway (Ptr<Packet> packet)
{
    NS_LOG_FUNCTION (this << packet);

    if (m_gatewayPath.empty ())
    {
        // Direct transmission to BS
        InetSocketAddress remote = InetSocketAddress (m_baseStationAddr, m_port);
        m_socket->SendTo (packet, 0, remote);

        double distToBS = GetDistanceToBaseStation ();
        ConsumeTransmissionEnergy (distToBS, packet->GetSize ());
    }
    else
    {
        // Multi-hop via gateways
        // Route to first gateway in path
        // Simplified: energy is already lower due to shorter hops

        double hopDistance = 50.0;  // Approximate hop distance
        ConsumeTransmissionEnergy (hopDistance, packet->GetSize ());
    }

    m_packetsSent++;
}

// ==================== Fairness Module Implementation ====================

void
AerisProtocol::UpdateHistoricalLoad ()
{
    // Exponential moving average of cluster head duty
    double alpha = 0.3;
    double currentLoad = (m_state == AerisNodeState::CLUSTER_HEAD) ? 1.0 : 0.0;
    m_historicalLoad = alpha * currentLoad + (1.0 - alpha) * m_historicalLoad;
}

double
AerisProtocol::CalculateFairnessPenalty ()
{
    // Penalize nodes that have been CH frequently
    // Returns value in [0, 1] where lower = more penalty

    double expectedTurns = m_currentRound * m_clusterHeadProb;
    double actualTurns = m_clusterHeadCount;

    if (actualTurns <= expectedTurns)
    {
        return 1.0;  // No penalty
    }
    else
    {
        double excess = actualTurns - expectedTurns;
        return std::exp(-0.5 * excess);  // Exponential penalty
    }
}

void
AerisProtocol::BalanceClusterSizes ()
{
    NS_LOG_FUNCTION (this);

    // Redistribute nodes if cluster sizes are uneven
    // Only applicable for cluster heads

    if (m_state != AerisNodeState::CLUSTER_HEAD)
    {
        return;
    }

    // Log cluster size for monitoring
    NS_LOG_INFO ("Cluster " << m_nodeId << " has " << m_clusterMembers.size () << " members");
}

// ==================== Safety Module Implementation ====================

double
AerisProtocol::CalculateSafetyMargin (double linkQuality)
{
    // Safety margin based on profile and link quality
    double baseMargin;

    switch (m_safetyProfile)
    {
        case AerisSafetyProfile::ROBUST:
            baseMargin = 6.0;  // 6 dB margin
            break;
        case AerisSafetyProfile::BALANCED:
            baseMargin = 3.0;  // 3 dB margin
            break;
        case AerisSafetyProfile::ENERGY_SAVING:
            baseMargin = 1.0;  // 1 dB margin
            break;
        default:
            baseMargin = 3.0;
    }

    // Adjust based on link quality
    if (linkQuality < 0.5)
    {
        baseMargin *= 1.5;  // Increase margin for poor links
    }

    return baseMargin;
}

double
AerisProtocol::GetTransmissionPower (double distance, double linkQuality)
{
    // Transmission power with safety margin
    double safetyMargin = CalculateSafetyMargin (linkQuality);

    // Path loss model (Log-distance with shadowing)
    double pathLossExponent = 3.5;
    double referenceDistance = 1.0;
    double referenceLoss = 40.0;  // dB at 1m

    double pathLoss = referenceLoss + 10.0 * pathLossExponent * std::log10 (distance / referenceDistance);

    // Required TX power = RX sensitivity + path loss + safety margin
    double rxSensitivity = -95.0;  // dBm
    double txPower = rxSensitivity + pathLoss + safetyMargin;

    // Clamp to valid range
    return std::min (std::max (txPower, -25.0), 0.0);  // -25 to 0 dBm
}

bool
AerisProtocol::IsTransmissionSafe (double snr)
{
    // Check if SNR is above threshold for reliable communication
    double minSnr;

    switch (m_safetyProfile)
    {
        case AerisSafetyProfile::ROBUST:
            minSnr = 10.0;  // 10 dB for robust
            break;
        case AerisSafetyProfile::BALANCED:
            minSnr = 6.0;   // 6 dB for balanced
            break;
        case AerisSafetyProfile::ENERGY_SAVING:
            minSnr = 3.0;   // 3 dB for energy saving
            break;
        default:
            minSnr = 6.0;
    }

    return snr >= minSnr;
}

// ==================== Communication ====================

void
AerisProtocol::SendDataToClusterHead ()
{
    NS_LOG_FUNCTION (this);

    if (!IsAlive () || m_state == AerisNodeState::CLUSTER_HEAD)
    {
        return;
    }

    // Create data packet (512 bytes typical sensor data)
    Ptr<Packet> packet = Create<Packet> (512);

    // Send to cluster head
    double distToCH = 30.0;  // Estimated
    double linkQuality = 0.8;

    if (IsTransmissionSafe (10.0))  // Simplified SNR check
    {
        // InetSocketAddress remote = InetSocketAddress (chAddress, m_port);
        // m_socket->SendTo (packet, 0, remote);

        ConsumeTransmissionEnergy (distToCH, 512);
        m_packetsSent++;
    }
    else
    {
        m_packetsDropped++;
        NS_LOG_WARN ("Node " << m_nodeId << " dropped packet due to unsafe channel");
    }
}

void
AerisProtocol::AggregateAndForward ()
{
    NS_LOG_FUNCTION (this);

    if (m_state != AerisNodeState::CLUSTER_HEAD)
    {
        return;
    }

    // Aggregate data from cluster members
    uint32_t numMembers = m_clusterMembers.size ();
    uint32_t aggregatedSize = 512 + numMembers * 64;  // Header + compressed data

    // Aggregation energy
    ConsumeAggregationEnergy (numMembers);

    // Create aggregated packet
    Ptr<Packet> packet = Create<Packet> (aggregatedSize);

    // Forward to base station (directly or via gateway)
    RouteViaGateway (packet);

    NS_LOG_INFO ("CH " << m_nodeId << " aggregated and forwarded " << numMembers << " packets");
}

void
AerisProtocol::HandleRead (Ptr<Socket> socket)
{
    NS_LOG_FUNCTION (this << socket);

    Ptr<Packet> packet;
    Address from;

    while ((packet = socket->RecvFrom (from)))
    {
        m_packetsReceived++;

        // Energy for reception
        ConsumeReceptionEnergy (packet->GetSize ());

        // Process packet based on type
        // (Simplified - would parse header in real implementation)

        if (m_state == AerisNodeState::CLUSTER_HEAD)
        {
            // Add to aggregation buffer
            m_clusterMembers.push_back (0);  // Placeholder
        }
    }
}

void
AerisProtocol::ScheduleNextRound ()
{
    m_roundEvent = Simulator::Schedule (m_roundDuration, &AerisProtocol::StartNewRound, this);
}

void
AerisProtocol::StartNewRound ()
{
    NS_LOG_FUNCTION (this);

    m_currentRound++;

    if (!IsAlive ())
    {
        m_state = AerisNodeState::DEAD;
        return;
    }

    // Reset round state
    m_clusterMembers.clear ();
    m_gatewayPath.clear ();

    // Phase 1: Cluster head election
    PerformClusterHeadElection ();

    // Phase 2: Cluster formation
    Simulator::Schedule (MilliSeconds (100), &AerisProtocol::JoinCluster, this);

    // Phase 3: Gateway selection (if enabled)
    if (m_enableGateway)
    {
        Simulator::Schedule (MilliSeconds (200), &AerisProtocol::SelectGatewayNodes, this);
    }

    // Phase 4: Data transmission
    Simulator::Schedule (MilliSeconds (300), &AerisProtocol::SendDataToClusterHead, this);

    // Phase 5: Aggregation and forwarding
    Simulator::Schedule (MilliSeconds (500), &AerisProtocol::AggregateAndForward, this);

    // Update fairness metrics
    UpdateHistoricalLoad ();

    // Trace outputs
    m_pdrTrace (m_nodeId, m_currentRound, GetPdr ());
    m_energyTrace (m_nodeId, m_totalEnergyConsumed);

    // Schedule next round
    ScheduleNextRound ();
}

// ==================== Energy Model ====================

void
AerisProtocol::ConsumeTransmissionEnergy (double distance, uint32_t packetSize)
{
    double energy;
    uint32_t bits = packetSize * 8;

    if (distance < D_CROSSOVER)
    {
        // Free space model
        energy = E_ELEC * bits + E_FS * bits * distance * distance;
    }
    else
    {
        // Multi-path model
        energy = E_ELEC * bits + E_MP * bits * std::pow (distance, 4);
    }

    m_residualEnergy -= energy;
    m_totalEnergyConsumed += energy;

    NS_LOG_DEBUG ("TX energy: " << energy * 1e6 << " uJ, remaining: " << m_residualEnergy);
}

void
AerisProtocol::ConsumeReceptionEnergy (uint32_t packetSize)
{
    uint32_t bits = packetSize * 8;
    double energy = E_ELEC * bits;

    m_residualEnergy -= energy;
    m_totalEnergyConsumed += energy;

    NS_LOG_DEBUG ("RX energy: " << energy * 1e6 << " uJ");
}

void
AerisProtocol::ConsumeAggregationEnergy (uint32_t numPackets)
{
    double energy = E_DA * numPackets * 512 * 8;  // 512 bytes per packet

    m_residualEnergy -= energy;
    m_totalEnergyConsumed += energy;

    NS_LOG_DEBUG ("Aggregation energy: " << energy * 1e6 << " uJ");
}

bool
AerisProtocol::IsAlive () const
{
    return m_residualEnergy > 0.0;
}

// ==================== Link Quality ====================

double
AerisProtocol::EstimateLinkQuality (uint32_t neighborId)
{
    // Would be based on RSSI measurements in real implementation
    // Simplified: use distance-based estimation

    return 0.8;  // Placeholder
}

void
AerisProtocol::UpdateNeighborTable ()
{
    NS_LOG_FUNCTION (this);

    // Would discover neighbors and update link quality
    // Simplified for cross-validation
}

double
AerisProtocol::GetDistanceToBaseStation () const
{
    // Calculate Euclidean distance to BS at (100, 200)
    Vector bsPosition (100.0, 200.0, 0.0);

    double dx = m_position.x - bsPosition.x;
    double dy = m_position.y - bsPosition.y;

    return std::sqrt (dx * dx + dy * dy);
}

// ==================== Getters ====================

void AerisProtocol::SetClusterHeadProbability (double p) { m_clusterHeadProb = p; }
void AerisProtocol::SetSafetyProfile (AerisSafetyProfile profile) { m_safetyProfile = profile; }
void AerisProtocol::SetEnableCas (bool enable) { m_enableCas = enable; }
void AerisProtocol::SetEnableFairness (bool enable) { m_enableFairness = enable; }
void AerisProtocol::SetEnableGateway (bool enable) { m_enableGateway = enable; }
void AerisProtocol::SetBaseStationAddress (Ipv4Address addr) { m_baseStationAddr = addr; }
void AerisProtocol::SetMaxGatewayHops (uint32_t hops) { m_maxGatewayHops = hops; }

AerisNodeState AerisProtocol::GetNodeState () const { return m_state; }
double AerisProtocol::GetResidualEnergy () const { return m_residualEnergy; }
uint32_t AerisProtocol::GetPacketsSent () const { return m_packetsSent; }
uint32_t AerisProtocol::GetPacketsReceived () const { return m_packetsReceived; }

double
AerisProtocol::GetPdr () const
{
    if (m_packetsSent == 0)
    {
        return 0.0;
    }
    return static_cast<double> (m_packetsReceived) / m_packetsSent;
}

bool AerisProtocol::IsClusterHead () const { return m_state == AerisNodeState::CLUSTER_HEAD; }
uint32_t AerisProtocol::GetClusterId () const { return m_clusterId; }
std::vector<uint32_t> AerisProtocol::GetClusterMembers () const { return m_clusterMembers; }
double AerisProtocol::GetTotalEnergyConsumed () const { return m_totalEnergyConsumed; }
uint32_t AerisProtocol::GetClusterHeadRotations () const { return m_clusterHeadCount; }

} // namespace ns3
