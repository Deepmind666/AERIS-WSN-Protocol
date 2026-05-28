/**
 * NS-3 AERIS Validation with Realistic Channel Model
 * Self-contained validation example
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <map>
#include <limits>
#include <algorithm>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("AerisValidation");

// ===================== Realistic Channel Model =====================

enum class RadioEnvironment { FREE_SPACE, INDOOR_LOS, INDOOR_NLOS, OUTDOOR_URBAN, INDUSTRIAL, OUTDOOR_SUBURBAN };

struct LinkQualityIndicator {
    double rssi, snr, lqi, per, successProb;
    bool isReliable;
};

class RealisticChannelModel
{
public:
    RealisticChannelModel ()
        : m_environment (RadioEnvironment::INDOOR_NLOS), m_pathLossExponent (3.5),
          m_referencePathLoss (45.0), m_shadowingStdDev (6.0), m_referenceDistance (1.0),
          m_txPowerDbm (10.0), m_rxSensitivityDbm (-95.0), m_noiseFigureDb (6.0)
    {
        m_thermalNoiseDbm = WattsToDbm (1.38e-23 * 290 * 2e6);
        m_shadowingRng = CreateObject<NormalRandomVariable> ();
        m_shadowingRng->SetAttribute ("Mean", DoubleValue (0.0));
        m_shadowingRng->SetAttribute ("Variance", DoubleValue (36.0));
        m_uniformRng = CreateObject<UniformRandomVariable> ();
        m_rayleighRng = CreateObject<ExponentialRandomVariable> ();
        m_rayleighRng->SetAttribute ("Mean", DoubleValue (1.0));
    }

    void SetEnvironment (RadioEnvironment env) {
        m_environment = env;
        switch (env) {
            case RadioEnvironment::FREE_SPACE: m_pathLossExponent = 2.0; m_referencePathLoss = 40.0; m_shadowingStdDev = 0.0; break;
            case RadioEnvironment::INDOOR_LOS: m_pathLossExponent = 2.0; m_referencePathLoss = 40.0; m_shadowingStdDev = 4.5; break;
            case RadioEnvironment::INDOOR_NLOS: m_pathLossExponent = 3.5; m_referencePathLoss = 45.0; m_shadowingStdDev = 6.0; break;
            case RadioEnvironment::OUTDOOR_URBAN: m_pathLossExponent = 3.4; m_referencePathLoss = 44.0; m_shadowingStdDev = 12.0; break;
            case RadioEnvironment::INDUSTRIAL: m_pathLossExponent = 2.7; m_referencePathLoss = 45.0; m_shadowingStdDev = 8.5; break;
            case RadioEnvironment::OUTDOOR_SUBURBAN: m_pathLossExponent = 2.8; m_referencePathLoss = 38.0; m_shadowingStdDev = 7.5; break;
        }
        m_shadowingRng->SetAttribute ("Variance", DoubleValue (m_shadowingStdDev * m_shadowingStdDev));
    }

    void SetTxPower (double txPowerDbm) { m_txPowerDbm = std::max (-25.0, std::min (10.0, txPowerDbm)); }

    LinkQualityIndicator CalculateLinkQuality (Vector txPos, Vector rxPos, uint32_t packetSize) {
        LinkQualityIndicator lqi;
        double dx = txPos.x - rxPos.x, dy = txPos.y - rxPos.y, dz = txPos.z - rxPos.z;
        double distance = std::sqrt (dx*dx + dy*dy + dz*dz);
        if (distance < 0.1) distance = 0.1;

        double pathLoss = m_referencePathLoss + 10.0 * m_pathLossExponent * std::log10 (distance / m_referenceDistance);
        double shadowFading = m_shadowingRng->GetValue ();
        double multipathFading = (m_environment == RadioEnvironment::FREE_SPACE || m_environment == RadioEnvironment::INDOOR_LOS) ?
            (std::pow(10.0, 0.6) + m_rayleighRng->GetValue()) / (1.0 + std::pow(10.0, 0.6)) : m_rayleighRng->GetValue();

        lqi.rssi = m_txPowerDbm - pathLoss - shadowFading;
        double rssiLinear = DbmToWatts (lqi.rssi) * multipathFading;
        lqi.rssi = WattsToDbm (rssiLinear);
        lqi.snr = lqi.rssi - (m_thermalNoiseDbm + m_noiseFigureDb);

        // PER calculation
        double snrLinear = std::pow (10.0, lqi.snr / 10.0);
        double ber = (lqi.snr < 0) ? 0.5 : ((lqi.snr > 20) ? 1e-10 : 0.5 * std::erfc (std::sqrt (snrLinear * 8)));
        lqi.per = std::min (1.0, std::max (0.0, 1.0 - std::pow (1.0 - ber, packetSize * 8)));
        lqi.successProb = 1.0 - lqi.per;
        lqi.isReliable = (lqi.snr > 8.0);
        return lqi;
    }

    bool TransmitPacket (Vector txPos, Vector rxPos, uint32_t packetSize) {
        LinkQualityIndicator lqi = CalculateLinkQuality (txPos, rxPos, packetSize);
        if (lqi.rssi < m_rxSensitivityDbm) return false;
        return m_uniformRng->GetValue (0.0, 1.0) > lqi.per;
    }

private:
    RadioEnvironment m_environment;
    double m_pathLossExponent, m_referencePathLoss, m_shadowingStdDev, m_referenceDistance;
    double m_txPowerDbm, m_rxSensitivityDbm, m_noiseFigureDb, m_thermalNoiseDbm;
    Ptr<NormalRandomVariable> m_shadowingRng;
    Ptr<UniformRandomVariable> m_uniformRng;
    Ptr<ExponentialRandomVariable> m_rayleighRng;

    double DbmToWatts (double dbm) { return std::pow (10.0, (dbm - 30.0) / 10.0); }
    double WattsToDbm (double watts) { return 10.0 * std::log10 (watts) + 30.0; }
};

// ===================== AERIS Protocol =====================

class AerisProtocolFull : public Application
{
public:
    static TypeId GetTypeId (void) {
        static TypeId tid = TypeId ("ns3::AerisProtocolFull").SetParent<Application> ().SetGroupName ("Aeris").AddConstructor<AerisProtocolFull> ();
        return tid;
    }

    AerisProtocolFull () : m_nodeId(0), m_bsPosition(100,200,0), m_isClusterHead(false), m_chProb(0.1),
        m_initialEnergy(2.0), m_residualEnergy(2.0), m_dataPacketSize(512), m_totalRounds(300),
        m_currentRound(0), m_packetsSent(0), m_packetsDelivered(0), m_totalEnergyConsumed(0),
        m_enableCas(true), m_enableFairness(true), m_enableGateway(true)
    {
        m_random = CreateObject<UniformRandomVariable> ();
        m_channelModel.SetEnvironment (RadioEnvironment::INDOOR_LOS);
    }

    void SetBaseStationPosition (Vector pos) { m_bsPosition = pos; }
    void SetClusterHeadProbability (double p) { m_chProb = p; }
    void SetInitialEnergy (double e) { m_initialEnergy = e; m_residualEnergy = e; }
    void SetDataPacketSize (uint32_t size) { m_dataPacketSize = size; }
    void SetNumRounds (uint32_t rounds) { m_totalRounds = rounds; }
    void SetEnableCas (bool e) { m_enableCas = e; }
    void SetEnableFairness (bool e) { m_enableFairness = e; }
    void SetEnableGateway (bool e) { m_enableGateway = e; }
    void SetRadioEnvironment (RadioEnvironment env) { m_channelModel.SetEnvironment (env); }
    double GetTotalEnergyConsumed () const { return m_totalEnergyConsumed; }
    bool IsAlive () const { return m_residualEnergy > 0; }

    static void ResetGlobalStats () { s_globalPacketsSent = 0; s_globalPacketsDelivered = 0; s_totalNodes = 0; s_globalTotalSnr = 0; s_globalSnrSamples = 0; s_currentRoundCHs = 0; s_currentRoundNumber = 0; }
    static uint32_t GetGlobalPacketsSent () { return s_globalPacketsSent; }
    static uint32_t GetGlobalPacketsDelivered () { return s_globalPacketsDelivered; }
    static double GetGlobalPdr () { return s_globalPacketsSent > 0 ? (double)s_globalPacketsDelivered / s_globalPacketsSent : 0.0; }
    static double GetGlobalAvgSnr () { return s_globalSnrSamples > 0 ? s_globalTotalSnr / s_globalSnrSamples : 0.0; }

protected:
    void DoDispose () override { Application::DoDispose (); }

private:
    void StartApplication () override {
        m_nodeId = GetNode()->GetId();
        m_residualEnergy = m_initialEnergy;
        Ptr<MobilityModel> mob = GetNode()->GetObject<MobilityModel>();
        m_position = mob ? mob->GetPosition() : Vector(0,0,0);
        s_totalNodes++;
        Simulator::Schedule (Seconds (m_random->GetValue (0, 0.1)), &AerisProtocolFull::StartRound, this);
    }
    void StopApplication () override {}

    void StartRound () {
        if (!IsAlive() || m_currentRound >= m_totalRounds) return;
        m_currentRound++;
        if (m_currentRound > s_currentRoundNumber) { s_currentRoundNumber = m_currentRound; s_currentRoundCHs = 0; }
        m_isClusterHead = ElectClusterHead();
        if (m_isClusterHead) { s_currentRoundCHs++; ConsumeEnergy(CalculateTxEnergy(50.0, 256)); }
        Simulator::Schedule (MilliSeconds(300), &AerisProtocolFull::DataTransmission, this);
        Simulator::Schedule (MilliSeconds(700), &AerisProtocolFull::Aggregation, this);
        Simulator::Schedule (Seconds(1.0), &AerisProtocolFull::StartRound, this);
    }

    bool ElectClusterHead () {
        if (!IsAlive()) return false;
        double casScore = m_enableCas ? CalculateCasScore() : m_random->GetValue(0.0, 1.0);
        return m_random->GetValue(0.0, 1.0) < m_chProb * (1.0 + casScore);
    }

    double CalculateCasScore () {
        double energyFactor = m_residualEnergy / m_initialEnergy;
        double distFactor = 1.0 - std::min(1.0, GetDistanceToBS() / 300.0);
        LinkQualityIndicator lqi = m_channelModel.CalculateLinkQuality(m_position, m_bsPosition, m_dataPacketSize);
        s_globalTotalSnr += lqi.snr; s_globalSnrSamples++;
        return 0.4 * energyFactor + 0.3 * distFactor + 0.3 * lqi.successProb;
    }

    void DataTransmission () {
        if (!IsAlive()) return;
        m_packetsSent++; s_globalPacketsSent++;
        ConsumeEnergy(CalculateTxEnergy(m_isClusterHead ? GetDistanceToBS() : 30.0, m_dataPacketSize * 8));
    }

    void Aggregation () {
        if (!IsAlive() || !m_isClusterHead) return;
        uint32_t numCHs = s_currentRoundCHs > 0 ? s_currentRoundCHs : 1;
        uint32_t clusterSize = std::max(1u, s_totalNodes / numCHs);
        if (clusterSize > 1) ConsumeEnergy(CalculateRxEnergy(m_dataPacketSize * 8 * (clusterSize - 1)));
        ConsumeEnergy(5e-9 * clusterSize * m_dataPacketSize * 8);

        double distToBS = GetDistanceToBS();
        uint32_t aggSize = m_dataPacketSize + clusterSize * 64;
        ConsumeEnergy(CalculateTxEnergy(distToBS, aggSize * 8));

        uint32_t delivered = 0;
        for (uint32_t i = 0; i < clusterSize; i++) {
            Vector memberPos = m_position;
            memberPos.x += m_random->GetValue(-30, 30);
            memberPos.y += m_random->GetValue(-30, 30);
            if (!m_channelModel.TransmitPacket(memberPos, m_position, m_dataPacketSize)) continue;

            bool success;
            if (m_enableGateway && distToBS > 100) {
                Vector gwPos((m_position.x + m_bsPosition.x)/2, (m_position.y + m_bsPosition.y)/2, 0);
                success = m_channelModel.TransmitPacket(m_position, gwPos, aggSize) && m_channelModel.TransmitPacket(gwPos, m_bsPosition, aggSize);
            } else {
                success = m_channelModel.TransmitPacket(m_position, m_bsPosition, aggSize);
            }
            if (success) delivered++;
        }
        s_globalPacketsDelivered += delivered;
    }

    void ConsumeEnergy (double e) { m_residualEnergy -= e; m_totalEnergyConsumed += e; if (m_residualEnergy < 0) m_residualEnergy = 0; }
    double CalculateTxEnergy (double d, uint32_t bits) { return d < 87.7 ? 50e-9*bits + 10e-12*bits*d*d : 50e-9*bits + 0.0013e-12*bits*std::pow(d,4); }
    double CalculateRxEnergy (uint32_t bits) { return 50e-9 * bits; }
    double GetDistanceToBS () { double dx = m_position.x - m_bsPosition.x, dy = m_position.y - m_bsPosition.y; return std::sqrt(dx*dx + dy*dy); }

    uint32_t m_nodeId;
    Vector m_position, m_bsPosition;
    bool m_isClusterHead;
    double m_chProb, m_initialEnergy, m_residualEnergy;
    uint32_t m_dataPacketSize, m_totalRounds, m_currentRound, m_packetsSent, m_packetsDelivered;
    double m_totalEnergyConsumed;
    bool m_enableCas, m_enableFairness, m_enableGateway;
    Ptr<UniformRandomVariable> m_random;
    RealisticChannelModel m_channelModel;

    static uint32_t s_globalPacketsSent, s_globalPacketsDelivered, s_totalNodes, s_globalSnrSamples;
    static uint32_t s_currentRoundCHs, s_currentRoundNumber;
    static double s_globalTotalSnr;
};

uint32_t AerisProtocolFull::s_globalPacketsSent = 0;
uint32_t AerisProtocolFull::s_globalPacketsDelivered = 0;
uint32_t AerisProtocolFull::s_totalNodes = 0;
uint32_t AerisProtocolFull::s_globalSnrSamples = 0;
uint32_t AerisProtocolFull::s_currentRoundCHs = 0;
uint32_t AerisProtocolFull::s_currentRoundNumber = 0;
double AerisProtocolFull::s_globalTotalSnr = 0;
NS_OBJECT_ENSURE_REGISTERED (AerisProtocolFull);

// ===================== LEACH Protocol =====================

class LeachProtocolNs3 : public Application
{
public:
    static TypeId GetTypeId (void) {
        static TypeId tid = TypeId ("ns3::LeachProtocolNs3").SetParent<Application> ().SetGroupName ("Aeris").AddConstructor<LeachProtocolNs3> ();
        return tid;
    }

    LeachProtocolNs3 () : m_nodeId(0), m_bsPosition(100,200,0), m_isClusterHead(false), m_chProb(0.1),
        m_initialEnergy(2.0), m_residualEnergy(2.0), m_dataPacketSize(512), m_totalRounds(300),
        m_currentRound(0), m_packetsSent(0), m_packetsDelivered(0), m_totalEnergyConsumed(0), m_roundsNotCH(0)
    {
        m_random = CreateObject<UniformRandomVariable> ();
        m_channelModel.SetEnvironment (RadioEnvironment::INDOOR_LOS);
    }

    void SetBaseStationPosition (Vector pos) { m_bsPosition = pos; }
    void SetClusterHeadProbability (double p) { m_chProb = p; }
    void SetInitialEnergy (double e) { m_initialEnergy = e; m_residualEnergy = e; }
    void SetDataPacketSize (uint32_t size) { m_dataPacketSize = size; }
    void SetNumRounds (uint32_t rounds) { m_totalRounds = rounds; }
    void SetRadioEnvironment (RadioEnvironment env) { m_channelModel.SetEnvironment (env); }
    double GetTotalEnergyConsumed () const { return m_totalEnergyConsumed; }
    bool IsAlive () const { return m_residualEnergy > 0; }

    static void ResetGlobalStats () { s_globalPacketsSent = 0; s_globalPacketsDelivered = 0; s_totalNodes = 0; s_currentRoundCHs = 0; s_currentRoundNumber = 0; s_globalTotalSnr = 0; s_globalSnrSamples = 0; }
    static uint32_t GetGlobalPacketsSent () { return s_globalPacketsSent; }
    static uint32_t GetGlobalPacketsDelivered () { return s_globalPacketsDelivered; }
    static double GetGlobalPdr () { return s_globalPacketsSent > 0 ? (double)s_globalPacketsDelivered / s_globalPacketsSent : 0.0; }
    static double GetGlobalAvgSnr () { return s_globalSnrSamples > 0 ? s_globalTotalSnr / s_globalSnrSamples : 0.0; }

protected:
    void DoDispose () override { Application::DoDispose (); }

private:
    void StartApplication () override {
        m_nodeId = GetNode()->GetId();
        m_residualEnergy = m_initialEnergy;
        Ptr<MobilityModel> mob = GetNode()->GetObject<MobilityModel>();
        m_position = mob ? mob->GetPosition() : Vector(0,0,0);
        s_totalNodes++;
        Simulator::Schedule (Seconds (m_random->GetValue (0, 0.1)), &LeachProtocolNs3::StartRound, this);
    }
    void StopApplication () override {}

    void StartRound () {
        if (!IsAlive() || m_currentRound >= m_totalRounds) return;
        m_currentRound++;
        if (m_currentRound > s_currentRoundNumber) { s_currentRoundNumber = m_currentRound; s_currentRoundCHs = 0; }
        m_isClusterHead = ElectClusterHead();
        if (m_isClusterHead) { s_currentRoundCHs++; ConsumeEnergy(CalculateTxEnergy(50.0, 256)); }
        Simulator::Schedule (MilliSeconds(300), &LeachProtocolNs3::DataTransmission, this);
        Simulator::Schedule (MilliSeconds(700), &LeachProtocolNs3::Aggregation, this);
        Simulator::Schedule (Seconds(1.0), &LeachProtocolNs3::StartRound, this);
    }

    bool ElectClusterHead () {
        if (!IsAlive()) return false;
        int roundMod = static_cast<int>(1.0 / m_chProb);
        int r = m_currentRound % roundMod;
        if (m_roundsNotCH < (uint32_t)roundMod) {
            if (r == 0) r = roundMod;
            double threshold = m_chProb / (1.0 - m_chProb * (r - 1));
            if (m_random->GetValue(0.0, 1.0) < threshold) { m_roundsNotCH = 0; return true; }
        }
        m_roundsNotCH++;
        return false;
    }

    void DataTransmission () {
        if (!IsAlive()) return;
        m_packetsSent++; s_globalPacketsSent++;
        ConsumeEnergy(CalculateTxEnergy(m_isClusterHead ? GetDistanceToBS() : 30.0, m_dataPacketSize * 8));
    }

    void Aggregation () {
        if (!IsAlive() || !m_isClusterHead) return;
        uint32_t numCHs = s_currentRoundCHs > 0 ? s_currentRoundCHs : 1;
        uint32_t clusterSize = std::max(1u, s_totalNodes / numCHs);
        if (clusterSize > 1) ConsumeEnergy(CalculateRxEnergy(m_dataPacketSize * 8 * (clusterSize - 1)));
        ConsumeEnergy(5e-9 * clusterSize * m_dataPacketSize * 8);

        double distToBS = GetDistanceToBS();
        uint32_t aggSize = m_dataPacketSize + clusterSize * 64;
        ConsumeEnergy(CalculateTxEnergy(distToBS, aggSize * 8));

        LinkQualityIndicator lqi = m_channelModel.CalculateLinkQuality(m_position, m_bsPosition, aggSize);
        s_globalTotalSnr += lqi.snr; s_globalSnrSamples++;

        uint32_t delivered = 0;
        for (uint32_t i = 0; i < clusterSize; i++) {
            Vector memberPos = m_position;
            memberPos.x += m_random->GetValue(-30, 30);
            memberPos.y += m_random->GetValue(-30, 30);
            if (!m_channelModel.TransmitPacket(memberPos, m_position, m_dataPacketSize)) continue;
            if (m_channelModel.TransmitPacket(m_position, m_bsPosition, aggSize)) delivered++;
        }
        s_globalPacketsDelivered += delivered;
    }

    void ConsumeEnergy (double e) { m_residualEnergy -= e; m_totalEnergyConsumed += e; if (m_residualEnergy < 0) m_residualEnergy = 0; }
    double CalculateTxEnergy (double d, uint32_t bits) { return d < 87.7 ? 50e-9*bits + 10e-12*bits*d*d : 50e-9*bits + 0.0013e-12*bits*std::pow(d,4); }
    double CalculateRxEnergy (uint32_t bits) { return 50e-9 * bits; }
    double GetDistanceToBS () { double dx = m_position.x - m_bsPosition.x, dy = m_position.y - m_bsPosition.y; return std::sqrt(dx*dx + dy*dy); }

    uint32_t m_nodeId;
    Vector m_position, m_bsPosition;
    bool m_isClusterHead;
    double m_chProb, m_initialEnergy, m_residualEnergy;
    uint32_t m_dataPacketSize, m_totalRounds, m_currentRound, m_packetsSent, m_packetsDelivered;
    double m_totalEnergyConsumed;
    uint32_t m_roundsNotCH;
    Ptr<UniformRandomVariable> m_random;
    RealisticChannelModel m_channelModel;

    static uint32_t s_globalPacketsSent, s_globalPacketsDelivered, s_totalNodes, s_currentRoundCHs, s_currentRoundNumber, s_globalSnrSamples;
    static double s_globalTotalSnr;
};

uint32_t LeachProtocolNs3::s_globalPacketsSent = 0;
uint32_t LeachProtocolNs3::s_globalPacketsDelivered = 0;
uint32_t LeachProtocolNs3::s_totalNodes = 0;
uint32_t LeachProtocolNs3::s_currentRoundCHs = 0;
uint32_t LeachProtocolNs3::s_currentRoundNumber = 0;
uint32_t LeachProtocolNs3::s_globalSnrSamples = 0;
double LeachProtocolNs3::s_globalTotalSnr = 0;
NS_OBJECT_ENSURE_REGISTERED (LeachProtocolNs3);

// ===================== HEED Protocol =====================
// Younis & Fahmy, IEEE TMC 2004
// Key difference from LEACH: energy-aware CH election with iterative probability doubling

class HeedProtocolNs3 : public Application
{
public:
    static TypeId GetTypeId (void) {
        static TypeId tid = TypeId ("ns3::HeedProtocolNs3").SetParent<Application> ().SetGroupName ("Aeris").AddConstructor<HeedProtocolNs3> ();
        return tid;
    }

    HeedProtocolNs3 () : m_nodeId(0), m_bsPosition(100,200,0), m_isClusterHead(false), m_chProb(0.05),
        m_initialEnergy(2.0), m_residualEnergy(2.0), m_dataPacketSize(512), m_totalRounds(300),
        m_currentRound(0), m_packetsSent(0), m_packetsDelivered(0), m_totalEnergyConsumed(0)
    {
        m_random = CreateObject<UniformRandomVariable> ();
        m_channelModel.SetEnvironment (RadioEnvironment::INDOOR_LOS);
    }

    void SetBaseStationPosition (Vector pos) { m_bsPosition = pos; }
    void SetClusterHeadProbability (double p) { m_chProb = p; }
    void SetInitialEnergy (double e) { m_initialEnergy = e; m_residualEnergy = e; }
    void SetDataPacketSize (uint32_t size) { m_dataPacketSize = size; }
    void SetNumRounds (uint32_t rounds) { m_totalRounds = rounds; }
    void SetRadioEnvironment (RadioEnvironment env) { m_channelModel.SetEnvironment (env); }
    double GetTotalEnergyConsumed () const { return m_totalEnergyConsumed; }
    bool IsAlive () const { return m_residualEnergy > 0; }

    static void ResetGlobalStats () { s_globalPacketsSent = 0; s_globalPacketsDelivered = 0; s_totalNodes = 0; s_currentRoundCHs = 0; s_currentRoundNumber = 0; s_globalTotalSnr = 0; s_globalSnrSamples = 0; }
    static uint32_t GetGlobalPacketsSent () { return s_globalPacketsSent; }
    static uint32_t GetGlobalPacketsDelivered () { return s_globalPacketsDelivered; }
    static double GetGlobalPdr () { return s_globalPacketsSent > 0 ? (double)s_globalPacketsDelivered / s_globalPacketsSent : 0.0; }
    static double GetGlobalAvgSnr () { return s_globalSnrSamples > 0 ? s_globalTotalSnr / s_globalSnrSamples : 0.0; }

protected:
    void DoDispose () override { Application::DoDispose (); }

private:
    void StartApplication () override {
        m_nodeId = GetNode()->GetId();
        m_residualEnergy = m_initialEnergy;
        Ptr<MobilityModel> mob = GetNode()->GetObject<MobilityModel>();
        m_position = mob ? mob->GetPosition() : Vector(0,0,0);
        s_totalNodes++;
        Simulator::Schedule (Seconds (m_random->GetValue (0, 0.1)), &HeedProtocolNs3::StartRound, this);
    }
    void StopApplication () override {}

    void StartRound () {
        if (!IsAlive() || m_currentRound >= m_totalRounds) return;
        m_currentRound++;
        if (m_currentRound > s_currentRoundNumber) { s_currentRoundNumber = m_currentRound; s_currentRoundCHs = 0; }
        m_isClusterHead = ElectClusterHead();
        if (m_isClusterHead) { s_currentRoundCHs++; ConsumeEnergy(CalculateTxEnergy(50.0, 256)); }
        Simulator::Schedule (MilliSeconds(300), &HeedProtocolNs3::DataTransmission, this);
        Simulator::Schedule (MilliSeconds(700), &HeedProtocolNs3::Aggregation, this);
        Simulator::Schedule (Seconds(1.0), &HeedProtocolNs3::StartRound, this);
    }

    bool ElectClusterHead () {
        if (!IsAlive()) return false;
        // HEED: energy-weighted probability with iterative doubling
        double energyRatio = m_residualEnergy / m_initialEnergy;
        double prob = m_chProb * energyRatio;
        // Iterative doubling (simplified: 3 iterations)
        for (int iter = 0; iter < 3; iter++) {
            if (m_random->GetValue(0.0, 1.0) < prob) return true;
            prob = std::min(1.0, prob * 2.0);
        }
        return false;
    }

    void DataTransmission () {
        if (!IsAlive()) return;
        m_packetsSent++; s_globalPacketsSent++;
        ConsumeEnergy(CalculateTxEnergy(m_isClusterHead ? GetDistanceToBS() : 30.0, m_dataPacketSize * 8));
    }

    void Aggregation () {
        if (!IsAlive() || !m_isClusterHead) return;
        uint32_t numCHs = s_currentRoundCHs > 0 ? s_currentRoundCHs : 1;
        uint32_t clusterSize = std::max(1u, s_totalNodes / numCHs);
        if (clusterSize > 1) ConsumeEnergy(CalculateRxEnergy(m_dataPacketSize * 8 * (clusterSize - 1)));
        ConsumeEnergy(5e-9 * clusterSize * m_dataPacketSize * 8);

        double distToBS = GetDistanceToBS();
        uint32_t aggSize = m_dataPacketSize + clusterSize * 64;
        ConsumeEnergy(CalculateTxEnergy(distToBS, aggSize * 8));

        LinkQualityIndicator lqi = m_channelModel.CalculateLinkQuality(m_position, m_bsPosition, aggSize);
        s_globalTotalSnr += lqi.snr; s_globalSnrSamples++;

        uint32_t delivered = 0;
        for (uint32_t i = 0; i < clusterSize; i++) {
            Vector memberPos = m_position;
            memberPos.x += m_random->GetValue(-30, 30);
            memberPos.y += m_random->GetValue(-30, 30);
            if (!m_channelModel.TransmitPacket(memberPos, m_position, m_dataPacketSize)) continue;
            if (m_channelModel.TransmitPacket(m_position, m_bsPosition, aggSize)) delivered++;
        }
        s_globalPacketsDelivered += delivered;
    }

    void ConsumeEnergy (double e) { m_residualEnergy -= e; m_totalEnergyConsumed += e; if (m_residualEnergy < 0) m_residualEnergy = 0; }
    double CalculateTxEnergy (double d, uint32_t bits) { return d < 87.7 ? 50e-9*bits + 10e-12*bits*d*d : 50e-9*bits + 0.0013e-12*bits*std::pow(d,4); }
    double CalculateRxEnergy (uint32_t bits) { return 50e-9 * bits; }
    double GetDistanceToBS () { double dx = m_position.x - m_bsPosition.x, dy = m_position.y - m_bsPosition.y; return std::sqrt(dx*dx + dy*dy); }

    uint32_t m_nodeId;
    Vector m_position, m_bsPosition;
    bool m_isClusterHead;
    double m_chProb, m_initialEnergy, m_residualEnergy;
    uint32_t m_dataPacketSize, m_totalRounds, m_currentRound, m_packetsSent, m_packetsDelivered;
    double m_totalEnergyConsumed;
    Ptr<UniformRandomVariable> m_random;
    RealisticChannelModel m_channelModel;

    static uint32_t s_globalPacketsSent, s_globalPacketsDelivered, s_totalNodes, s_currentRoundCHs, s_currentRoundNumber, s_globalSnrSamples;
    static double s_globalTotalSnr;
};

uint32_t HeedProtocolNs3::s_globalPacketsSent = 0;
uint32_t HeedProtocolNs3::s_globalPacketsDelivered = 0;
uint32_t HeedProtocolNs3::s_totalNodes = 0;
uint32_t HeedProtocolNs3::s_currentRoundCHs = 0;
uint32_t HeedProtocolNs3::s_currentRoundNumber = 0;
uint32_t HeedProtocolNs3::s_globalSnrSamples = 0;
double HeedProtocolNs3::s_globalTotalSnr = 0;
NS_OBJECT_ENSURE_REGISTERED (HeedProtocolNs3);

// ===================== PEGASIS Protocol =====================
// Lindsey & Raghavendra, IEEE Aerospace 2002
// Chain-based: greedy chain construction, round-robin leader, sequential forwarding

class PegasisProtocolNs3 : public Application
{
public:
    static TypeId GetTypeId (void) {
        static TypeId tid = TypeId ("ns3::PegasisProtocolNs3").SetParent<Application> ().SetGroupName ("Aeris").AddConstructor<PegasisProtocolNs3> ();
        return tid;
    }

    PegasisProtocolNs3 () : m_nodeId(0), m_bsPosition(100,200,0), m_isLeader(false),
        m_initialEnergy(2.0), m_residualEnergy(2.0), m_dataPacketSize(512), m_totalRounds(300),
        m_currentRound(0), m_packetsSent(0), m_packetsDelivered(0), m_totalEnergyConsumed(0)
    {
        m_random = CreateObject<UniformRandomVariable> ();
        m_channelModel.SetEnvironment (RadioEnvironment::INDOOR_LOS);
    }

    void SetBaseStationPosition (Vector pos) { m_bsPosition = pos; }
    void SetInitialEnergy (double e) { m_initialEnergy = e; m_residualEnergy = e; }
    void SetDataPacketSize (uint32_t size) { m_dataPacketSize = size; }
    void SetNumRounds (uint32_t rounds) { m_totalRounds = rounds; }
    void SetRadioEnvironment (RadioEnvironment env) { m_channelModel.SetEnvironment (env); }
    double GetTotalEnergyConsumed () const { return m_totalEnergyConsumed; }
    bool IsAlive () const { return m_residualEnergy > 0; }

    static void ResetGlobalStats () {
        s_globalPacketsSent = 0;
        s_globalPacketsDelivered = 0;
        s_totalNodes = 0;
        s_globalTotalSnr = 0;
        s_globalSnrSamples = 0;
        s_nodePositions.clear ();
        s_chainOrder.clear ();
        s_chainBuilt = false;
    }
    static uint32_t GetGlobalPacketsSent () { return s_globalPacketsSent; }
    static uint32_t GetGlobalPacketsDelivered () { return s_globalPacketsDelivered; }
    static double GetGlobalPdr () { return s_globalPacketsSent > 0 ? (double)s_globalPacketsDelivered / s_globalPacketsSent : 0.0; }
    static double GetGlobalAvgSnr () { return s_globalSnrSamples > 0 ? s_globalTotalSnr / s_globalSnrSamples : 0.0; }

protected:
    void DoDispose () override { Application::DoDispose (); }

private:
    static double Distance (const Vector& a, const Vector& b) {
        double dx = a.x - b.x;
        double dy = a.y - b.y;
        return std::sqrt (dx * dx + dy * dy);
    }

    static void EnsureChainOrder (const Vector& bsPos) {
        if (s_chainBuilt && s_chainOrder.size () == s_nodePositions.size ()) return;
        if (s_nodePositions.empty ()) return;

        std::vector<uint32_t> remaining;
        remaining.reserve (s_nodePositions.size ());
        for (uint32_t i = 0; i < s_nodePositions.size (); ++i) {
            remaining.push_back (i);
        }

        s_chainOrder.clear ();

        uint32_t startId = remaining.front ();
        double maxDist = -1.0;
        for (uint32_t id : remaining) {
            double dist = Distance (s_nodePositions[id], bsPos);
            if (dist > maxDist) {
                maxDist = dist;
                startId = id;
            }
        }
        s_chainOrder.push_back (startId);
        remaining.erase (std::remove (remaining.begin (), remaining.end (), startId), remaining.end ());

        while (!remaining.empty ()) {
            uint32_t lastId = s_chainOrder.back ();
            uint32_t bestId = remaining.front ();
            double bestDist = std::numeric_limits<double>::infinity ();
            for (uint32_t candId : remaining) {
                double dist = Distance (s_nodePositions[lastId], s_nodePositions[candId]);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestId = candId;
                }
            }
            s_chainOrder.push_back (bestId);
            remaining.erase (std::remove (remaining.begin (), remaining.end (), bestId), remaining.end ());
        }

        s_chainBuilt = true;
    }

    static int32_t ChainIndexForNode (uint32_t nodeId) {
        for (size_t i = 0; i < s_chainOrder.size (); ++i) {
            if (s_chainOrder[i] == nodeId) return static_cast<int32_t> (i);
        }
        return -1;
    }

    uint32_t AggregatedPacketSizeBytes (uint32_t mergedPackets) const {
        uint32_t safeMerged = std::max (1u, mergedPackets);
        return m_dataPacketSize + safeMerged * 64;
    }

    void StartApplication () override {
        m_nodeId = GetNode()->GetId();
        m_residualEnergy = m_initialEnergy;
        Ptr<MobilityModel> mob = GetNode()->GetObject<MobilityModel>();
        m_position = mob ? mob->GetPosition() : Vector(0,0,0);
        if (s_nodePositions.size () <= m_nodeId) {
            s_nodePositions.resize (m_nodeId + 1, Vector (0, 0, 0));
        }
        s_nodePositions[m_nodeId] = m_position;
        s_totalNodes++;
        Simulator::Schedule (Seconds (m_random->GetValue (0, 0.1)), &PegasisProtocolNs3::StartRound, this);
    }
    void StopApplication () override {}

    void StartRound () {
        if (!IsAlive() || m_currentRound >= m_totalRounds) return;
        m_currentRound++;
        EnsureChainOrder (m_bsPosition);
        uint32_t leaderId = 0;
        if (!s_chainOrder.empty ()) {
            leaderId = s_chainOrder[m_currentRound % s_chainOrder.size ()];
        } else if (s_totalNodes > 0) {
            leaderId = (m_currentRound % s_totalNodes);
        }
        m_isLeader = (m_nodeId == leaderId);
        Simulator::Schedule (MilliSeconds(300), &PegasisProtocolNs3::ChainForward, this);
        Simulator::Schedule (Seconds(1.0), &PegasisProtocolNs3::StartRound, this);
    }

    void ChainForward () {
        if (!IsAlive()) return;
        m_packetsSent++; s_globalPacketsSent++;

        // Greedy nearest-neighbor chain: hop distance scales with node density
        // Beardwood-Halton-Hammersley: avg hop ≈ 0.7 * sqrt(Area / N)
        EnsureChainOrder (m_bsPosition);
        if (s_chainOrder.empty ()) return;

        uint32_t leaderId = s_chainOrder[m_currentRound % s_chainOrder.size ()];
        int32_t leaderIdx = ChainIndexForNode (leaderId);
        int32_t currentIdx = ChainIndexForNode (m_nodeId);
        if (leaderIdx < 0 || currentIdx < 0) return;

        // Multi-hop chain delivery with cumulative packet loss
        // Average hops to leader ≈ N/2 for uniform chain position
        bool chainOk = true;
        Vector hopPos = m_position;
        uint32_t mergedPackets = 1;
        if (!m_isLeader) {
            int32_t step = (currentIdx < leaderIdx) ? 1 : -1;
            for (int32_t idx = currentIdx; idx != leaderIdx && chainOk; idx += step) {
                int32_t targetIdx = idx + step;
                if (targetIdx < 0 || targetIdx >= static_cast<int32_t> (s_chainOrder.size ())) {
                    chainOk = false;
                    break;
                }
                Vector hopDst = s_nodePositions[s_chainOrder[targetIdx]];
                uint32_t hopPacketSize = AggregatedPacketSizeBytes (mergedPackets);
                double hopDist = Distance (hopPos, hopDst);
                ConsumeEnergy (CalculateTxEnergy (hopDist, hopPacketSize * 8));
                chainOk = m_channelModel.TransmitPacket (hopPos, hopDst, hopPacketSize);
                hopPos = hopDst;
                mergedPackets++;
            }
        }

        if (!chainOk) return;  // packet lost along chain

        Vector leaderPos = (leaderId < s_nodePositions.size ()) ? s_nodePositions[leaderId] : m_position;
        uint32_t aggSize = AggregatedPacketSizeBytes (std::max (1u, s_totalNodes));
        double distLeaderBS = std::sqrt(std::pow(leaderPos.x - m_bsPosition.x, 2) +
                                        std::pow(leaderPos.y - m_bsPosition.y, 2));

        if (m_isLeader) {
            // Leader uplink should scale with the full chain aggregate, not a fixed 2x payload.
            ConsumeEnergy(CalculateTxEnergy(distLeaderBS, aggSize * 8));

            LinkQualityIndicator lqi = m_channelModel.CalculateLinkQuality(leaderPos, m_bsPosition, aggSize);
            s_globalTotalSnr += lqi.snr; s_globalSnrSamples++;

            if (m_channelModel.TransmitPacket(leaderPos, m_bsPosition, aggSize)) {
                s_globalPacketsDelivered++;
            }
        } else {
            // Non-leader packets share the same topology-aware leader uplink model.
            LinkQualityIndicator lqi = m_channelModel.CalculateLinkQuality(leaderPos, m_bsPosition, aggSize);
            s_globalTotalSnr += lqi.snr; s_globalSnrSamples++;

            if (m_channelModel.TransmitPacket(leaderPos, m_bsPosition, aggSize)) {
                s_globalPacketsDelivered++;
            }
        }
    }

    void ConsumeEnergy (double e) { m_residualEnergy -= e; m_totalEnergyConsumed += e; if (m_residualEnergy < 0) m_residualEnergy = 0; }
    double CalculateTxEnergy (double d, uint32_t bits) { return d < 87.7 ? 50e-9*bits + 10e-12*bits*d*d : 50e-9*bits + 0.0013e-12*bits*std::pow(d,4); }
    double CalculateRxEnergy (uint32_t bits) { return 50e-9 * bits; }
    double GetDistanceToBS () { double dx = m_position.x - m_bsPosition.x, dy = m_position.y - m_bsPosition.y; return std::sqrt(dx*dx + dy*dy); }

    uint32_t m_nodeId;
    Vector m_position, m_bsPosition;
    bool m_isLeader;
    double m_initialEnergy, m_residualEnergy;
    uint32_t m_dataPacketSize, m_totalRounds, m_currentRound, m_packetsSent, m_packetsDelivered;
    double m_totalEnergyConsumed;
    Ptr<UniformRandomVariable> m_random;
    RealisticChannelModel m_channelModel;

    static uint32_t s_globalPacketsSent, s_globalPacketsDelivered, s_totalNodes, s_globalSnrSamples;
    static double s_globalTotalSnr;
    static std::vector<Vector> s_nodePositions;
    static std::vector<uint32_t> s_chainOrder;
    static bool s_chainBuilt;
};

uint32_t PegasisProtocolNs3::s_globalPacketsSent = 0;
uint32_t PegasisProtocolNs3::s_globalPacketsDelivered = 0;
uint32_t PegasisProtocolNs3::s_totalNodes = 0;
uint32_t PegasisProtocolNs3::s_globalSnrSamples = 0;
double PegasisProtocolNs3::s_globalTotalSnr = 0;
std::vector<Vector> PegasisProtocolNs3::s_nodePositions;
std::vector<uint32_t> PegasisProtocolNs3::s_chainOrder;
bool PegasisProtocolNs3::s_chainBuilt = false;
NS_OBJECT_ENSURE_REGISTERED (PegasisProtocolNs3);

// ===================== TEEN Protocol =====================
// Manjeshwar & Agrawal, IPDPS 2001
// Threshold-based: hard/soft thresholds control when nodes transmit

class TeenProtocolNs3 : public Application
{
public:
    static TypeId GetTypeId (void) {
        static TypeId tid = TypeId ("ns3::TeenProtocolNs3").SetParent<Application> ().SetGroupName ("Aeris").AddConstructor<TeenProtocolNs3> ();
        return tid;
    }

    TeenProtocolNs3 () : m_nodeId(0), m_bsPosition(100,200,0), m_isClusterHead(false), m_chProb(0.08),
        m_initialEnergy(2.0), m_residualEnergy(2.0), m_dataPacketSize(512), m_totalRounds(300),
        m_currentRound(0), m_packetsSent(0), m_packetsDelivered(0), m_totalEnergyConsumed(0),
        m_hardThreshold(45.0), m_softThreshold(0.5), m_lastSensedValue(0.0), m_lastTransmittedValue(-999.0)
    {
        m_random = CreateObject<UniformRandomVariable> ();
        m_channelModel.SetEnvironment (RadioEnvironment::INDOOR_LOS);
    }

    void SetBaseStationPosition (Vector pos) { m_bsPosition = pos; }
    void SetClusterHeadProbability (double p) { m_chProb = p; }
    void SetInitialEnergy (double e) { m_initialEnergy = e; m_residualEnergy = e; }
    void SetDataPacketSize (uint32_t size) { m_dataPacketSize = size; }
    void SetNumRounds (uint32_t rounds) { m_totalRounds = rounds; }
    void SetRadioEnvironment (RadioEnvironment env) { m_channelModel.SetEnvironment (env); }
    double GetTotalEnergyConsumed () const { return m_totalEnergyConsumed; }
    bool IsAlive () const { return m_residualEnergy > 0; }

    static void ResetGlobalStats () { s_globalPacketsSent = 0; s_globalPacketsDelivered = 0; s_totalNodes = 0; s_currentRoundCHs = 0; s_currentRoundNumber = 0; s_globalTotalSnr = 0; s_globalSnrSamples = 0; }
    static uint32_t GetGlobalPacketsSent () { return s_globalPacketsSent; }
    static uint32_t GetGlobalPacketsDelivered () { return s_globalPacketsDelivered; }
    static double GetGlobalPdr () { return s_globalPacketsSent > 0 ? (double)s_globalPacketsDelivered / s_globalPacketsSent : 0.0; }
    static double GetGlobalAvgSnr () { return s_globalSnrSamples > 0 ? s_globalTotalSnr / s_globalSnrSamples : 0.0; }

protected:
    void DoDispose () override { Application::DoDispose (); }

private:
    void StartApplication () override {
        m_nodeId = GetNode()->GetId();
        m_residualEnergy = m_initialEnergy;
        Ptr<MobilityModel> mob = GetNode()->GetObject<MobilityModel>();
        m_position = mob ? mob->GetPosition() : Vector(0,0,0);
        s_totalNodes++;
        // Initialize sensed value based on position
        m_lastSensedValue = 50.0 + (m_position.x + m_position.y) / 20.0;
        Simulator::Schedule (Seconds (m_random->GetValue (0, 0.1)), &TeenProtocolNs3::StartRound, this);
    }
    void StopApplication () override {}

    void StartRound () {
        if (!IsAlive() || m_currentRound >= m_totalRounds) return;
        m_currentRound++;
        if (m_currentRound > s_currentRoundNumber) { s_currentRoundNumber = m_currentRound; s_currentRoundCHs = 0; }
        m_isClusterHead = ElectClusterHead();
        if (m_isClusterHead) { s_currentRoundCHs++; ConsumeEnergy(CalculateTxEnergy(50.0, 256)); }
        Simulator::Schedule (MilliSeconds(300), &TeenProtocolNs3::SenseAndTransmit, this);
        Simulator::Schedule (MilliSeconds(700), &TeenProtocolNs3::Aggregation, this);
        Simulator::Schedule (Seconds(1.0), &TeenProtocolNs3::StartRound, this);
    }

    bool ElectClusterHead () {
        if (!IsAlive()) return false;
        double energyRatio = m_residualEnergy / m_initialEnergy;
        return m_random->GetValue(0.0, 1.0) < m_chProb * energyRatio;
    }

    bool ShouldTransmit () {
        // Sense environment (synthetic: base + location + noise)
        m_lastSensedValue = 50.0 + (m_position.x + m_position.y) / 20.0 + m_random->GetValue(-3.0, 3.0);
        // Hard threshold check
        if (m_lastSensedValue < m_hardThreshold) return false;
        // First transmission
        if (m_lastTransmittedValue < -900.0) { m_lastTransmittedValue = m_lastSensedValue; return true; }
        // Soft threshold check
        if (std::abs(m_lastSensedValue - m_lastTransmittedValue) >= m_softThreshold) {
            m_lastTransmittedValue = m_lastSensedValue;
            return true;
        }
        // Time-based force (every 3 rounds)
        if (m_currentRound % 3 == 0) { m_lastTransmittedValue = m_lastSensedValue; return true; }
        return false;
    }

    void SenseAndTransmit () {
        if (!IsAlive()) return;
        if (!ShouldTransmit()) return;
        m_packetsSent++; s_globalPacketsSent++;
        ConsumeEnergy(CalculateTxEnergy(m_isClusterHead ? GetDistanceToBS() : 30.0, m_dataPacketSize * 8));
    }

    void Aggregation () {
        if (!IsAlive() || !m_isClusterHead) return;
        uint32_t numCHs = s_currentRoundCHs > 0 ? s_currentRoundCHs : 1;
        uint32_t clusterSize = std::max(1u, s_totalNodes / numCHs);
        if (clusterSize > 1) ConsumeEnergy(CalculateRxEnergy(m_dataPacketSize * 8 * (clusterSize - 1)));
        ConsumeEnergy(5e-9 * clusterSize * m_dataPacketSize * 8);

        double distToBS = GetDistanceToBS();
        uint32_t aggSize = m_dataPacketSize + clusterSize * 64;
        ConsumeEnergy(CalculateTxEnergy(distToBS, aggSize * 8));

        LinkQualityIndicator lqi = m_channelModel.CalculateLinkQuality(m_position, m_bsPosition, aggSize);
        s_globalTotalSnr += lqi.snr; s_globalSnrSamples++;

        // TEEN: not all members transmit every round (threshold-controlled)
        // Estimate ~60% of members transmit per round on average
        uint32_t activeMembers = std::max(1u, (uint32_t)(clusterSize * 0.6));
        uint32_t delivered = 0;
        for (uint32_t i = 0; i < activeMembers; i++) {
            Vector memberPos = m_position;
            memberPos.x += m_random->GetValue(-30, 30);
            memberPos.y += m_random->GetValue(-30, 30);
            if (!m_channelModel.TransmitPacket(memberPos, m_position, m_dataPacketSize)) continue;
            if (m_channelModel.TransmitPacket(m_position, m_bsPosition, aggSize)) delivered++;
        }
        s_globalPacketsDelivered += delivered;
    }

    void ConsumeEnergy (double e) { m_residualEnergy -= e; m_totalEnergyConsumed += e; if (m_residualEnergy < 0) m_residualEnergy = 0; }
    double CalculateTxEnergy (double d, uint32_t bits) { return d < 87.7 ? 50e-9*bits + 10e-12*bits*d*d : 50e-9*bits + 0.0013e-12*bits*std::pow(d,4); }
    double CalculateRxEnergy (uint32_t bits) { return 50e-9 * bits; }
    double GetDistanceToBS () { double dx = m_position.x - m_bsPosition.x, dy = m_position.y - m_bsPosition.y; return std::sqrt(dx*dx + dy*dy); }

    uint32_t m_nodeId;
    Vector m_position, m_bsPosition;
    bool m_isClusterHead;
    double m_chProb, m_initialEnergy, m_residualEnergy;
    uint32_t m_dataPacketSize, m_totalRounds, m_currentRound, m_packetsSent, m_packetsDelivered;
    double m_totalEnergyConsumed;
    double m_hardThreshold, m_softThreshold, m_lastSensedValue, m_lastTransmittedValue;
    Ptr<UniformRandomVariable> m_random;
    RealisticChannelModel m_channelModel;

    static uint32_t s_globalPacketsSent, s_globalPacketsDelivered, s_totalNodes, s_currentRoundCHs, s_currentRoundNumber, s_globalSnrSamples;
    static double s_globalTotalSnr;
};

uint32_t TeenProtocolNs3::s_globalPacketsSent = 0;
uint32_t TeenProtocolNs3::s_globalPacketsDelivered = 0;
uint32_t TeenProtocolNs3::s_totalNodes = 0;
uint32_t TeenProtocolNs3::s_currentRoundCHs = 0;
uint32_t TeenProtocolNs3::s_currentRoundNumber = 0;
uint32_t TeenProtocolNs3::s_globalSnrSamples = 0;
double TeenProtocolNs3::s_globalTotalSnr = 0;
NS_OBJECT_ENSURE_REGISTERED (TeenProtocolNs3);

// ===================== Collection-Tree Baselines =====================

class CollectionBaselineProtocolNs3 : public Application
{
public:
    enum class Mode { RPL_MRHOF, CTP };

    static TypeId GetTypeId (void) {
        static TypeId tid = TypeId ("ns3::CollectionBaselineProtocolNs3").SetParent<Application> ().SetGroupName ("Aeris").AddConstructor<CollectionBaselineProtocolNs3> ();
        return tid;
    }

    CollectionBaselineProtocolNs3 () : m_nodeIndex(0), m_protocolKey("RPL-MRHOF"), m_mode(Mode::RPL_MRHOF),
        m_bsPosition(100,200,0), m_initialEnergy(2.0), m_residualEnergy(2.0), m_dataPacketSize(512),
        m_totalRounds(300), m_currentRound(0), m_totalEnergyConsumed(0)
    {
        m_random = CreateObject<UniformRandomVariable> ();
        m_channelModel.SetEnvironment (RadioEnvironment::INDOOR_LOS);
    }

    void SetNodeIndex (uint32_t i) { m_nodeIndex = i; }
    void SetProtocolKey (const std::string& key) {
        m_protocolKey = key;
        m_mode = (key == "CTP") ? Mode::CTP : Mode::RPL_MRHOF;
    }
    void SetBaseStationPosition (Vector pos) { m_bsPosition = pos; }
    void SetInitialEnergy (double e) { m_initialEnergy = e; m_residualEnergy = e; }
    void SetDataPacketSize (uint32_t size) { m_dataPacketSize = size; }
    void SetNumRounds (uint32_t rounds) { m_totalRounds = rounds; }
    void SetRadioEnvironment (RadioEnvironment env) { m_channelModel.SetEnvironment (env); m_radioEnv = env; }
    double GetTotalEnergyConsumed () const { return m_totalEnergyConsumed; }
    bool IsAlive () const { return m_residualEnergy > 0; }

    static void ResetGlobalStats (const std::string& key) { s_stats[key] = Stats(); }
    static uint32_t GetGlobalPacketsSent (const std::string& key) { return s_stats[key].packetsSent; }
    static uint32_t GetGlobalPacketsDelivered (const std::string& key) { return s_stats[key].packetsDelivered; }
    static double GetGlobalPdr (const std::string& key) {
        const auto& st = s_stats[key];
        return st.packetsSent > 0 ? (double) st.packetsDelivered / st.packetsSent : 0.0;
    }
    static double GetGlobalAvgSnr (const std::string& key) {
        const auto& st = s_stats[key];
        return st.snrSamples > 0 ? st.totalSnr / st.snrSamples : 0.0;
    }

protected:
    void DoDispose () override { Application::DoDispose (); }

public:
    struct Stats {
        uint32_t packetsSent = 0;
        uint32_t packetsDelivered = 0;
        uint32_t totalNodes = 0;
        uint32_t snrSamples = 0;
        double totalSnr = 0.0;
        bool routesBuilt = false;
        std::vector<Vector> positions;
        std::vector<int32_t> parents;
        std::vector<double> ranks;
    };

private:
    void StartApplication () override {
        Ptr<MobilityModel> mob = GetNode()->GetObject<MobilityModel>();
        m_position = mob ? mob->GetPosition() : Vector(0,0,0);
        auto& st = s_stats[m_protocolKey];
        if (st.positions.size() <= m_nodeIndex) st.positions.resize(m_nodeIndex + 1);
        st.positions[m_nodeIndex] = m_position;
        st.totalNodes = std::max(st.totalNodes, m_nodeIndex + 1);
        Simulator::Schedule (Seconds (m_random->GetValue (0, 0.1)), &CollectionBaselineProtocolNs3::StartRound, this);
    }
    void StopApplication () override {}

    void StartRound () {
        if (!IsAlive() || m_currentRound >= m_totalRounds) return;
        m_currentRound++;
        Simulator::Schedule (MilliSeconds(300), &CollectionBaselineProtocolNs3::TransmitToRoot, this);
        Simulator::Schedule (Seconds(1.0), &CollectionBaselineProtocolNs3::StartRound, this);
    }

    void TransmitToRoot () {
        if (!IsAlive()) return;
        auto& st = s_stats[m_protocolKey];
        EnsureRoutes(st);
        st.packetsSent++;

        bool delivered = true;
        uint32_t current = m_nodeIndex;
        uint32_t hops = 0;
        while (hops++ < 32) {
            int32_t parent = (current < st.parents.size()) ? st.parents[current] : -1;
            Vector txPos = st.positions[current];
            Vector rxPos = parent < 0 ? m_bsPosition : st.positions[(uint32_t) parent];
            double distance = Distance(txPos, rxPos);
            LinkQualityIndicator lqi = m_channelModel.CalculateLinkQuality(txPos, rxPos, m_dataPacketSize);
            st.totalSnr += lqi.snr;
            st.snrSamples++;
            ConsumeEnergy(CalculateTxEnergy(distance, m_dataPacketSize * 8));
            if (parent >= 0) ConsumeEnergy(CalculateRxEnergy(m_dataPacketSize * 8));
            if (!m_channelModel.TransmitPacket(txPos, rxPos, m_dataPacketSize)) {
                delivered = false;
                break;
            }
            if (parent < 0) break;
            current = (uint32_t) parent;
        }

        if (hops >= 32) delivered = false;
        if (delivered) st.packetsDelivered++;
    }

    void EnsureRoutes (Stats& st) {
        if (st.routesBuilt) return;
        uint32_t n = st.totalNodes;
        st.parents.assign(n, -1);
        st.ranks.assign(n, 1e9);

        std::vector<uint32_t> order(n);
        for (uint32_t i = 0; i < n; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            return Distance(st.positions[a], m_bsPosition) < Distance(st.positions[b], m_bsPosition);
        });

        for (uint32_t node : order) {
            double distToBs = Distance(st.positions[node], m_bsPosition);
            LinkQualityIndicator direct = m_channelModel.CalculateLinkQuality(st.positions[node], m_bsPosition, m_dataPacketSize);
            double bestCost = LinkCost(direct.successProb, distToBs);
            int32_t bestParent = -1;

            for (uint32_t cand : order) {
                if (cand == node) continue;
                double candDist = Distance(st.positions[cand], m_bsPosition);
                if (candDist + 1e-6 >= distToBs) continue;
                if (m_mode == Mode::RPL_MRHOF && st.ranks[cand] >= 1e8) continue;

                double hopDistance = Distance(st.positions[node], st.positions[cand]);
                if (hopDistance > 140.0 && m_mode == Mode::CTP) continue;
                LinkQualityIndicator lqi = m_channelModel.CalculateLinkQuality(st.positions[node], st.positions[cand], m_dataPacketSize);
                if (lqi.successProb < 0.02) continue;

                double cost;
                if (m_mode == Mode::RPL_MRHOF) {
                    cost = st.ranks[cand] + LinkCost(lqi.successProb, hopDistance);
                } else {
                    double progress = std::max(1.0, distToBs - candDist);
                    cost = LinkCost(lqi.successProb, hopDistance) + 80.0 / progress;
                }
                if (cost < bestCost) {
                    bestCost = cost;
                    bestParent = (int32_t) cand;
                }
            }

            st.parents[node] = bestParent;
            st.ranks[node] = bestCost;
        }
        st.routesBuilt = true;
    }

    double LinkCost (double successProb, double distance) const {
        double etx = 1.0 / std::max(0.01, successProb);
        double distancePenalty = m_mode == Mode::RPL_MRHOF ? distance / 220.0 : distance / 320.0;
        return etx + distancePenalty;
    }

    void ConsumeEnergy (double e) { m_residualEnergy -= e; m_totalEnergyConsumed += e; if (m_residualEnergy < 0) m_residualEnergy = 0; }
    double CalculateTxEnergy (double d, uint32_t bits) { return d < 87.7 ? 50e-9*bits + 10e-12*bits*d*d : 50e-9*bits + 0.0013e-12*bits*std::pow(d,4); }
    double CalculateRxEnergy (uint32_t bits) { return 50e-9 * bits; }
    double Distance (Vector a, Vector b) const { double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z; return std::sqrt(dx*dx + dy*dy + dz*dz); }

    uint32_t m_nodeIndex;
    std::string m_protocolKey;
    Mode m_mode;
    Vector m_position, m_bsPosition;
    RadioEnvironment m_radioEnv = RadioEnvironment::INDOOR_LOS;
    double m_initialEnergy, m_residualEnergy;
    uint32_t m_dataPacketSize, m_totalRounds, m_currentRound;
    double m_totalEnergyConsumed;
    Ptr<UniformRandomVariable> m_random;
    RealisticChannelModel m_channelModel;

    static std::map<std::string, Stats> s_stats;
};

std::map<std::string, CollectionBaselineProtocolNs3::Stats> CollectionBaselineProtocolNs3::s_stats;
NS_OBJECT_ENSURE_REGISTERED (CollectionBaselineProtocolNs3);

// ===================== Main Validation =====================

std::string EnvToString (RadioEnvironment env) {
    switch (env) {
        case RadioEnvironment::INDOOR_LOS: return "indoor_office";
        case RadioEnvironment::INDUSTRIAL: return "indoor_factory";
        case RadioEnvironment::OUTDOOR_URBAN: return "outdoor_urban";
        case RadioEnvironment::OUTDOOR_SUBURBAN: return "outdoor_suburban";
        case RadioEnvironment::FREE_SPACE: return "free_space";
        case RadioEnvironment::INDOOR_NLOS: return "indoor_nlos";
        default: return "unknown";
    }
}

RadioEnvironment StringToEnv (const std::string& s) {
    if (s == "indoor_office") return RadioEnvironment::INDOOR_LOS;
    if (s == "indoor_factory") return RadioEnvironment::INDUSTRIAL;
    if (s == "outdoor_urban") return RadioEnvironment::OUTDOOR_URBAN;
    if (s == "outdoor_suburban") return RadioEnvironment::OUTDOOR_SUBURBAN;
    return RadioEnvironment::INDOOR_LOS;
}

struct EnvironmentParams {
    RadioEnvironment env;
    double pathLossExponent, shadowFadingStdDb;
};

struct ExperimentResult {
    std::string protocol, environment;
    uint32_t numNodes, numRounds, seed;
    double pdr, totalEnergy, avgEnergyPerNode, avgSnr;
    uint32_t aliveNodes, deadNodes, packetsSent, packetsDelivered;
};

ExperimentResult RunExperiment (std::string protocol, uint32_t numNodes, uint32_t numRounds,
                                 uint32_t seed, double areaWidth, double areaHeight,
                                 bool enableCas, bool enableFairness, bool enableGateway,
                                 RadioEnvironment env = RadioEnvironment::INDOOR_LOS)
{
    ExperimentResult result;
    result.protocol = protocol; result.numNodes = numNodes; result.numRounds = numRounds; result.seed = seed;
    result.environment = EnvToString(env);

    RngSeedManager::SetSeed (seed); RngSeedManager::SetRun (1);

    // Determine protocol type
    bool useAeris = (protocol.find("AERIS") == 0);
    bool useHeed = (protocol == "HEED");
    bool usePegasis = (protocol == "PEGASIS");
    bool useTeen = (protocol == "TEEN");
    bool useCollection = (protocol == "RPL-MRHOF" || protocol == "CTP");

    // Reset global stats for the appropriate protocol
    if (useAeris) AerisProtocolFull::ResetGlobalStats();
    else if (useHeed) HeedProtocolNs3::ResetGlobalStats();
    else if (usePegasis) PegasisProtocolNs3::ResetGlobalStats();
    else if (useTeen) TeenProtocolNs3::ResetGlobalStats();
    else if (useCollection) CollectionBaselineProtocolNs3::ResetGlobalStats(protocol);
    else LeachProtocolNs3::ResetGlobalStats();

    NodeContainer sensorNodes; sensorNodes.Create (numNodes);
    MobilityHelper mobility;
    mobility.SetPositionAllocator ("ns3::RandomRectanglePositionAllocator",
        "X", StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string(areaWidth) + "]"),
        "Y", StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string(areaHeight) + "]"));
    mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
    mobility.Install (sensorNodes);

    Vector bsPos (areaWidth / 2, areaHeight, 0);
    InternetStackHelper internet; internet.Install (sensorNodes);
    double simTime = numRounds + 10;

    for (uint32_t i = 0; i < sensorNodes.GetN(); ++i) {
        if (useAeris) {
            Ptr<AerisProtocolFull> app = CreateObject<AerisProtocolFull>();
            app->SetBaseStationPosition(bsPos); app->SetClusterHeadProbability(0.1);
            app->SetInitialEnergy(2.0); app->SetDataPacketSize(512); app->SetNumRounds(numRounds);
            app->SetEnableCas(enableCas); app->SetEnableFairness(enableFairness); app->SetEnableGateway(enableGateway);
            app->SetRadioEnvironment(env);
            sensorNodes.Get(i)->AddApplication(app);
            app->SetStartTime(Seconds(1.0)); app->SetStopTime(Seconds(simTime));
        } else if (useHeed) {
            Ptr<HeedProtocolNs3> app = CreateObject<HeedProtocolNs3>();
            app->SetBaseStationPosition(bsPos); app->SetClusterHeadProbability(0.05);
            app->SetInitialEnergy(2.0); app->SetDataPacketSize(512); app->SetNumRounds(numRounds);
            app->SetRadioEnvironment(env);
            sensorNodes.Get(i)->AddApplication(app);
            app->SetStartTime(Seconds(1.0)); app->SetStopTime(Seconds(simTime));
        } else if (usePegasis) {
            Ptr<PegasisProtocolNs3> app = CreateObject<PegasisProtocolNs3>();
            app->SetBaseStationPosition(bsPos);
            app->SetInitialEnergy(2.0); app->SetDataPacketSize(512); app->SetNumRounds(numRounds);
            app->SetRadioEnvironment(env);
            sensorNodes.Get(i)->AddApplication(app);
            app->SetStartTime(Seconds(1.0)); app->SetStopTime(Seconds(simTime));
        } else if (useTeen) {
            Ptr<TeenProtocolNs3> app = CreateObject<TeenProtocolNs3>();
            app->SetBaseStationPosition(bsPos); app->SetClusterHeadProbability(0.08);
            app->SetInitialEnergy(2.0); app->SetDataPacketSize(512); app->SetNumRounds(numRounds);
            app->SetRadioEnvironment(env);
            sensorNodes.Get(i)->AddApplication(app);
            app->SetStartTime(Seconds(1.0)); app->SetStopTime(Seconds(simTime));
        } else if (useCollection) {
            Ptr<CollectionBaselineProtocolNs3> app = CreateObject<CollectionBaselineProtocolNs3>();
            app->SetNodeIndex(i);
            app->SetProtocolKey(protocol);
            app->SetBaseStationPosition(bsPos);
            app->SetInitialEnergy(2.0); app->SetDataPacketSize(512); app->SetNumRounds(numRounds);
            app->SetRadioEnvironment(env);
            sensorNodes.Get(i)->AddApplication(app);
            app->SetStartTime(Seconds(1.0)); app->SetStopTime(Seconds(simTime));
        } else {
            Ptr<LeachProtocolNs3> app = CreateObject<LeachProtocolNs3>();
            app->SetBaseStationPosition(bsPos); app->SetClusterHeadProbability(0.1);
            app->SetInitialEnergy(2.0); app->SetDataPacketSize(512); app->SetNumRounds(numRounds);
            app->SetRadioEnvironment(env);
            sensorNodes.Get(i)->AddApplication(app);
            app->SetStartTime(Seconds(1.0)); app->SetStopTime(Seconds(simTime));
        }
    }

    Simulator::Stop (Seconds(simTime + 1)); Simulator::Run();

    double totalEnergy = 0; uint32_t aliveNodes = 0, deadNodes = 0;
    for (uint32_t i = 0; i < sensorNodes.GetN(); ++i) {
        Ptr<Application> baseApp = sensorNodes.Get(i)->GetApplication(0);
        if (useAeris) {
            Ptr<AerisProtocolFull> app = DynamicCast<AerisProtocolFull>(baseApp);
            if (app) { totalEnergy += app->GetTotalEnergyConsumed(); if (app->IsAlive()) aliveNodes++; else deadNodes++; }
        } else if (useHeed) {
            Ptr<HeedProtocolNs3> app = DynamicCast<HeedProtocolNs3>(baseApp);
            if (app) { totalEnergy += app->GetTotalEnergyConsumed(); if (app->IsAlive()) aliveNodes++; else deadNodes++; }
        } else if (usePegasis) {
            Ptr<PegasisProtocolNs3> app = DynamicCast<PegasisProtocolNs3>(baseApp);
            if (app) { totalEnergy += app->GetTotalEnergyConsumed(); if (app->IsAlive()) aliveNodes++; else deadNodes++; }
        } else if (useTeen) {
            Ptr<TeenProtocolNs3> app = DynamicCast<TeenProtocolNs3>(baseApp);
            if (app) { totalEnergy += app->GetTotalEnergyConsumed(); if (app->IsAlive()) aliveNodes++; else deadNodes++; }
        } else if (useCollection) {
            Ptr<CollectionBaselineProtocolNs3> app = DynamicCast<CollectionBaselineProtocolNs3>(baseApp);
            if (app) { totalEnergy += app->GetTotalEnergyConsumed(); if (app->IsAlive()) aliveNodes++; else deadNodes++; }
        } else {
            Ptr<LeachProtocolNs3> app = DynamicCast<LeachProtocolNs3>(baseApp);
            if (app) { totalEnergy += app->GetTotalEnergyConsumed(); if (app->IsAlive()) aliveNodes++; else deadNodes++; }
        }
    }

    if (useAeris) {
        result.pdr = AerisProtocolFull::GetGlobalPdr(); result.packetsSent = AerisProtocolFull::GetGlobalPacketsSent();
        result.packetsDelivered = AerisProtocolFull::GetGlobalPacketsDelivered(); result.avgSnr = AerisProtocolFull::GetGlobalAvgSnr();
    } else if (useHeed) {
        result.pdr = HeedProtocolNs3::GetGlobalPdr(); result.packetsSent = HeedProtocolNs3::GetGlobalPacketsSent();
        result.packetsDelivered = HeedProtocolNs3::GetGlobalPacketsDelivered(); result.avgSnr = HeedProtocolNs3::GetGlobalAvgSnr();
    } else if (usePegasis) {
        result.pdr = PegasisProtocolNs3::GetGlobalPdr(); result.packetsSent = PegasisProtocolNs3::GetGlobalPacketsSent();
        result.packetsDelivered = PegasisProtocolNs3::GetGlobalPacketsDelivered(); result.avgSnr = PegasisProtocolNs3::GetGlobalAvgSnr();
    } else if (useTeen) {
        result.pdr = TeenProtocolNs3::GetGlobalPdr(); result.packetsSent = TeenProtocolNs3::GetGlobalPacketsSent();
        result.packetsDelivered = TeenProtocolNs3::GetGlobalPacketsDelivered(); result.avgSnr = TeenProtocolNs3::GetGlobalAvgSnr();
    } else if (useCollection) {
        result.pdr = CollectionBaselineProtocolNs3::GetGlobalPdr(protocol); result.packetsSent = CollectionBaselineProtocolNs3::GetGlobalPacketsSent(protocol);
        result.packetsDelivered = CollectionBaselineProtocolNs3::GetGlobalPacketsDelivered(protocol); result.avgSnr = CollectionBaselineProtocolNs3::GetGlobalAvgSnr(protocol);
    } else {
        result.pdr = LeachProtocolNs3::GetGlobalPdr(); result.packetsSent = LeachProtocolNs3::GetGlobalPacketsSent();
        result.packetsDelivered = LeachProtocolNs3::GetGlobalPacketsDelivered(); result.avgSnr = LeachProtocolNs3::GetGlobalAvgSnr();
    }

    result.totalEnergy = totalEnergy; result.avgEnergyPerNode = totalEnergy / numNodes;
    result.aliveNodes = aliveNodes; result.deadNodes = deadNodes;
    Simulator::Destroy();
    return result;
}

void PrintResult (const ExperimentResult& r) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << r.protocol << " | N=" << r.numNodes << " | PDR=" << (r.pdr * 100) << "%"
              << " | E=" << (r.totalEnergy * 1000) << "mJ | Alive=" << r.aliveNodes
              << " | SNR=" << r.avgSnr << "dB" << std::endl;
}

void ExportResults (const std::vector<ExperimentResult>& results, const std::string& filename) {
    std::ofstream ofs(filename);
    ofs << "{\n  \"channel_model\": {\"type\": \"realistic_physics_based\", \"multi_environment\": true},\n";
    ofs << "  \"experiments\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        ofs << "    {\"protocol\": \"" << r.protocol
            << "\", \"environment\": \"" << r.environment
            << "\", \"num_nodes\": " << r.numNodes
            << ", \"pdr\": " << std::fixed << std::setprecision(4) << r.pdr
            << ", \"avg_snr_db\": " << std::setprecision(2) << r.avgSnr
            << ", \"alive_nodes\": " << r.aliveNodes
            << ", \"dead_nodes\": " << r.deadNodes << "}";
        if (i < results.size() - 1) ofs << ","; ofs << "\n";
    }
    ofs << "  ]\n}\n"; ofs.close();
}

int main (int argc, char *argv[])
{
    uint32_t numNodes = 100, numRounds = 300, seed = 42001;
    double areaWidth = 200.0, areaHeight = 200.0;
    bool runAll = false, runMultiEnv = false, runShard = false;
    std::string outputFile = "ns3_multienv_publication.json";
    std::string shardProtocol = "", shardEnv = "";
    std::string shardNodes = "";

    CommandLine cmd (__FILE__);
    cmd.AddValue ("numNodes", "Number of nodes (single run or comma-sep for shard)", numNodes);
    cmd.AddValue ("numRounds", "Number of rounds", numRounds);
    cmd.AddValue ("seed", "Random seed", seed);
    cmd.AddValue ("runAll", "Run single-env comprehensive validation", runAll);
    cmd.AddValue ("runMultiEnv", "Run 4-environment publication validation", runMultiEnv);
    cmd.AddValue ("runShard", "Run a single protocol+env shard (use with --protocol, --env, --nodes)", runShard);
    cmd.AddValue ("protocol", "Protocol for shard mode (AERIS/LEACH/HEED/PEGASIS/TEEN/RPL-MRHOF/CTP/ABLATION)", shardProtocol);
    cmd.AddValue ("env", "Environment for shard mode (indoor_office/indoor_factory/outdoor_urban/outdoor_suburban)", shardEnv);
    cmd.AddValue ("nodes", "Comma-separated node counts for shard mode (e.g. 50,100,200)", shardNodes);
    cmd.AddValue ("output", "Output file", outputFile);
    cmd.Parse (argc, argv);

    std::cout << "================================================\n";
    std::cout << "NS-3 AERIS Multi-Environment Validation\n";
    std::cout << "Protocols: AERIS, LEACH, HEED, PEGASIS, TEEN, RPL-MRHOF, CTP\n";
    std::cout << "Parameters: energy=2.0J, rounds=300, tx_power=10dBm\n";
    std::cout << "Seeds: 42001-42030 (n=30)\n";
    std::cout << "================================================\n";

    std::vector<ExperimentResult> allResults;

    // 30 seeds for publication tier
    std::vector<uint32_t> seeds;
    for (uint32_t s = 42001; s <= 42030; s++) seeds.push_back(s);

    // 4 environments aligned with Python
    std::vector<EnvironmentParams> envList = {
        {RadioEnvironment::INDOOR_LOS, 2.0, 4.5},
        {RadioEnvironment::INDUSTRIAL, 2.7, 8.5},
        {RadioEnvironment::OUTDOOR_URBAN, 3.4, 12.0},
        {RadioEnvironment::OUTDOOR_SUBURBAN, 2.8, 7.5}
    };

    if (runMultiEnv) {
        // Multi-environment publication run: 7 protocols, 7 node counts, 30 seeds, 4 envs
        std::vector<uint32_t> nodeCounts = {50, 100, 200, 300, 500, 800, 1000};
        std::vector<std::string> protocols = {"AERIS", "LEACH", "HEED", "PEGASIS", "TEEN", "RPL-MRHOF", "CTP"};

        for (const auto& ep : envList) {
            std::string envName = EnvToString(ep.env);
            std::cout << "\n--- Environment: " << envName << " (PLE="
                      << ep.pathLossExponent << ", sigma=" << ep.shadowFadingStdDb << "dB) ---\n";

            for (uint32_t nodes : nodeCounts) {
                for (uint32_t s : seeds) {
                    for (const auto& proto : protocols) {
                        bool cas = (proto == "AERIS"), fair = cas, gw = cas;
                        auto r = RunExperiment(proto, nodes, numRounds, s, areaWidth, areaHeight,
                                               cas, fair, gw, ep.env);
                        allResults.push_back(r);
                    }
                }
                std::cout << "  " << envName << " n=" << nodes << " done (" << seeds.size() << " seeds x " << protocols.size() << " protocols)\n";
            }

            // Ablation for this environment (100 nodes only, AERIS variants)
            std::cout << "  Ablation (" << envName << ", 100 nodes)...\n";
            for (uint32_t s : seeds) {
                auto full = RunExperiment("AERIS-FULL", 100, numRounds, s, areaWidth, areaHeight,
                                          true, true, true, ep.env);
                allResults.push_back(full);
                auto noCas = RunExperiment("AERIS-noCAS", 100, numRounds, s, areaWidth, areaHeight,
                                           false, true, true, ep.env);
                allResults.push_back(noCas);
                auto noFair = RunExperiment("AERIS-noFair", 100, numRounds, s, areaWidth, areaHeight,
                                            true, false, true, ep.env);
                allResults.push_back(noFair);
                auto noGW = RunExperiment("AERIS-noGW", 100, numRounds, s, areaWidth, areaHeight,
                                          true, true, false, ep.env);
                allResults.push_back(noGW);
            }
            std::cout << "  Ablation " << envName << " done\n";
        }
    } else if (runAll) {
        // Legacy single-env run (INDOOR_LOS only, 30 seeds, 7 protocols)
        std::vector<uint32_t> nodeCounts = {50, 100, 200};
        std::vector<std::string> protocols = {"AERIS", "LEACH", "HEED", "PEGASIS", "TEEN", "RPL-MRHOF", "CTP"};
        RadioEnvironment env = RadioEnvironment::INDOOR_LOS;

        std::cout << "\n--- Scalability Comparison (INDOOR_LOS, 7 protocols) ---\n";
        for (uint32_t nodes : nodeCounts) {
            for (uint32_t s : seeds) {
                for (const auto& proto : protocols) {
                    bool cas = (proto == "AERIS"), fair = cas, gw = cas;
                    auto r = RunExperiment(proto, nodes, numRounds, s, areaWidth, areaHeight,
                                           cas, fair, gw, env);
                    allResults.push_back(r); PrintResult(r);
                }
            }
        }

        std::cout << "\n--- Ablation Study (100 nodes, INDOOR_LOS) ---\n";
        for (uint32_t s : seeds) {
            auto full = RunExperiment("AERIS-FULL", 100, numRounds, s, areaWidth, areaHeight,
                                      true, true, true, env);
            allResults.push_back(full); PrintResult(full);
            auto noCas = RunExperiment("AERIS-noCAS", 100, numRounds, s, areaWidth, areaHeight,
                                       false, true, true, env);
            allResults.push_back(noCas); PrintResult(noCas);
            auto noFair = RunExperiment("AERIS-noFair", 100, numRounds, s, areaWidth, areaHeight,
                                        true, false, true, env);
            allResults.push_back(noFair); PrintResult(noFair);
            auto noGW = RunExperiment("AERIS-noGW", 100, numRounds, s, areaWidth, areaHeight,
                                      true, true, false, env);
            allResults.push_back(noGW); PrintResult(noGW);
        }
    } else if (runShard) {
        // Shard mode: run a single protocol in a single environment across node counts and seeds
        // Usage: --runShard --protocol=LEACH --env=indoor_factory --nodes=50,100,200,300,500,800,1000
        RadioEnvironment shardRadioEnv = StringToEnv(shardEnv);
        std::string envName = EnvToString(shardRadioEnv);

        // Parse comma-separated node counts
        std::vector<uint32_t> nodeCounts;
        if (!shardNodes.empty()) {
            std::stringstream ss(shardNodes);
            std::string token;
            while (std::getline(ss, token, ',')) nodeCounts.push_back(std::stoi(token));
        } else {
            nodeCounts = {50, 100, 200, 300, 500, 800, 1000};
        }

        std::cout << "Shard: protocol=" << shardProtocol << " env=" << envName
                  << " nodes=" << nodeCounts.size() << " counts, seeds=30\n";

        if (shardProtocol == "ABLATION") {
            // Ablation shard: run AERIS-FULL/noCAS/noFair/noGW across the requested node counts.
            uint32_t completed = 0;
            for (uint32_t nodes : nodeCounts) {
                for (uint32_t s : seeds) {
                    auto full = RunExperiment("AERIS-FULL", nodes, numRounds, s, areaWidth, areaHeight, true, true, true, shardRadioEnv);
                    allResults.push_back(full);
                    auto noCas = RunExperiment("AERIS-noCAS", nodes, numRounds, s, areaWidth, areaHeight, false, true, true, shardRadioEnv);
                    allResults.push_back(noCas);
                    auto noFair = RunExperiment("AERIS-noFair", nodes, numRounds, s, areaWidth, areaHeight, true, false, true, shardRadioEnv);
                    allResults.push_back(noFair);
                    auto noGW = RunExperiment("AERIS-noGW", nodes, numRounds, s, areaWidth, areaHeight, true, true, false, shardRadioEnv);
                    allResults.push_back(noGW);
                    completed += 4;
                }
                std::cout << "  Ablation " << envName << " n=" << nodes << " done\n";
            }
            std::cout << "  Ablation " << envName << " done (" << completed << " experiments)\n";
        } else {
            // Single protocol shard
            bool cas = (shardProtocol == "AERIS"), fair = cas, gw = cas;
            for (uint32_t nodes : nodeCounts) {
                for (uint32_t s : seeds) {
                    auto r = RunExperiment(shardProtocol, nodes, numRounds, s, areaWidth, areaHeight, cas, fair, gw, shardRadioEnv);
                    allResults.push_back(r);
                }
                std::cout << "  " << shardProtocol << " " << envName << " n=" << nodes << " done\n";
            }
        }
    } else {
        // Single experiment — all 5 protocols
        auto r1 = RunExperiment("AERIS", numNodes, numRounds, seed, areaWidth, areaHeight, true, true, true);
        allResults.push_back(r1); PrintResult(r1);
        auto r2 = RunExperiment("LEACH", numNodes, numRounds, seed, areaWidth, areaHeight, false, false, false);
        allResults.push_back(r2); PrintResult(r2);
        auto r3 = RunExperiment("HEED", numNodes, numRounds, seed, areaWidth, areaHeight, false, false, false);
        allResults.push_back(r3); PrintResult(r3);
        auto r4 = RunExperiment("PEGASIS", numNodes, numRounds, seed, areaWidth, areaHeight, false, false, false);
        allResults.push_back(r4); PrintResult(r4);
        auto r5 = RunExperiment("TEEN", numNodes, numRounds, seed, areaWidth, areaHeight, false, false, false);
        allResults.push_back(r5); PrintResult(r5);
        auto r6 = RunExperiment("RPL-MRHOF", numNodes, numRounds, seed, areaWidth, areaHeight, false, false, false);
        allResults.push_back(r6); PrintResult(r6);
        auto r7 = RunExperiment("CTP", numNodes, numRounds, seed, areaWidth, areaHeight, false, false, false);
        allResults.push_back(r7); PrintResult(r7);
    }

    ExportResults(allResults, outputFile);
    std::cout << "\nResults saved to: " << outputFile << std::endl;
    std::cout << "Total experiments: " << allResults.size() << std::endl;

    // Per-environment summary
    std::map<std::string, std::pair<double, int>> envProtoSums;
    for (const auto& r : allResults) {
        std::string key = r.environment + "|" + r.protocol;
        envProtoSums[key].first += r.pdr;
        envProtoSums[key].second++;
    }
    std::cout << "\n=== Per-Environment Summary ===" << std::endl;
    for (const auto& kv : envProtoSums) {
        double avg = kv.second.first / kv.second.second;
        std::cout << "  " << kv.first << ": avg PDR=" << std::fixed << std::setprecision(4)
                  << (avg * 100) << "% (n=" << kv.second.second << ")" << std::endl;
    }

    return 0;
}
