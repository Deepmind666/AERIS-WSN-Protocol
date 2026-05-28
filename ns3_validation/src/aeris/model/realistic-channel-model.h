/**
 * Realistic Channel Model for WSN Cross-Validation
 *
 * Implements physics-based channel modeling including:
 * - Log-distance path loss model
 * - Shadow fading (log-normal)
 * - Multi-path fading (Rayleigh/Rician)
 * - SNR-based packet error rate
 * - CC2420 radio characteristics
 *
 * References:
 * - IEEE 802.15.4 Standard
 * - Rappaport, "Wireless Communications", 2nd Ed.
 * - CC2420 Datasheet (Texas Instruments)
 */

#ifndef REALISTIC_CHANNEL_MODEL_H
#define REALISTIC_CHANNEL_MODEL_H

#include "ns3/random-variable-stream.h"
#include "ns3/vector.h"
#include <cmath>
#include <map>

namespace ns3 {

/**
 * Radio environment type
 */
enum class RadioEnvironment {
    FREE_SPACE,      // Outdoor, line-of-sight
    INDOOR_LOS,      // Indoor, line-of-sight
    INDOOR_NLOS,     // Indoor, non-line-of-sight
    OUTDOOR_URBAN,   // Outdoor urban environment
    INDUSTRIAL       // Industrial/factory environment
};

/**
 * Link quality indicator
 */
struct LinkQualityIndicator {
    double rssi;           // Received Signal Strength (dBm)
    double snr;            // Signal-to-Noise Ratio (dB)
    double lqi;            // Link Quality Indicator [0-255]
    double per;            // Packet Error Rate [0-1]
    double successProb;    // Transmission success probability
    bool isReliable;       // Link considered reliable (SNR > threshold)
};

/**
 * Realistic Channel Model for IEEE 802.15.4 / CC2420
 */
class RealisticChannelModel
{
public:
    RealisticChannelModel ();

    /**
     * Set the radio environment
     */
    void SetEnvironment (RadioEnvironment env);

    /**
     * Set transmission power (dBm)
     * CC2420 range: -25 to 0 dBm
     */
    void SetTxPower (double txPowerDbm);

    /**
     * Calculate link quality for a given transmission
     * @param txPos Transmitter position
     * @param rxPos Receiver position
     * @param packetSizeBytes Packet size in bytes
     * @return LinkQualityIndicator with all metrics
     */
    LinkQualityIndicator CalculateLinkQuality (Vector txPos, Vector rxPos,
                                                uint32_t packetSizeBytes);

    /**
     * Determine if packet transmission succeeds
     * @param txPos Transmitter position
     * @param rxPos Receiver position
     * @param packetSizeBytes Packet size
     * @return true if packet delivered successfully
     */
    bool TransmitPacket (Vector txPos, Vector rxPos, uint32_t packetSizeBytes);

    /**
     * Get path loss in dB
     */
    double GetPathLoss (double distanceMeters);

    /**
     * Get shadow fading component (dB)
     */
    double GetShadowFading ();

    /**
     * Get multi-path fading component (linear scale)
     */
    double GetMultipathFading ();

    /**
     * Calculate RSSI from distance
     */
    double CalculateRssi (double distanceMeters);

    /**
     * Calculate SNR from RSSI
     */
    double CalculateSnr (double rssiDbm);

    /**
     * Calculate Packet Error Rate from SNR
     * Based on O-QPSK modulation (IEEE 802.15.4)
     */
    double CalculatePer (double snrDb, uint32_t packetSizeBytes);

    /**
     * Convert RSSI to LQI (Link Quality Indicator)
     * CC2420 LQI range: 50-110 (typical)
     */
    uint8_t RssiToLqi (double rssiDbm);

    /**
     * Get environment parameters as string (for logging)
     */
    std::string GetEnvironmentDescription () const;

private:
    // Environment-specific parameters
    RadioEnvironment m_environment;
    double m_pathLossExponent;      // n: typically 2-4
    double m_referencePathLoss;     // PL(d0) at reference distance
    double m_shadowingStdDev;       // σ: shadow fading std dev (dB)
    double m_referenceDistance;     // d0: reference distance (m)

    // CC2420 Radio parameters
    double m_txPowerDbm;            // Transmission power (dBm)
    double m_rxSensitivityDbm;      // Receiver sensitivity (-95 dBm typical)
    double m_noiseFigureDb;         // Noise figure (typically 6 dB)
    double m_thermalNoiseDbm;       // Thermal noise floor

    // IEEE 802.15.4 parameters
    static constexpr double CARRIER_FREQ_HZ = 2.4e9;      // 2.4 GHz
    static constexpr double BANDWIDTH_HZ = 2e6;           // 2 MHz
    static constexpr double DATA_RATE_BPS = 250e3;        // 250 kbps
    static constexpr double CHIP_RATE = 2e6;              // 2 Mchip/s

    // Random number generators
    Ptr<NormalRandomVariable> m_shadowingRng;
    Ptr<UniformRandomVariable> m_uniformRng;
    Ptr<ExponentialRandomVariable> m_rayleighRng;

    // Internal methods
    void ConfigureEnvironment ();
    double DbmToWatts (double dbm);
    double WattsToDbm (double watts);
};

// ======================= Implementation =======================

RealisticChannelModel::RealisticChannelModel ()
    : m_environment (RadioEnvironment::INDOOR_LOS),
      m_pathLossExponent (3.0),
      m_referencePathLoss (40.0),
      m_shadowingStdDev (4.0),
      m_referenceDistance (1.0),
      m_txPowerDbm (0.0),
      m_rxSensitivityDbm (-95.0),
      m_noiseFigureDb (6.0)
{
    // Calculate thermal noise: kTB
    // k = 1.38e-23 J/K, T = 290K, B = 2MHz
    double thermalNoiseWatts = 1.38e-23 * 290 * BANDWIDTH_HZ;
    m_thermalNoiseDbm = WattsToDbm (thermalNoiseWatts);

    // Initialize random number generators
    m_shadowingRng = CreateObject<NormalRandomVariable> ();
    m_shadowingRng->SetAttribute ("Mean", DoubleValue (0.0));
    m_shadowingRng->SetAttribute ("Variance", DoubleValue (m_shadowingStdDev * m_shadowingStdDev));

    m_uniformRng = CreateObject<UniformRandomVariable> ();
    m_rayleighRng = CreateObject<ExponentialRandomVariable> ();
    m_rayleighRng->SetAttribute ("Mean", DoubleValue (1.0));

    ConfigureEnvironment ();
}

void
RealisticChannelModel::SetEnvironment (RadioEnvironment env)
{
    m_environment = env;
    ConfigureEnvironment ();
}

void
RealisticChannelModel::ConfigureEnvironment ()
{
    // Set environment-specific parameters based on empirical studies
    // Reference: IEEE 802.15.4 channel models, Rappaport textbook
    switch (m_environment)
    {
        case RadioEnvironment::FREE_SPACE:
            m_pathLossExponent = 2.0;
            m_referencePathLoss = 40.0;  // Free space at 1m, 2.4GHz
            m_shadowingStdDev = 0.0;
            break;

        case RadioEnvironment::INDOOR_LOS:
            // Indoor line-of-sight (office, lab)
            m_pathLossExponent = 2.5;
            m_referencePathLoss = 40.0;
            m_shadowingStdDev = 3.0;
            break;

        case RadioEnvironment::INDOOR_NLOS:
            // Indoor non-line-of-sight (walls, obstacles)
            m_pathLossExponent = 3.5;
            m_referencePathLoss = 45.0;
            m_shadowingStdDev = 6.0;
            break;

        case RadioEnvironment::OUTDOOR_URBAN:
            // Outdoor urban environment
            m_pathLossExponent = 3.0;
            m_referencePathLoss = 38.0;
            m_shadowingStdDev = 4.0;
            break;

        case RadioEnvironment::INDUSTRIAL:
            // Industrial/factory environment (metal, machinery)
            m_pathLossExponent = 3.8;
            m_referencePathLoss = 48.0;
            m_shadowingStdDev = 8.0;
            break;
    }

    // Update shadowing RNG variance
    m_shadowingRng->SetAttribute ("Variance",
        DoubleValue (m_shadowingStdDev * m_shadowingStdDev));
}

void
RealisticChannelModel::SetTxPower (double txPowerDbm)
{
    // CC2420 valid range: -25 to 0 dBm
    m_txPowerDbm = std::max (-25.0, std::min (0.0, txPowerDbm));
}

double
RealisticChannelModel::GetPathLoss (double distanceMeters)
{
    if (distanceMeters < m_referenceDistance)
    {
        distanceMeters = m_referenceDistance;
    }

    // Log-distance path loss model:
    // PL(d) = PL(d0) + 10 * n * log10(d/d0)
    double pathLoss = m_referencePathLoss +
                      10.0 * m_pathLossExponent *
                      std::log10 (distanceMeters / m_referenceDistance);

    return pathLoss;
}

double
RealisticChannelModel::GetShadowFading ()
{
    // Log-normal shadow fading
    // Returns value in dB (can be positive or negative)
    return m_shadowingRng->GetValue ();
}

double
RealisticChannelModel::GetMultipathFading ()
{
    // Rayleigh fading for NLOS, Rician for LOS
    // Returns linear scale factor
    if (m_environment == RadioEnvironment::FREE_SPACE ||
        m_environment == RadioEnvironment::INDOOR_LOS)
    {
        // Rician fading (K-factor = 6 dB for LOS)
        double K = std::pow (10.0, 0.6);  // K = 6 dB
        double scattered = m_rayleighRng->GetValue ();
        double los = K;
        return (los + scattered) / (1.0 + K);
    }
    else
    {
        // Rayleigh fading for NLOS
        return m_rayleighRng->GetValue ();
    }
}

double
RealisticChannelModel::CalculateRssi (double distanceMeters)
{
    // RSSI = Tx Power - Path Loss - Shadow Fading
    double pathLoss = GetPathLoss (distanceMeters);
    double shadowFading = GetShadowFading ();

    double rssi = m_txPowerDbm - pathLoss - shadowFading;

    return rssi;
}

double
RealisticChannelModel::CalculateSnr (double rssiDbm)
{
    // SNR = RSSI - Noise Floor
    // Noise Floor = Thermal Noise + Noise Figure
    double noiseFloorDbm = m_thermalNoiseDbm + m_noiseFigureDb;

    return rssiDbm - noiseFloorDbm;
}

double
RealisticChannelModel::CalculatePer (double snrDb, uint32_t packetSizeBytes)
{
    // Packet Error Rate for O-QPSK modulation (IEEE 802.15.4)
    // Based on BER approximation and packet size
    //
    // BER ≈ 0.5 * erfc(sqrt(SNR_linear))
    // PER = 1 - (1 - BER)^(8 * packet_size)

    double snrLinear = std::pow (10.0, snrDb / 10.0);

    // Chip error rate for DSSS with 16-ary orthogonal modulation
    // Approximation for IEEE 802.15.4 O-QPSK
    double ber;

    if (snrDb < 0)
    {
        ber = 0.5;  // Very poor channel
    }
    else if (snrDb > 20)
    {
        ber = 1e-10;  // Excellent channel
    }
    else
    {
        // BER approximation for O-QPSK with DSSS
        // Reference: IEEE 802.15.4 PHY performance
        double Eb_N0 = snrLinear * (BANDWIDTH_HZ / DATA_RATE_BPS);
        ber = 0.5 * std::erfc (std::sqrt (Eb_N0));
    }

    // PER from BER (assuming independent bit errors)
    uint32_t packetSizeBits = packetSizeBytes * 8;
    double per = 1.0 - std::pow (1.0 - ber, packetSizeBits);

    return std::min (1.0, std::max (0.0, per));
}

uint8_t
RealisticChannelModel::RssiToLqi (double rssiDbm)
{
    // CC2420 LQI correlation with RSSI
    // LQI range: 50 (poor) to 110 (excellent)
    // RSSI range: -95 dBm to -20 dBm (typical)

    if (rssiDbm < -95.0)
    {
        return 0;  // Below sensitivity
    }
    else if (rssiDbm > -20.0)
    {
        return 255;  // Very strong signal
    }

    // Linear mapping: RSSI=-95 -> LQI=50, RSSI=-20 -> LQI=110
    double lqi = 50.0 + (rssiDbm + 95.0) * (110.0 - 50.0) / (95.0 - 20.0);
    return static_cast<uint8_t> (std::min (255.0, std::max (0.0, lqi)));
}

LinkQualityIndicator
RealisticChannelModel::CalculateLinkQuality (Vector txPos, Vector rxPos,
                                              uint32_t packetSizeBytes)
{
    LinkQualityIndicator lqi;

    // Calculate distance
    double dx = txPos.x - rxPos.x;
    double dy = txPos.y - rxPos.y;
    double dz = txPos.z - rxPos.z;
    double distance = std::sqrt (dx*dx + dy*dy + dz*dz);

    // Minimum distance
    if (distance < 0.1) distance = 0.1;

    // Calculate RSSI with all fading components
    double pathLoss = GetPathLoss (distance);
    double shadowFading = GetShadowFading ();
    double multipathFading = GetMultipathFading ();

    // RSSI in dBm
    lqi.rssi = m_txPowerDbm - pathLoss - shadowFading;

    // Apply multipath fading (linear domain)
    double rssiLinear = DbmToWatts (lqi.rssi) * multipathFading;
    lqi.rssi = WattsToDbm (rssiLinear);

    // Calculate SNR
    lqi.snr = CalculateSnr (lqi.rssi);

    // Calculate LQI (CC2420 style)
    lqi.lqi = RssiToLqi (lqi.rssi);

    // Calculate PER
    lqi.per = CalculatePer (lqi.snr, packetSizeBytes);

    // Success probability = 1 - PER
    lqi.successProb = 1.0 - lqi.per;

    // Link is reliable if SNR > 8 dB (typical threshold for 802.15.4)
    lqi.isReliable = (lqi.snr > 8.0);

    return lqi;
}

bool
RealisticChannelModel::TransmitPacket (Vector txPos, Vector rxPos,
                                        uint32_t packetSizeBytes)
{
    LinkQualityIndicator lqi = CalculateLinkQuality (txPos, rxPos, packetSizeBytes);

    // Check if RSSI above sensitivity
    if (lqi.rssi < m_rxSensitivityDbm)
    {
        return false;  // Signal too weak
    }

    // Probabilistic packet reception based on PER
    double randomValue = m_uniformRng->GetValue (0.0, 1.0);
    return randomValue > lqi.per;
}

std::string
RealisticChannelModel::GetEnvironmentDescription () const
{
    std::string desc;
    switch (m_environment)
    {
        case RadioEnvironment::FREE_SPACE:
            desc = "Free Space (n=2.0, σ=0dB)";
            break;
        case RadioEnvironment::INDOOR_LOS:
            desc = "Indoor LOS (n=2.5, σ=3dB)";
            break;
        case RadioEnvironment::INDOOR_NLOS:
            desc = "Indoor NLOS (n=3.5, σ=6dB)";
            break;
        case RadioEnvironment::OUTDOOR_URBAN:
            desc = "Outdoor Urban (n=3.0, σ=4dB)";
            break;
        case RadioEnvironment::INDUSTRIAL:
            desc = "Industrial (n=3.8, σ=8dB)";
            break;
    }
    return desc;
}

double
RealisticChannelModel::DbmToWatts (double dbm)
{
    return std::pow (10.0, (dbm - 30.0) / 10.0);
}

double
RealisticChannelModel::WattsToDbm (double watts)
{
    return 10.0 * std::log10 (watts) + 30.0;
}

} // namespace ns3

#endif /* REALISTIC_CHANNEL_MODEL_H */
