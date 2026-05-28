/**
 * AERIS Protocol Helper Implementation for NS-3
 */

#include "aeris-helper.h"
#include "ns3/log.h"
#include "ns3/names.h"
#include "ns3/string.h"
#include "ns3/uinteger.h"
#include "ns3/double.h"
#include "ns3/boolean.h"
#include <fstream>
#include <iomanip>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("AerisHelper");

AerisHelper::AerisHelper ()
{
    m_factory.SetTypeId ("ns3::AerisProtocol");
}

void
AerisHelper::SetAttribute (std::string name, const AttributeValue &value)
{
    m_factory.Set (name, value);
}

ApplicationContainer
AerisHelper::Install (Ptr<Node> node) const
{
    return ApplicationContainer (InstallPriv (node));
}

ApplicationContainer
AerisHelper::Install (NodeContainer nodes) const
{
    ApplicationContainer apps;
    for (NodeContainer::Iterator i = nodes.Begin (); i != nodes.End (); ++i)
    {
        apps.Add (InstallPriv (*i));
    }
    return apps;
}

Ptr<Application>
AerisHelper::InstallPriv (Ptr<Node> node) const
{
    Ptr<Application> app = m_factory.Create<AerisProtocol> ();
    node->AddApplication (app);

    // Set base station address if configured
    if (!m_baseStationAddr.IsAny ())
    {
        Ptr<AerisProtocol> aeris = DynamicCast<AerisProtocol> (app);
        if (aeris)
        {
            aeris->SetBaseStationAddress (m_baseStationAddr);
        }
    }

    return app;
}

void
AerisHelper::SetBaseStation (Ptr<Node> bsNode)
{
    m_baseStation = bsNode;
}

void
AerisHelper::SetBaseStationAddress (Ipv4Address addr)
{
    m_baseStationAddr = addr;
}

void
AerisHelper::EnableCas (bool enable)
{
    m_factory.Set ("EnableCAS", BooleanValue (enable));
}

void
AerisHelper::EnableFairness (bool enable)
{
    m_factory.Set ("EnableFairness", BooleanValue (enable));
}

void
AerisHelper::EnableGateway (bool enable)
{
    m_factory.Set ("EnableGateway", BooleanValue (enable));
}

void
AerisHelper::SetSafetyProfile (AerisSafetyProfile profile)
{
    // Convert enum to attribute
    // (Would need proper enum attribute support)
}

void
AerisHelper::SetClusterHeadProbability (double prob)
{
    m_factory.Set ("ClusterHeadProbability", DoubleValue (prob));
}

void
AerisHelper::SetInitialEnergy (double energy)
{
    m_factory.Set ("InitialEnergy", DoubleValue (energy));
}

void
AerisHelper::SetSimulationDuration (Time duration)
{
    m_factory.Set ("RoundDuration", TimeValue (duration));
}

// ==================== Stats Collector ====================

AerisStatsCollector::AerisStatsCollector ()
    : m_totalPacketsSent (0),
      m_totalPacketsReceived (0),
      m_totalEnergy (0.0),
      m_aliveNodes (0),
      m_deadNodes (0),
      m_networkLifetime (0)
{
}

void
AerisStatsCollector::Collect (ApplicationContainer apps)
{
    m_totalPacketsSent = 0;
    m_totalPacketsReceived = 0;
    m_totalEnergy = 0.0;
    m_aliveNodes = 0;
    m_deadNodes = 0;
    m_nodeEnergies.clear ();
    m_nodePdrs.clear ();

    for (uint32_t i = 0; i < apps.GetN (); ++i)
    {
        Ptr<AerisProtocol> aeris = DynamicCast<AerisProtocol> (apps.Get (i));
        if (aeris)
        {
            uint32_t sent = aeris->GetPacketsSent ();
            uint32_t received = aeris->GetPacketsReceived ();
            double energy = aeris->GetTotalEnergyConsumed ();
            double residual = aeris->GetResidualEnergy ();

            m_totalPacketsSent += sent;
            m_totalPacketsReceived += received;
            m_totalEnergy += energy;

            if (residual > 0)
            {
                m_aliveNodes++;
            }
            else
            {
                m_deadNodes++;
            }

            m_nodeEnergies.push_back (energy);

            if (sent > 0)
            {
                m_nodePdrs.push_back (static_cast<double> (received) / sent);
            }
        }
    }
}

double
AerisStatsCollector::GetOverallPdr () const
{
    if (m_totalPacketsSent == 0)
    {
        return 0.0;
    }
    return static_cast<double> (m_totalPacketsReceived) / m_totalPacketsSent;
}

double
AerisStatsCollector::GetTotalEnergy () const
{
    return m_totalEnergy;
}

double
AerisStatsCollector::GetAverageEnergy () const
{
    uint32_t totalNodes = m_aliveNodes + m_deadNodes;
    if (totalNodes == 0)
    {
        return 0.0;
    }
    return m_totalEnergy / totalNodes;
}

uint32_t
AerisStatsCollector::GetNetworkLifetime () const
{
    return m_networkLifetime;
}

uint32_t
AerisStatsCollector::GetAliveNodes () const
{
    return m_aliveNodes;
}

uint32_t
AerisStatsCollector::GetDeadNodes () const
{
    return m_deadNodes;
}

void
AerisStatsCollector::ExportJson (std::string filename) const
{
    std::ofstream ofs (filename);
    ofs << "{\n";
    ofs << "  \"pdr\": " << std::fixed << std::setprecision (4) << GetOverallPdr () << ",\n";
    ofs << "  \"total_energy\": " << std::fixed << std::setprecision (6) << m_totalEnergy << ",\n";
    ofs << "  \"avg_energy\": " << std::fixed << std::setprecision (6) << GetAverageEnergy () << ",\n";
    ofs << "  \"alive_nodes\": " << m_aliveNodes << ",\n";
    ofs << "  \"dead_nodes\": " << m_deadNodes << ",\n";
    ofs << "  \"packets_sent\": " << m_totalPacketsSent << ",\n";
    ofs << "  \"packets_received\": " << m_totalPacketsReceived << "\n";
    ofs << "}\n";
    ofs.close ();
}

void
AerisStatsCollector::ExportCsv (std::string filename) const
{
    std::ofstream ofs (filename);
    ofs << "metric,value\n";
    ofs << "pdr," << std::fixed << std::setprecision (4) << GetOverallPdr () << "\n";
    ofs << "total_energy," << m_totalEnergy << "\n";
    ofs << "avg_energy," << GetAverageEnergy () << "\n";
    ofs << "alive_nodes," << m_aliveNodes << "\n";
    ofs << "dead_nodes," << m_deadNodes << "\n";
    ofs << "packets_sent," << m_totalPacketsSent << "\n";
    ofs << "packets_received," << m_totalPacketsReceived << "\n";
    ofs.close ();
}

void
AerisStatsCollector::PrintSummary () const
{
    std::cout << "========== AERIS Experiment Results ==========" << std::endl;
    std::cout << "PDR:             " << std::fixed << std::setprecision (2)
              << (GetOverallPdr () * 100) << "%" << std::endl;
    std::cout << "Total Energy:    " << std::fixed << std::setprecision (4)
              << (m_totalEnergy * 1000) << " mJ" << std::endl;
    std::cout << "Avg Energy/Node: " << std::fixed << std::setprecision (4)
              << (GetAverageEnergy () * 1000) << " mJ" << std::endl;
    std::cout << "Alive Nodes:     " << m_aliveNodes << std::endl;
    std::cout << "Dead Nodes:      " << m_deadNodes << std::endl;
    std::cout << "Packets Sent:    " << m_totalPacketsSent << std::endl;
    std::cout << "Packets Received:" << m_totalPacketsReceived << std::endl;
    std::cout << "==============================================" << std::endl;
}

} // namespace ns3
