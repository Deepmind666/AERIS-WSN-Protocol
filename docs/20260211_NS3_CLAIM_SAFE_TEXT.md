# 20260211 NS-3 Claim Safe Text (for manuscript use)

## Allowed text (current evidence)
1. "NS-3 aligned experiments (INDOOR_LOS, n=30) provide trend-level support for AERIS-vs-LEACH."
2. "At 50 nodes, NS-3 shows a significant AERIS advantage over LEACH (Holm-adjusted p < 0.001)."
3. "At 100 and 200 nodes under INDOOR_LOS, AERIS-vs-LEACH differences are not statistically significant."
4. "Current NS-3 evidence is limited to one channel environment and therefore is not treated as numeric-level cross-platform validation."
5. "CAS contribution in good-channel NS-3 setting is mixed; noCAS outperforms FULL at 100 nodes."

## Forbidden text (current evidence)
1. "NS-3 numerically validates Python results."
2. "NS-3 confirms AERIS superiority across scales/environments."
3. "NS-3 confirms five-protocol ranking."
4. "CAS consistently improves reliability in NS-3."

## Evidence anchors
- C:\AERIS-WSN-Protocol\ns3_validation\results\ns3_aligned_significance.csv
  - line 2 (n=50 significant), line 3-4 (n=100/200 non-significant), line 5 (FULL vs noCAS significant negative)
- C:\AERIS-WSN-Protocol\ns3_validation\results\NS3_ALIGNMENT_EVIDENCE.md
  - line 237 (single environment gap), line 307 (TREND-LEVEL ONLY)
- C:\AERIS-WSN-Protocol\ns3_validation\results\ns3_aligned_publication_20260211.json
  - top-level schema lacks project publication metadata fields (needs fixed export)
