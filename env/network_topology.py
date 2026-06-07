"""
network_topology.py
────────────────────────────────────────────────────────────────
Enterprise network model using NetworkX DiGraph.

Architecture:
  Internet → [DMZ] → Firewall-1 → [Corporate] → Firewall-2 → [DataCenter]

Each node carries realistic metadata: OS, open services, mapped CVEs,
exploit probability, and IDS detection probability. These properties
drive the stochastic reward mechanics in the Gymnasium environment.
"""

import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List

# ──────────────────────────────────────────────────────────────
# Global constants
# ──────────────────────────────────────────────────────────────
NUM_NODES   = 9
TARGET_NODE = 8   # dc-active-directory — the crown jewel

SUBNET_ORDER = ["DMZ", "FIREWALL", "CORPORATE", "DATACENTER"]


# ──────────────────────────────────────────────────────────────
# Node metadata schema
# ──────────────────────────────────────────────────────────────
@dataclass
class NodeProperties:
    """All properties attached to a simulated host."""
    node_id:       int
    name:          str
    subnet:        str
    os:            str
    services:      List[Dict]   # [{port, service, cve, exploit_prob, detect_prob}]
    is_target:     bool = False
    firewall:      bool = False
    ids_monitored: bool = False


# ──────────────────────────────────────────────────────────────
# Node Registry  (9 nodes, 3 subnets + 2 firewalls)
# ──────────────────────────────────────────────────────────────
NODE_REGISTRY: Dict[int, NodeProperties] = {

    # ── DMZ (Internet-facing) ────────────────────────────────
    0: NodeProperties(
        node_id=0, name="dmz-webserver", subnet="DMZ",
        os="Ubuntu 22.04 / Apache 2.4",
        services=[
            {   # Path traversal → RCE
                "port": 80,  "service": "Apache 2.4.49",
                "cve":  "CVE-2021-41773",
                "exploit_prob": 0.80, "detect_prob": 0.10,
            },
            {   # Username enumeration (low-priv recon)
                "port": 22,  "service": "OpenSSH 7.4",
                "cve":  "CVE-2018-15473",
                "exploit_prob": 0.50, "detect_prob": 0.20,
            },
        ],
        ids_monitored=False,
    ),
    1: NodeProperties(
        node_id=1, name="dmz-dns", subnet="DMZ",
        os="Debian 11 / BIND 9.11",
        services=[
            {   # Assertion failure → DoS + cache poisoning
                "port": 53,  "service": "BIND 9.11",
                "cve":  "CVE-2020-8617",
                "exploit_prob": 0.40, "detect_prob": 0.15,
            },
        ],
        ids_monitored=False,
    ),

    # ── Firewall: DMZ → Corporate ────────────────────────────
    2: NodeProperties(
        node_id=2, name="fw-dmz-corp", subnet="FIREWALL",
        os="Cisco IOS 15.x",
        services=[
            {   # Directory traversal on management interface
                "port": 443, "service": "Cisco HTTPS Admin",
                "cve":  "CVE-2020-3452",
                "exploit_prob": 0.35, "detect_prob": 0.60,
            },
        ],
        firewall=True, ids_monitored=True,
    ),

    # ── Corporate Subnet ─────────────────────────────────────
    3: NodeProperties(
        node_id=3, name="corp-workstation-1", subnet="CORPORATE",
        os="Windows 10 21H2",
        services=[
            {   # EternalBlue — SMBv1 remote code execution
                "port": 445,  "service": "SMBv1",
                "cve":  "CVE-2017-0144",
                "exploit_prob": 0.75, "detect_prob": 0.40,
            },
            {   # BlueKeep — pre-auth RDP RCE
                "port": 3389, "service": "RDP",
                "cve":  "CVE-2019-0708",
                "exploit_prob": 0.60, "detect_prob": 0.50,
            },
        ],
        ids_monitored=True,
    ),
    4: NodeProperties(
        node_id=4, name="corp-workstation-2", subnet="CORPORATE",
        os="Windows 10 21H2",
        services=[
            {
                "port": 445,  "service": "SMBv1",
                "cve":  "CVE-2017-0144",
                "exploit_prob": 0.65, "detect_prob": 0.40,
            },
        ],
        ids_monitored=True,
    ),
    5: NodeProperties(
        node_id=5, name="corp-webapp", subnet="CORPORATE",
        os="CentOS 7 / Tomcat 9.0",
        services=[
            {   # Ghostcat — AJP file read / RCE
                "port": 8080, "service": "Apache Tomcat 9.0.1",
                "cve":  "CVE-2020-1938",
                "exploit_prob": 0.55, "detect_prob": 0.30,
            },
        ],
        ids_monitored=False,
    ),

    # ── Firewall: Corporate → DataCenter ─────────────────────
    6: NodeProperties(
        node_id=6, name="fw-corp-dc", subnet="FIREWALL",
        os="Palo Alto PAN-OS 9.1",
        services=[
            {   # Auth bypass on management interface
                "port": 443,  "service": "PAN-OS HTTPS Mgmt",
                "cve":  "CVE-2020-2021",
                "exploit_prob": 0.30, "detect_prob": 0.80,
            },
        ],
        firewall=True, ids_monitored=True,
    ),

    # ── Data Center ───────────────────────────────────────────
    7: NodeProperties(
        node_id=7, name="dc-database", subnet="DATACENTER",
        os="Windows Server 2019 / SQL Server 2017",
        services=[
            {   # SQL Server Reporting Services RCE
                "port": 1433, "service": "SQL Server 2017",
                "cve":  "CVE-2020-0618",
                "exploit_prob": 0.45, "detect_prob": 0.70,
            },
        ],
        ids_monitored=True,
    ),
    8: NodeProperties(                         # ← CROWN JEWEL
        node_id=8, name="dc-active-directory", subnet="DATACENTER",
        os="Windows Server 2019 / AD DS",
        services=[
            {   # NoPac — sAMAccountName spoofing → DA
                "port": 389,  "service": "LDAP / Active Directory",
                "cve":  "CVE-2021-42278",
                "exploit_prob": 0.55, "detect_prob": 0.85,
            },
            {   # PAC validation bypass → Silver/Golden Ticket
                "port": 88,   "service": "Kerberos KDC",
                "cve":  "CVE-2021-42287",
                "exploit_prob": 0.50, "detect_prob": 0.85,
            },
        ],
        is_target=True, ids_monitored=True,
    ),
}


# Subnet → node ID lists (used for rendering and reachability logic)
SUBNETS: Dict[str, List[int]] = {
    "DMZ":        [0, 1],
    "FIREWALL":   [2, 6],
    "CORPORATE":  [3, 4, 5],
    "DATACENTER": [7, 8],
}


# ──────────────────────────────────────────────────────────────
# Graph construction
# ──────────────────────────────────────────────────────────────
def build_network_graph() -> nx.DiGraph:
    """
    Build a directed graph of the enterprise network.

    Nodes  = hosts (with metadata attached as attributes)
    Edges  = valid lateral movement / attack paths

    The graph intentionally omits direct DMZ → DataCenter edges —
    the agent must pivot through Corporate and both firewalls.
    """
    G = nx.DiGraph()

    # Register all nodes with their properties
    for nid, props in NODE_REGISTRY.items():
        G.add_node(
            nid,
            name=props.name,
            subnet=props.subnet,
            os=props.os,
            services=props.services,
            is_target=props.is_target,
            firewall=props.firewall,
            ids_monitored=props.ids_monitored,
        )

    # ── Directed attack-path edges ────────────────────────────
    # DMZ lateral movement
    G.add_edges_from([(0, 1), (1, 0)])

    # DMZ → Firewall-1 (pivoting from compromised DMZ)
    G.add_edges_from([(0, 2), (1, 2)])

    # Firewall-1 → Corporate (pass-through after fw compromise)
    G.add_edges_from([(2, 3), (2, 4), (2, 5)])

    # Corporate lateral (pass-the-hash / SMB relay style)
    G.add_edges_from([
        (3, 4), (4, 3),
        (3, 5), (4, 5), (5, 3),
    ])

    # Corporate → Firewall-2
    G.add_edges_from([(3, 6), (4, 6), (5, 6)])

    # Firewall-2 → DataCenter
    G.add_edges_from([(6, 7), (6, 8)])

    # DataCenter lateral
    G.add_edges_from([(7, 8), (8, 7)])

    return G
