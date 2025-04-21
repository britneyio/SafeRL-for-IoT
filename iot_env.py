"""Model a Smart Home Environment """
""" See Jupyter notebook smarthome-simulation.ipynb for an example of game
played on this simulation"""

from cyberbattle.simulation import model as m
from cyberbattle.simulation.model import VulnerabilityInfo, FirewallConfiguration, FirewallRule, Identifiers, RulePermission, VulnerabilityType, Rates, PrivilegeEscalation, LeakedCredentials, CachedCredential, LeakedNodesId
from typing import Dict, Iterator, cast, Tuple
import cyberbattle._env.cyberbattle_env as cyberbattle_env
import gymnasium as gym
from gymnasium import spaces

firewall_conf = FirewallConfiguration(
    [
        FirewallRule("SSH", RulePermission.BLOCK),
        FirewallRule("HTTP", RulePermission.ALLOW)
    ],
    [
        FirewallRule("SSH", RulePermission.ALLOW),
        FirewallRule("HTTP", RulePermission.ALLOW)
    ])



def ip_gateway_vulnerabilities() -> m.VulnerabilityLibrary:
    return {
        "WeakPassword": VulnerabilityInfo(
            description="Weak SSH password allows credential theft",
            type=VulnerabilityType.REMOTE,
            outcome=LeakedCredentials([CachedCredential(node="IPGateway", port="SSH", credential="admin123")]),
            rates=Rates(successRate=0.8, probingDetectionRate=0.1, exploitDetectionRate=0.3)
        ),
        "UnauthenticatedPathTraversal": VulnerabilityInfo(
            description="Sensitive user files can be accessed",
            type=VulnerabilityType.REMOTE,
            outcome=m.LateralMove(),
            rates=Rates(successRate=0.6),
        ),
        "OldBackupsOnFrontEnd": VulnerabilityInfo(
            description="Accessible backups leak device topology",
            type=VulnerabilityType.REMOTE,
            outcome=LeakedNodesId(["SmartControlWebsite", "SmartCamera"]),
            rates=Rates(successRate=0.7)
        ),
        "OpenRouterConfig": VulnerabilityInfo(
            description="Router exposes config page without authentication",
            type=VulnerabilityType.REMOTE,
            outcome=PrivilegeEscalation(m.PrivilegeLevel.Admin),
            rates=Rates(successRate=0.9, exploitDetectionRate=0.4)
        )
    }

def smart_device_vulnerabilities() -> m.VulnerabilityLibrary:
    return {
        "StealCustomerData": VulnerabilityInfo(
            description="Device leaks personal user data",
            type=VulnerabilityType.REMOTE,
            outcome=m.CustomerData(),
            rates=Rates(successRate=0.85),
                    ),
        "SendCustomerData": VulnerabilityInfo(
            description="Streams to attacker server",
            type=VulnerabilityType.REMOTE,
            outcome=m.CustomerData(),
            rates=Rates(successRate=0.9, exploitDetectionRate=0.3),
                    ),
        "ChangeValues": VulnerabilityInfo(
            description="Compromises device control (e.g., temp, lock)",
            type=VulnerabilityType.REMOTE,
            outcome=PrivilegeEscalation(m.PrivilegeLevel.Admin),
            rates=Rates(successRate=0.7),
                    ),
        "RemoteUnlockDoor": VulnerabilityInfo(
            description="Smart lock system uses static unlock token that can be replayed",
            type=VulnerabilityType.REMOTE,
            outcome=PrivilegeEscalation(m.PrivilegeLevel.Admin),
            rates=Rates(successRate=0.8, exploitDetectionRate=0.2),
        ),
        "HardcodedKey": VulnerabilityInfo(
            description="Hardcoded key allows bypass of authorization on door unlock API",
            type=VulnerabilityType.REMOTE,
            outcome=LeakedCredentials([CachedCredential(node="SmartLock", port="HTTPS", credential="unlock_token")]),
            rates=Rates(successRate=0.85),
        ),
        "CloudlessNoAuthBypass": VulnerabilityInfo(
            description="No TLS and no authentication for local network commands to smart lock",
            type=VulnerabilityType.LOCAL,
            outcome=PrivilegeEscalation(m.PrivilegeLevel.Admin),
            rates=Rates(successRate=0.75, exploitDetectionRate=0.1),
        ),
        "VideoLeak": VulnerabilityInfo(
            description="Camera streams can be intercepted due to lack of encryption",
            type=VulnerabilityType.REMOTE,
            outcome=m.CustomerData(),
            rates=Rates(successRate=0.9),
        ),
        "AudioEavesdropping": VulnerabilityInfo(
            description="Assistant transmits voice data to unauthorized recipients",
            type=VulnerabilityType.REMOTE,
            outcome=m.CustomerData(),
            rates=Rates(successRate=0.85),
        ),
        "BankCredentialLeak": VulnerabilityInfo(
            description="Compromised assistant exposes banking credentials",
            type=VulnerabilityType.REMOTE,
            outcome=LeakedCredentials([CachedCredential(node="SmartAssistant", port="HTTPS", credential="bank_user:bank_pass")]),
            rates=Rates(successRate=0.8),
        )
    }

def router_vulnerabilities() -> m.VulnerabilityLibrary:
    return {
        "ScanForDevices": VulnerabilityInfo(
            description="Scanning reveals local network devices",
            type=VulnerabilityType.LOCAL,
            outcome=LeakedNodesId(["SmartCamera", "SmartLock", "SmartLight", "SmartBabyMonitor", "Computer"]),
            rates=Rates(successRate=0.9),
        ),
        "UPnPExploit": VulnerabilityInfo(
            description="Universal Plug and Play misconfiguration allows remote access",
            type=VulnerabilityType.REMOTE,
            outcome=PrivilegeEscalation(m.PrivilegeLevel.Admin),
            rates=Rates(successRate=0.6, exploitDetectionRate=0.4),
        ),
        "RouterCredentialDump": VulnerabilityInfo(
            description="Router debug interface leaks admin credentials",
            type=VulnerabilityType.LOCAL,
            outcome=LeakedCredentials([CachedCredential(node="Computer", port="SSH", credential="root:admin")]),
            rates=Rates(successRate=0.7),
        )
    }

def non_iot_vulnerabilities() -> m.VulnerabilityLibrary:
    return {
        "StolenCredentials": VulnerabilityInfo(
            description="Insecure credential storage in browser",
            type=VulnerabilityType.LOCAL,
            outcome=LeakedCredentials([CachedCredential(node="Computer", port="HTTPS", credential="user:pass123")]),
            rates=Rates(successRate=0.85),
                    ),
        "MaliciousBrowserPlugin": VulnerabilityInfo(
            description="Browser plugin captures keystrokes and form data",
            type=VulnerabilityType.LOCAL,
            outcome=m.CustomerData(),
            rates=Rates(successRate=0.8),
                    ),
        "AutoRunMalware": VulnerabilityInfo(
            description="USB-based malware executes via AutoRun",
            type=VulnerabilityType.LOCAL,
            outcome=PrivilegeEscalation(m.PrivilegeLevel.Admin),
            rates=Rates(successRate=0.75, exploitDetectionRate=0.3),
                    )
    }


# default vulns for smart home
# weak password
# connected to the internet
# - availability of user info/data
# - unauthorized remote control
# - unpatched software
# - unencrypyed data -> leaked credentials

# amazon alexa (audio data)
# smart camera - video/audio data
# light bulb - turn the lights out
# door locks - drain the bttery / lock you out
# smart thermostat -> change the temp
# your pc -> steal credentials / usage history
# your phone -> steal message / call data
# your smart baby monitor ->
smart_camera = m.NodeInfo(
    services=[m.ListeningService("HTTPS")],
    firewall=firewall_conf,
    value=90,
    properties=["Linux", "camera", "v1.0"],
    vulnerabilities=smart_device_vulnerabilities()
)
#wifi connection with internet access required
#requires a web browers or a compatible phone or tablet with the free nest app
# Nest Thermostat
smart_thermostat = m.NodeInfo(
    services=[m.ListeningService("HTTPS")],
    firewall=FirewallConfiguration(
    [
        FirewallRule("SSH", RulePermission.ALLOW),
        FirewallRule("HTTP", RulePermission.ALLOW)
    ],
    [
        FirewallRule("SSH", RulePermission.ALLOW),
        FirewallRule("HTTP", RulePermission.ALLOW)
    ]),
    value=30,
    properties=["Linux", "thermostat", "Nest", "v2.6.37"],
    vulnerabilities=smart_device_vulnerabilities()
)
#smart assistant - Google Home
# can control nest thermostat
google_home = m.NodeInfo(
    services=[m.ListeningService("HTTPS")],
    firewall=firewall_conf,
    value=95,
    properties=["Google Fuchsia", "assistant", "F23", "Google Home Hub"],
    vulnerabilities=smart_device_vulnerabilities()
)
#smart door locks
smart_lock = m.NodeInfo(
    services=[m.ListeningService("HTTPS")],
    firewall=firewall_conf,
    value=20,
    properties=["Linux", "lock", "v2.6.37"],
    vulnerabilities=smart_device_vulnerabilities()
)
# smart light
smart_light = m.NodeInfo(
    services=[m.ListeningService("HTTPS")],
    firewall=firewall_conf,
    value=20,
    properties=["Linux", "light", "v2.6.37"],
    vulnerabilities=smart_device_vulnerabilities()
)
# smart app / smart control/setup website
website = m.NodeInfo(
    services=[m.ListeningService("HTTPS",allowedCredentials=["firmware_update"])],
    firewall=firewall_conf,
    value=20,
    agent_installed=True,
    properties=["html/css", "control_panel"],
    vulnerabilities=smart_device_vulnerabilities()
)
# smart baby monitor
baby_monitor = m.NodeInfo(
    services=[m.ListeningService("HTTPS")],
    firewall=firewall_conf,
    value=100,
    properties=["Linux", "camera", "baby_monitor"],
    vulnerabilities=smart_device_vulnerabilities()
)

# router
ip_gateway = m.NodeInfo(
    services=[m.ListeningService("HTTPS"), m.ListeningService("SSH")],
    firewall=FirewallConfiguration(
    [
        FirewallRule("SSH", RulePermission.ALLOW),
        FirewallRule("HTTP", RulePermission.ALLOW)
    ],
    [
        FirewallRule("SSH", RulePermission.ALLOW),
        FirewallRule("HTTP", RulePermission.ALLOW)
    ]),
    value=50,
    properties=["Linux", "router", "v2.6.37"],
    vulnerabilities=ip_gateway_vulnerabilities()
)

# computer
computer = m.NodeInfo(
    services=[m.ListeningService("HTTPS")],
    firewall=firewall_conf,
    value=90,
    properties=["Windows", "user_device", "v11"],
    vulnerabilities=non_iot_vulnerabilities()
)
nodes = {
    "SmartCamera" : smart_camera,
    "SmartAssistant": google_home,
    "SmartLock": smart_lock,
    "SmartLight": smart_light,
    "SmartBabyMonitor": baby_monitor,
    "SmartControlWebsite": website,
    "IPGateway": ip_gateway,
    "Computer": computer
}
global_vulnerability_library : Dict[m.VulnerabilityID, m.VulnerabilityInfo] = dict([])

ENV_IDENTIFIERS = Identifiers(
    properties=["ip_gateway", "IoT_device", "root", "camera", "Windows", "baby_monitor", "v2.6.37", "router", "lock",
        "v1.0", "user_device", "F23", "Linux", "Google Fuchsia", "light",
        "html/css", "Google Home Hub", "assistant", "v11", "control_panel"],
    ports=["HTTPS", "SSH"],
    local_vulnerabilities=[
    "AutoRunMalware",
    "CloudlessNoAuthBypass",
    "MaliciousBrowserPlugin",
    "RouterCredentialDump",
    "ScanForDevices",
    "StolenCredentials"
]
,
    remote_vulnerabilities=[
    "AudioEavesdropping",
    "BankCredentialLeak",
    "HardcodedKey",
    "OldBackupsOnFrontEnd",
    "OpenRouterConfig",
    "RemoteUnlockDoor",
    "SendCustomerData",
    "StealCustomerData",
    "UPnPExploit",
    "UnauthenticatedPathTraversal",
    "VideoLeak",
    "WeakPassword",
        "ChangeValues"
]

)

def new_environment() -> m.Environment:
    return m.Environment(
        network=m.create_network(nodes),
        vulnerability_library=global_vulnerability_library,
        identifiers=ENV_IDENTIFIERS
    )

class FlattenMultiDiscreteWrapper(gym.ActionWrapper):
    """Wrapper to flatten multidimensional MultiDiscrete action space to 1D array."""

    def __init__(self, env):
        super().__init__(env)
        if isinstance(env.action_space, spaces.MultiDiscrete):
            # Flatten the MultiDiscrete space to 1D
            self.action_space = spaces.MultiDiscrete(env.action_space.nvec.flatten())

    def action(self, action):
        # No transformation needed since FlattenActionWrapper already handles this
        return action

    def reverse_action(self, action):
        # No transformation needed
        return action

class CyberBattleIoT(cyberbattle_env.CyberBattleEnv):
    """CyberBattle simulation based on a smart home environment"""
    def __init__(self, **kwargs):
        super().__init__(initial_environment=new_environment(), **kwargs)
