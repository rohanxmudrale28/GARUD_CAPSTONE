"""
Rule-based risk scoring for GARUD's cybersecurity workstream.

This version uses simple and explainable rules. It should not be
described as machine learning. ML-based threat scoring can be added
later after enough authorized laboratory scan data is collected.
"""

from typing import List

from cybersecurity.config import RISK_POINTS
from cybersecurity.models import (
    HTTPResult,
    PortResult,
    SecurityFinding,
)


class RiskEngine:
    def generate_findings(
        self,
        port_results: List[PortResult],
        http_results: List[HTTPResult],
    ) -> List[SecurityFinding]:
        """
        Generate security findings from the scan results.

        Args Results of the selected TCP port checks.
            http_results: Results of safe HTTP or HTTPS checks.

        Returns:
            A list of security findings.
        """
        findings: List[SecurityFinding] = []

        open_ports = {
            result.port
            for result in port_results
            if result.is_open
        }

        # Telnet exposure
        if 23 in open_ports:
            findings.append(
                self._create_finding(
                    finding_code="CYBER-TELNET-001",
                    title="Telnet service is exposed",
                    severity="high",
                    description=(
                        "The device is accepting connections on the "
                        "Telnet port. Telnet usually sends management "
                        "traffic without encryption."
                    ),
                    recommendation=(
                        "Disable Telnet. Use an encrypted management "
                        "service and limit management access to the "
                        "administrator network."
                    ),
                    evidence="TCP port 23 was reachable.",
                )
            )

        # FTP exposure
        if 21 in open_ports:
            findings.append(
                self._create_finding(
                    finding_code="CYBER-FTP-001",
                    title="FTP service is exposed",
                    severity="medium",
                    description=(
                        "An FTP service is reachable on the device. "
                        "Standard FTP does not encrypt usernames, "
                        "passwords, or transferred information."
                    ),
                    recommendation=(
                        "Disable FTP if it is not required. If file "
                        "transfer is needed, use a protected option "
                        "supported by the device."
                    ),
                    evidence="TCP port 21 was reachable.",
                )
            )

        # HTTP present without HTTPS
        http_open = 80 in open_ports or 8080 in open_ports
        https_open = 443 in open_ports or 8443 in open_ports

        if http_open and not https_open:
            findings.append(
                self._create_finding(
                    finding_code="CYBER-HTTP-001",
                    title="Unencrypted web interface detected",
                    severity="medium",
                    description=(
                        "The device exposes an HTTP interface, but an "
                        "HTTPS interface was not detected on the "
                        "selected ports."
                    ),
                    recommendation=(
                        "Enable HTTPS if the device supports it. "
                        "Disable plain HTTP where possible and limit "
                        "the administration interface to a protected "
                        "management network."
                    ),
                    evidence=(
                        "An HTTP port was reachable while the selected "
                        "HTTPS ports were not reachable."
                    ),
                )
            )

        # RTSP exposure
        if 554 in open_ports:
            findings.append(
                self._create_finding(
                    finding_code="CYBER-RTSP-001",
                    title="RTSP streaming service is reachable",
                    severity="low",
                    description=(
                        "The device exposes an RTSP streaming service. "
                        "RTSP may be required for normal camera use, "
                        "but it should not be available to untrusted "
                        "networks."
                    ),
                    recommendation=(
                        "Require authentication and restrict RTSP "
                        "access to approved monitoring systems or the "
                        "camera network."
                    ),
                    evidence="TCP port 554 was reachable.",
                )
            )

        # Information disclosed through HTTP Server header
        for result in http_results:
            if result.reachable and result.server_header:
                findings.append(
                    self._create_finding(
                        finding_code="CYBER-HEADER-001",
                        title="Web server information is disclosed",
                        severity="low",
                        description=(
                            "The web interface returns a Server header. "
                            "This information may reveal the software "
                            "used by the device."
                        ),
                        recommendation=(
                            "Reduce unnecessary software or version "
                            "information in HTTP response headers if "
                            "the device supports this setting."
                        ),
                        evidence=(
                            f"Port {result.port} returned the header "
                            f"'Server: {result.server_header}'."
                        ),
                    )
                )

        # Several management-related services exposed
        management_ports = {
            21,
            22,
            23,
            80,
            443,
            8000,
            8080,
            8443,
        }

        exposed_management_ports = sorted(
            open_ports.intersection(management_ports)
        )

        if len(exposed_management_ports) >= 4:
            port_text = ", ".join(
                str(port) for port in exposed_management_ports
            )

            findings.append(
                self._create_finding(
                    finding_code="CYBER-SERVICES-001",
                    title="Several management services are exposed",
                    severity="medium",
                    description=(
                        "The device exposes several management or "
                        "file-transfer services. Every unnecessary "
                        "service increases the possible attack surface."
                    ),
                    recommendation=(
                        "Review all enabled services and disable the "
                        "services that are not needed for camera "
                        "operation."
                    ),
                    evidence=(
                        f"Reachable management ports: {port_text}"
                    ),
                )
            )

        # No current rule matched
        if not findings:
            findings.append(
                self._create_finding(
                    finding_code="CYBER-INFO-001",
                    title="No selected exposure rule was triggered",
                    severity="informational",
                    description=(
                        "The limited initial scan did not trigger any "
                        "of the current rule-based findings."
                    ),
                    recommendation=(
                        "Continue regular authorized assessment and "
                        "verify the device configuration manually."
                    ),
                    evidence=(
                        "No current risk rule matched the scan output."
                    ),
                )
            )

        return findings

    @staticmethod
    def calculate_risk_score(
        findings: List[SecurityFinding],
    ) -> int:
        """
        Calculate the total risk score.
        """
        return sum(finding.points for finding in findings)

    @staticmethod
    def calculate_risk_level(risk_score: int) -> str:
        """
        Convert the numerical risk score into a risk level.
        """
        if risk_score >= 16:
            return "critical"

        if risk_score >= 9:
            return "high"

        if risk_score >= 4:
            return "medium"

        if risk_score >= 1:
            return "low"

        return "clear"

    @staticmethod
    def _create_finding(
        finding_code: str,
        title: str,
        severity: str,
        description: str,
        recommendation: str,
        evidence: str,
    ) -> SecurityFinding:
        """
        Create one SecurityFinding object.
        """
        return SecurityFinding(
            finding_code=finding_code,
            title=title,
            severity=severity,
            description=description,
            recommendation=recommendation,
            evidence=evidence,
            points=RISK_POINTS[severity],
        )