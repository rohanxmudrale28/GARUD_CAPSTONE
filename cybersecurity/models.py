"""
Common data structures used by the GARUD cybersecurity workstream.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class TargetDevice:
    device_id: str
    name: str
    ip_address: str
    device_type: str
    location: str
    authorized: bool


@dataclass
class PortResult:
    port: int
    service: str
    is_open: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HTTPResult:
    port: int
    scheme: str
    reachable: bool
    status_code: int | None = None
    server_header: str = ""
    authentication_header: str = ""
    redirect_location: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SecurityFinding:
    finding_code: str
    title: str
    severity: str
    description: str
    recommendation: str
    evidence: str
    points: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScanReport:
    scan_id: str
    device: Dict[str, Any]
    started_at: str
    completed_at: str
    status: str
    port_results: List[Dict[str, Any]] = field(default_factory=list)
    http_results: List[Dict[str, Any]] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    risk_score: int = 0
    risk_level: str = "clear"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)