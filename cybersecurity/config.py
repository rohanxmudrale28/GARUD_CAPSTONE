"""
Configuration values for GARUD's CCTV/DVR security assessment.

The scanner is intentionally limited to selected CCTV-related services.
It should only be used on devices owned by the team or devices for
which written authorization has been provided.
"""

TCP_PORTS = {
    21: "FTP",
    22: "SSH",
    23: "Telnet",
    80: "HTTP",
    443: "HTTPS",
    554: "RTSP",
    8000: "Vendor Management Service",
    8080: "Alternative HTTP",
    8443: "Alternative HTTPS",
}

PORT_TIMEOUT = 1.0
HTTP_TIMEOUT = 3.0

ALLOWED_DEVICE_TYPES = {
    "ip_camera",
    "dvr",
    "nvr",
    "test_host",
}

RISK_POINTS = {
    "informational": 0,
    "low": 1,
    "medium": 3,
    "high": 6,
    "critical": 10,
}