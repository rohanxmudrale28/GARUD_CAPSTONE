"""
Limited TCP port scanner for authorized CCTV and DVR devices.

This scanner only checks whether selected TCP ports are reachable.
It does not attempt login, exploitation, or device modification.
"""

import socket

from cybersecurity.config import PORT_TIMEOUT, TCP_PORTS
from cybersecurity.models import PortResult


class PortScanner:
    def __init__(self, ports=None, timeout=PORT_TIMEOUT):
        """
        Initialize the scanner with selected CCTV-related ports.
        """
        if ports is None:
            self.ports = TCP_PORTS
        else:
            self.ports = ports

        self.timeout = timeout

    def check_port(self, ip_address, port, service):
        """
        Check whether one TCP port is reachable.
        """
        try:
            connection = socket.create_connection(
                (ip_address, port),
                timeout=self.timeout,
            )

            connection.close()
            is_open = True

        except (TimeoutError, ConnectionRefusedError, OSError):
            is_open = False

        return PortResult(
            port=port,
            service=service,
            is_open=is_open,
        )

    def scan_target(self, target):
        """
        Check selected ports on one authorized target.
        """
        if not target.authorized:
            raise PermissionError(
                f"Scanning is not authorized for {target.device_id}."
            )

        results = []

        for port, service in self.ports.items():
            result = self.check_port(
                ip_address=target.ip_address,
                port=port,
                service=service,
            )

            results.append(result)

        return results