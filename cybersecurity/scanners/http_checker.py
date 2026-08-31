"""
Safe HTTP and HTTPS checker for GARUD.

This checker sends only a HEAD request to an authorized device.
It does not attempt passwords, submit forms, or bypass authentication.
"""

import http.client
import ssl

from cybersecurity.config import HTTP_TIMEOUT
from cybersecurity.models import HTTPResult


class HTTPChecker:
    WEB_PORTS = {
        80: "http",
        443: "https",
        8080: "http",
        8443: "https",
    }

    def __init__(self, timeout=HTTP_TIMEOUT):
        self.timeout = timeout

    def check(self, target, port, scheme):
        """
        Send one safe HEAD request to an authorized web interface.
        """
        if not target.authorized:
            raise PermissionError(
                f"HTTP checking is not authorized for "
                f"{target.device_id}."
            )

        connection = None

        try:
            if scheme == "https":
                context = ssl.create_default_context()

                # Lab cameras may use self-signed certificates.
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE

                connection = http.client.HTTPSConnection(
                    host=target.ip_address,
                    port=port,
                    timeout=self.timeout,
                    context=context,
                )

            else:
                connection = http.client.HTTPConnection(
                    host=target.ip_address,
                    port=port,
                    timeout=self.timeout,
                )

            connection.request(
                method="HEAD",
                url="/",
                headers={
                    "User-Agent": (
                        "GARUD-Authorized-Security-Assessment/1.0"
                    )
                },
            )

            response = connection.getresponse()

            return HTTPResult(
                port=port,
                scheme=scheme,
                reachable=True,
                status_code=response.status,
                server_header=response.getheader(
                    "Server",
                    "",
                ),
                authentication_header=response.getheader(
                    "WWW-Authenticate",
                    "",
                ),
                redirect_location=response.getheader(
                    "Location",
                    "",
                ),
                error="",
            )

        except Exception as error:
            return HTTPResult(
                port=port,
                scheme=scheme,
                reachable=False,
                status_code=None,
                server_header="",
                authentication_header="",
                redirect_location="",
                error=str(error),
            )

        finally:
            if connection is not None:
                connection.close()

    def inspect_open_web_ports(self, target, port_results):
        """
        Check only the web ports reported as open by PortScanner.
        """
        results = []

        open_ports = {
            result.port
            for result in port_results
            if result.is_open
        }

        for port, scheme in self.WEB_PORTS.items():
            if port not in open_ports:
                continue

            result = self.check(
                target=target,
                port=port,
                scheme=scheme,
            )

            results.append(result)

        return results