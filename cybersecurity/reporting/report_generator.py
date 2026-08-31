"""
JSON and console reporting for GARUD cybersecurity scans.
"""

import json
from pathlib import Path

from cybersecurity.models import ScanReport


BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = BASE_DIR / "reports"


class ReportGenerator:
    def __init__(self, report_dir: Path = REPORT_DIR):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, report: ScanReport) -> Path:
        safe_scan_id = report.scan_id.replace(":", "-")
        output_path = self.report_dir / f"{safe_scan_id}.json"

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(
                report.to_dict(),
                file,
                indent=2,
            )

        return output_path

    @staticmethod
    def print_summary(report: ScanReport) -> None:
        device = report.device

        print("\n" + "=" * 60)
        print("GARUD CCTV/DVR SECURITY ASSESSMENT")
        print("=" * 60)

        print(f"Scan ID      : {report.scan_id}")
        print(f"Device ID    : {device['device_id']}")
        print(f"Device Name  : {device['name']}")
        print(f"Device Type  : {device['device_type']}")
        print(f"IP Address   : {device['ip_address']}")
        print(f"Location     : {device['location']}")
        print(f"Authorization: Verified")
        print(f"Status       : {report.status}")

        print("\nSelected service check")
        print("-" * 60)

        for result in report.port_results:
            status = "OPEN" if result["is_open"] else "CLOSED"
            print(
                f"Port {result['port']:<5} "
                f"{result['service']:<30} {status}"
            )

        print("\nSecurity findings")
        print("-" * 60)

        for index, finding in enumerate(report.findings, start=1):
            print(
                f"{index}. [{finding['severity'].upper()}] "
                f"{finding['title']}"
            )
            print(f"   Evidence: {finding['evidence']}")
            print(
                f"   Recommendation: "
                f"{finding['recommendation']}"
            )

        print("\nFinal assessment")
        print("-" * 60)
        print(f"Risk score : {report.risk_score}")
        print(f"Risk level : {report.risk_level.upper()}")
        print("=" * 60)