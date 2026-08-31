"""
Entry point for GARUD Workstream B.

Usage:

    python -m cybersecurity.cyber_main --list

    python -m cybersecurity.cyber_main --device CAM-LAB-001
"""

import argparse
import datetime
import uuid

from cybersecurity.analysis.risk_engine import RiskEngine
from cybersecurity.inventory.target_manager import TargetManager
from cybersecurity.models import ScanReport
from cybersecurity.reporting.report_generator import ReportGenerator
from cybersecurity.scanners.http_checker import HTTPChecker
from cybersecurity.scanners.port_scanner import PortScanner


def list_authorized_devices(
    target_manager: TargetManager,
) -> None:
    print("\nAuthorized GARUD cybersecurity targets")
    print("-" * 65)

    for target in target_manager.list_targets():
        print(
            f"{target.device_id:<16} "
            f"{target.ip_address:<16} "
            f"{target.device_type:<12} "
            f"{target.name}"
        )


def scan_device(device_id: str) -> ScanReport:
    target_manager = TargetManager()

    # The target must exist in the local authorized inventory.
    target = target_manager.get_target(device_id)

    started_at = datetime.datetime.now().isoformat()

    print(
        f"\nStarting authorized assessment for "
        f"{target.device_id} ({target.ip_address})"
    )

    port_scanner = PortScanner()
    port_results = port_scanner.scan_target(target)

    http_checker = HTTPChecker()
    http_results = http_checker.inspect_open_web_ports(
        target=target,
        port_results=port_results,
    )

    risk_engine = RiskEngine()
    findings = risk_engine.generate_findings(
        port_results=port_results,
        http_results=http_results,
    )

    risk_score = risk_engine.calculate_risk_score(findings)
    risk_level = risk_engine.calculate_risk_level(risk_score)

    completed_at = datetime.datetime.now().isoformat()

    return ScanReport(
        scan_id=(
            f"CYBER-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            f"-{uuid.uuid4().hex[:6].upper()}"
        ),
        device={
            "device_id": target.device_id,
            "name": target.name,
            "ip_address": target.ip_address,
            "device_type": target.device_type,
            "location": target.location,
        },
        started_at=started_at,
        completed_at=completed_at,
        status="completed",
        port_results=[
            result.to_dict()
            for result in port_results
        ],
        http_results=[
            result.to_dict()
            for result in http_results
        ],
        findings=[
            finding.to_dict()
            for finding in findings
        ],
        risk_score=risk_score,
        risk_level=risk_level,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "GARUD authorized CCTV and DVR security assessment"
        )
    )

    action_group = parser.add_mutually_exclusive_group(
        required=True
    )

    action_group.add_argument(
        "--list",
        action="store_true",
        help="List authorized laboratory devices.",
    )

    action_group.add_argument(
        "--device",
        help=(
            "Scan one device by its approved inventory device ID."
        ),
    )

    args = parser.parse_args()

    target_manager = TargetManager()

    if args.list:
        list_authorized_devices(target_manager)
        return

    try:
        report = scan_device(args.device)

        report_generator = ReportGenerator()
        report_generator.print_summary(report)

        report_path = report_generator.save_json(report)

        print(f"\nReport saved to: {report_path}")

    except (KeyError, ValueError, PermissionError) as error:
        print(f"\nAssessment blocked: {error}")

    except KeyboardInterrupt:
        print("\nAssessment cancelled by the user.")


if __name__ == "__main__":
    main()