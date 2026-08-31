"""
Authorized target inventory manager for GARUD.

Only devices listed in authorized_targets.json and marked as
authorized can be checked by the cybersecurity scanner.
"""

import ipaddress
import json
from pathlib import Path

from cybersecurity.config import ALLOWED_DEVICE_TYPES
from cybersecurity.models import TargetDevice


# Path to the main cybersecurity folder.
BASE_DIR = Path(__file__).resolve().parents[1]

# Default path of the approved target inventory.
DEFAULT_INVENTORY_PATH = (
    BASE_DIR / "data" / "authorized_targets.json"
)


class TargetManager:
    """
    Loads and validates the approved CCTV and DVR targets.
    """

    def __init__(self, inventory_path=None):
        """
        Initialize the target manager.

        Args:
            inventory_path: Optional custom path to the target JSON file.
        """
        if inventory_path is None:
            self.inventory_path = DEFAULT_INVENTORY_PATH
        else:
            self.inventory_path = Path(inventory_path)

        self.targets = self._load_targets()

    def _load_targets(self):
        """
        Load all targets from the authorized target inventory.

        Returns:
            A list of validated TargetDevice objects.
        """
        if not self.inventory_path.exists():
            raise FileNotFoundError(
                f"Target inventory not found: "
                f"{self.inventory_path}"
            )

        try:
            with open(
                self.inventory_path,
                "r",
                encoding="utf-8",
            ) as file:
                raw_targets = json.load(file)

        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in target inventory: {error}"
            ) from error

        if not isinstance(raw_targets, list):
            raise ValueError(
                "The authorized target inventory must contain "
                "a JSON list."
            )

        targets = []
        device_ids = set()

        for raw_target in raw_targets:
            if not isinstance(raw_target, dict):
                raise ValueError(
                    "Every entry in the target inventory must "
                    "be a JSON object."
                )

            target = TargetDevice(
                device_id=str(
                    raw_target.get("device_id", "")
                ).strip(),
                name=str(
                    raw_target.get("name", "")
                ).strip(),
                ip_address=str(
                    raw_target.get("ip_address", "")
                ).strip(),
                device_type=str(
                    raw_target.get("device_type", "")
                ).strip().lower(),
                location=str(
                    raw_target.get("location", "")
                ).strip(),
                authorized=bool(
                    raw_target.get("authorized", False)
                ),
            )

            self._validate_target(target)

            normalized_device_id = target.device_id.lower()

            if normalized_device_id in device_ids:
                raise ValueError(
                    f"Duplicate device ID found: "
                    f"{target.device_id}"
                )

            device_ids.add(normalized_device_id)
            targets.append(target)

        return targets

    def _validate_target(self, target):
        """
        Check whether one target contains valid information.

        The target must:
        - Have a device ID and name
        - Have a supported device type
        - Use a private IP address
        - Be marked as authorized
        """

        if not target.device_id:
            raise ValueError(
                "Every target must have a device_id."
            )

        if not target.name:
            raise ValueError(
                f"Target {target.device_id} must have a name."
            )

        if not target.ip_address:
            raise ValueError(
                f"Target {target.device_id} must have an "
                "IP address."
            )

        if not target.location:
            raise ValueError(
                f"Target {target.device_id} must have a location."
            )

        if target.device_type not in ALLOWED_DEVICE_TYPES:
            allowed_types = ", ".join(
                sorted(ALLOWED_DEVICE_TYPES)
            )

            raise ValueError(
                f"Unsupported device type for "
                f"{target.device_id}: "
                f"{target.device_type}. "
                f"Allowed types: {allowed_types}"
            )

        try:
            ip_object = ipaddress.ip_address(
                target.ip_address
            )

        except ValueError as error:
            raise ValueError(
                f"Invalid IP address for "
                f"{target.device_id}: "
                f"{target.ip_address}"
            ) from error

        if not ip_object.is_private:
            raise ValueError(
                f"Target {target.device_id} must use a "
                "private laboratory IP address. "
                f"Public IP addresses are not allowed."
            )

        if ip_object.is_loopback:
            raise ValueError(
                f"Target {target.device_id} uses a loopback "
                "address. Add the device's private LAN IP address."
            )

        if ip_object.is_multicast:
            raise ValueError(
                f"Target {target.device_id} uses a multicast "
                "address, which is not allowed as a scan target."
            )

        if ip_object.is_unspecified:
            raise ValueError(
                f"Target {target.device_id} uses an unspecified "
                "IP address."
            )

        if not target.authorized:
            raise PermissionError(
                f"Target {target.device_id} is not marked "
                "as authorized."
            )

    def get_target(self, device_id):
        """
        Find and return a device using its device ID.

        Args:
            device_id: Device ID from authorized_targets.json.

        Returns:
            The matching TargetDevice object.

        Raises:
            KeyError: If the requested device is not present.
        """
        if device_id is None:
            raise ValueError(
                "A device ID must be provided."
            )

        requested_id = str(device_id).strip().lower()

        if not requested_id:
            raise ValueError(
                "The device ID cannot be empty."
            )

        for target in self.targets:
            if target.device_id.lower() == requested_id:
                return target

        raise KeyError(
            f"Device '{device_id}' is not present in "
            "the authorized inventory."
        )

    def list_targets(self):
        """
        Return all authorized devices.

        Returns:
            A copy of the validated target list.
        """
        return list(self.targets)

    def target_exists(self, device_id):
        """
        Check whether a device ID exists in the inventory.

        Returns:
            True if the device exists, otherwise False.
        """
        if device_id is None:
            return False

        requested_id = str(device_id).strip().lower()

        for target in self.targets:
            if target.device_id.lower() == requested_id:
                return True

        return False