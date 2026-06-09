"""Base classes for data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field


@dataclass
class BatteryDataRecord:
    """Single battery measurement record.

    Attributes:
        sequence_number: Row counter (monotonically increasing).
        time_stamp: ISO-format timestamp string.
        temperature_battery: Pack temperature in °C.
        current: Pack current in A (positive = charging, negative = discharging).
        cell_voltages: Mapping of cell index (1-based) to voltage in V.
        charge_in: Cumulative charge delivered to pack in Coulombs.
        charge_out: Cumulative charge drawn from pack in Coulombs.
        soc_nominal: BMS-reported SoC in % (reference only).
        delta_charge: Net charge change since previous tick in Coulombs
            (charge_in_delta - charge_out_delta).  None for the very first
            record where no previous tick exists.
    """

    sequence_number: int
    time_stamp: str
    temperature_battery: float
    current: float | None
    cell_voltages: dict[int, float]  # cell_idx → voltage_v
    charge_in: float
    charge_out: float
    soc_nominal: float
    delta_charge: float | None = field(default=None)


class DataSource(ABC):
    """Abstract base class for SOC data sources."""

    @abstractmethod
    def get_records(self) -> Iterator[BatteryDataRecord]:
        """
        Synchronously iterate over battery data records.

        Yields:
            BatteryDataRecord for each measurement
        """
        pass

    async def get_records_async(self) -> AsyncIterator[BatteryDataRecord]:
        """
        Asynchronously iterate over battery data records.

        Default implementation wraps synchronous get_records().
        Override in subclasses for true async sources (e.g., MQTT).

        Yields:
            BatteryDataRecord for each measurement
        """
        for record in self.get_records():
            yield record

    def __iter__(self) -> Iterator[BatteryDataRecord]:
        """Allow direct iteration: for record in data_source:"""
        return self.get_records()
