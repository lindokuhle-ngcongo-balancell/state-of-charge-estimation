"""MQTT listener data source for real-time battery data."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

import aiomqtt

from soc_estimation.data_sources.base import BatteryDataRecord, DataSource


class MQTTDataSource(DataSource):
    """Listen to MQTT messages for real-time battery data."""

    def __init__(
        self,
        broker: str,
        gateway_ids: list[str],
        bem_ids: list[int],
        num_cells: int,
        port: int = 1883,
        use_tls: bool = False,
        username: str | None = None,
        password: str | None = None,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize MQTT data source.

        Args:
            broker: MQTT broker hostname
            gateway_ids: List of gateway IDs (e.g., ["gw-000027", "gw-000105"])
            bem_ids: List of BEM IDs (e.g., [1, 2, 3, 4])
            num_cells: Number of cells per BEM
            port: MQTT port (default 1883)
            use_tls: Enable TLS (default False)
            username: MQTT username (optional)
            password: MQTT password (optional)
            logger: Optional logger instance
        """
        self.broker = broker
        self.gateway_ids = gateway_ids
        self.bem_ids = bem_ids
        self.num_cells = num_cells
        self.port = port
        self.use_tls = use_tls
        self.username = username
        self.password = password
        self.logger = logger or logging.getLogger(__name__)

    async def get_records_async(self) -> AsyncIterator[BatteryDataRecord]:
        """
        Listen to MQTT and yield battery records as they arrive.

        Yields:
            BatteryDataRecord for each MQTT message received
        """
        self.logger.info(
            "Connecting to MQTT broker %s:%d (gateways: %s)",
            self.broker,
            self.port,
            ", ".join(self.gateway_ids),
        )

        async with aiomqtt.Client(
            self.broker,
            port=self.port,
            tls=self.use_tls,
            username=self.username,
            password=self.password,
        ) as client:
            # Subscribe to BEM datagram topics for all gateway/BEM combinations
            topics = []
            for gateway_id in self.gateway_ids:
                for bem_id in self.bem_ids:
                    topic = f"{gateway_id}/local/bem/response/{bem_id:02d}/getLatestDatagram"
                    topics.append(topic)
                    await client.subscribe(topic)
                    self.logger.info("Subscribed to %s", topic)

            self.logger.info("Listening for messages on %d topics", len(topics))

            # Process incoming messages
            async for message in client.messages:
                try:
                    payload = json.loads(message.payload.decode("utf-8"))
                    record = self._parse_bem_datagram(payload)
                    if record:
                        yield record
                except Exception as e:
                    self.logger.warning(f"Failed to parse MQTT message: {e}")
                    continue

    def _parse_bem_datagram(self, payload: dict) -> BatteryDataRecord | None:
        """
        Parse BEM datagram from MQTT payload.

        Expects payload with keys like:
            - bemCurrentInst (mA)
            - extTemp (°C)
            - cellVoltage_1, cellVoltage_2, ... (V)
            - soc_nominal (%)
            - sequence_number, timestamp

        Args:
            payload: MQTT message payload (dict)

        Returns:
            BatteryDataRecord or None if parsing fails
        """
        try:
            # Extract cell voltages
            cell_voltages = {}
            for i in range(1, self.num_cells + 1):
                key = f"cellVoltage_{i}"
                if key in payload:
                    cell_voltages[i] = float(payload[key])

            # Convert current from mA to A
            current_ma = payload.get("bemCurrentInst", 0)
            current_a = current_ma / 1000.0 if current_ma is not None else None

            record = BatteryDataRecord(
                sequence_number=int(payload.get("sequence_number", 0)),
                time_stamp=str(payload.get("timestamp", "")),
                temperature_battery=float(payload.get("extTemp", 25.0)),
                current=current_a,
                cell_voltages=cell_voltages,
                charge_in=float(payload.get("charge_in", 0)),
                charge_out=float(payload.get("charge_out", 0)),
                soc_nominal=float(payload.get("soc_nominal", 0)),
            )
            return record
        except (KeyError, ValueError, TypeError) as e:
            self.logger.debug(f"Could not parse BEM datagram: {e}")
            return None

    def get_records(self):
        """
        Synchronous wrapper using asyncio.run().

        Note: This blocks until MQTT connection closes. For real-time use,
        prefer get_records_async() with asyncio.
        """
        self.logger.warning(
            "Using synchronous MQTT interface; consider using "
            "get_records_async() for true async operation"
        )
        return asyncio.run(self.get_records_async())
