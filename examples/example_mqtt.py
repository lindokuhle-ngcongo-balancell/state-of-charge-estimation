"""Example: Real-time MQTT-based SOC estimation."""

import asyncio
from soc_estimation import (
    MQTTDataSource,
    SOCEstimationEngine,
    setup_logger,
)

# Setup
logger = setup_logger("example_mqtt", "example_mqtt.log")

# Configuration
BROKER = "mqtt.example.com"
PORT = 1883
GATEWAYS = ["gw-000027", "gw-000105"]
BEM_IDS = [1, 2, 3, 4]
LUT_FILE = "data/csv_lut/lut_ocv.csv"
CAPACITY_AH = 277
NUM_CELLS = 32

async def main():
    """Main async function for real-time SOC estimation."""
    print(f"Connecting to MQTT broker {BROKER}:{PORT}...")
    
    # Create data source
    data_source = MQTTDataSource(
        broker=BROKER,
        gateway_ids=GATEWAYS,
        bem_ids=BEM_IDS,
        num_cells=NUM_CELLS,
        port=PORT,
        logger=logger,
    )
    
    # Create engine
    print(f"Initializing SOC engine...")
    engine = SOCEstimationEngine(
        data_source=data_source,
        num_cells=NUM_CELLS,
        capacity_ah=CAPACITY_AH,
        ocv_lut_path=LUT_FILE,
        logger=logger,
    )
    
    # Listen to MQTT messages
    print(f"Listening for MQTT messages from {len(GATEWAYS)} gateways...")
    print("Processing real-time SOC estimation (press Ctrl+C to stop)")
    
    try:
        async for record in data_source.get_records_async():
            print(f"Received record: seq={record.sequence_number}, "
                  f"cells={len(record.cell_voltages)}, "
                  f"current={record.current}A")
            # In a real application, you would integrate this with
            # an event streaming system or database
    except KeyboardInterrupt:
        print("Stopping...")

if __name__ == "__main__":
    asyncio.run(main())
