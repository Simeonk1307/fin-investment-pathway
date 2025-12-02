import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any

class BronzeFormatter:
    """
    Standardizes raw data into the Bronze Layer 'Envelope' schema.
    """
    
    @staticmethod
    def format(data: Dict[str, Any], source_type: str, ticker: str, timestamp_ms: int) -> Dict[str, Any]:
        """
        Wraps raw data into the Bronze Schema.
        
        Args:
            data (dict): The actual data (e.g., filing info, stock price).
            source_type (str): 'filings', 'stocks', or 'news'.
            ticker (str): The main identifier (e.g., 'AAPL').
            timestamp_ms (int): The event time in milliseconds.
            
        Returns:
            dict: The standardized bronze record ready for Redpanda.
        """
        
        # 1. Generate IDs and Timestamps
        event_id = str(uuid.uuid4())
        
        # Current time (Ingest Time) - When we processed it
        ingest_ts = datetime.now(timezone.utc).isoformat()
        
        # Event time - When it actually happened (derived from data)
        event_ts = datetime.fromtimestamp(
            timestamp_ms / 1000.0, timezone.utc
        ).isoformat()

        # 2. Construct the Bronze Envelope
        return {
            "event_id": event_id,
            "source_type": source_type,
            "ticker": ticker,
            "event_ts": event_ts,
            "ingest_ts": ingest_ts,
            
            # The Requirement: raw_payload must be a STRING representation of the JSON
            "raw_payload": json.dumps(data), 
            
            "schema_version": 1
        }