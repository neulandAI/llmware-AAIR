from dataclasses import dataclass, field
from typing import Dict, Any
from datetime import datetime


@dataclass
class ExperimentResult:
    """Result from a single experiment run."""
    
    config: Dict[str, Any]
    metrics: Dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "config": self.config,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ExperimentResult":
        """Create from dictionary."""
        return cls(
            config=data.get("config", {}),
            metrics=data.get("metrics", {}),
            timestamp=data.get("timestamp", ""),
            metadata=data.get("metadata", {}),
        )
