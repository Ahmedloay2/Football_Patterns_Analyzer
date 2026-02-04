"""
Configuration settings for the Football Tactical Pattern Analysis system.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AnalysisConfig:
    """Configuration for play analysis and clustering."""
    
    min_play_duration: float = 3.0
    max_play_duration: float = 30.0
    min_forward_progress: float = 5.0
    clustering_threshold: float = 12.0
    verbose: bool = True
    
    # Thresholds
    ahead_threshold: float = 1.0  # meters
    forward_threshold: float = 1.0  # meters
    
    # Field position constants (pitch divided into thirds)
    # Assuming pitch is -50 to +50 in x-axis (100 meters total)
    defensive_third_max: float = -16.67  # -50 to -16.67
    middle_third_max: float = 16.67      # -16.67 to 16.67
    attacking_third_min: float = 16.67   # 16.67 to 50
    deep_attacking_min: float = 20.0     # Minimum for deep attacking play


@dataclass
class PathConfig:
    """Configuration for file paths."""
    
    def __init__(self, base_dir: Path = None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        
        self.base_dir = base_dir
        self.data_dir = base_dir / "Event Data"
        self.output_dir = base_dir / "output"
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)
    
    @property
    def cluster_csv(self) -> Path:
        """Path to cluster analysis CSV file."""
        return self.output_dir / "cluster_analysis.csv"


# Global configuration instances
analysis_config = AnalysisConfig()
path_config = PathConfig()
