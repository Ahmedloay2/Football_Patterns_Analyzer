"""
Data models for football tactical analysis.
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MatchMetadata:
    """Metadata for a single match."""
    match_id: int
    match_name: str
    home_team: str
    away_team: str
    file_path: str


@dataclass
class PlayEvent:
    """Single event within a play."""
    event_type: str
    time: float  # in seconds
    ball_x: float
    ball_y: float
    attacking_players_ahead: int
    defending_players_ahead: int
    team_id: int
    player_name: Optional[str] = None


@dataclass
class Play:
    """
    Complete attacking play with metadata.
    Clustering focuses on PATTERN (ball movement, passes, positions), not outcome.
    """
    # Identifiers
    play_id: str
    match_id: int
    match_name: str
    team_id: int
    team_name: str
    
    # Timing (stored in seconds)
    start_time: float
    end_time: float
    duration: float
    
    # Events and structure
    events: List[PlayEvent]
    normalized_events: List[PlayEvent]
    num_events: int
    
    # Outcome (NOT used in clustering, only for display)
    outcome: str
    is_goal: bool
    
    # Spatial features (used for clustering)
    delta_x: float
    delta_y: float
    max_x: float
    total_distance: float
    
    # Tactical features (used for clustering)
    avg_attackers_ahead: float
    avg_defenders_ahead: float
    wing_side: str
    
    # Video reference
    video_url: str
    start_game_clock: int
    end_game_clock: int
    
    # Cluster assignment
    cluster_id: Optional[int] = None
    cluster_name: Optional[str] = None
    
    # Original attack direction
    original_attack_direction: Optional[str] = None
    
    @property
    def start_time_display(self) -> str:
        """Display start time in MM:SS format."""
        return self._seconds_to_mmss(self.start_game_clock)
    
    @property
    def end_time_display(self) -> str:
        """Display end time in MM:SS format."""
        return self._seconds_to_mmss(self.end_game_clock)
    
    @property
    def time_range_display(self) -> str:
        """Display time range as 'MM:SS - MM:SS'."""
        return f"{self.start_time_display} - {self.end_time_display}"
    
    @staticmethod
    def _seconds_to_mmss(seconds: float) -> str:
        """Convert seconds to MM:SS format."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
