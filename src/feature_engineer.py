"""
Feature engineering for play analysis.
Calculates spatial and tactical features.
"""
import numpy as np
from typing import List

from .models import Play, PlayEvent
from .utils import normalize_coordinates_to_right, calculate_distance
from .config import analysis_config


class FeatureEngineer:
    """Calculates features for plays."""
    
    def __init__(self, config=None):
        self.config = config or analysis_config
    
    def engineer_features(self, plays: List[Play]) -> List[Play]:
        """
        Calculate all features for a list of plays.
        
        Args:
            plays: List of Play objects
        
        Returns:
            Same list with features calculated
        """
        for play in plays:
            self._process_play(play)
        
        return plays
    
    def _process_play(self, play: Play) -> None:
        """Process a single play to calculate all features."""
        # Normalize coordinates
        play.normalized_events = normalize_coordinates_to_right(
            play.events, 
            play.original_attack_direction
        )
        
        events = play.normalized_events
        
        # Spatial features
        play.delta_x = self._calculate_delta_x(events)
        play.delta_y = self._calculate_delta_y(events)
        play.max_x = self._calculate_max_x(events)
        play.total_distance = self._calculate_total_distance(events)
        
        # Tactical features
        play.avg_attackers_ahead = np.mean([e.attacking_players_ahead for e in events])
        play.avg_defenders_ahead = np.mean([e.defending_players_ahead for e in events])
        play.wing_side = self._identify_wing_side(events)
    
    def _calculate_delta_x(self, events: List[PlayEvent]) -> float:
        """Calculate forward progress."""
        if not events:
            return 0.0
        return events[-1].ball_x - events[0].ball_x
    
    def _calculate_delta_y(self, events: List[PlayEvent]) -> float:
        """Calculate lateral movement."""
        if not events:
            return 0.0
        return abs(events[-1].ball_y - events[0].ball_y)
    
    def _calculate_max_x(self, events: List[PlayEvent]) -> float:
        """Calculate maximum forward penetration."""
        if not events:
            return 0.0
        return max(e.ball_x for e in events)
    
    def _calculate_total_distance(self, events: List[PlayEvent]) -> float:
        """Calculate total ball movement distance."""
        if len(events) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(events) - 1):
            total += calculate_distance(
                events[i].ball_x, events[i].ball_y,
                events[i + 1].ball_x, events[i + 1].ball_y
            )
        return total
    
    def _identify_wing_side(self, events: List[PlayEvent]) -> str:
        """Identify if play occurred on wing or center."""
        if not events:
            return 'CENTER'
        
        avg_y = np.mean([abs(e.ball_y) for e in events])
        return 'WING' if avg_y > 15 else 'CENTER'
