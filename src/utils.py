"""
Utility functions for football tactical analysis.
"""
import numpy as np
from typing import List
from .models import Play, PlayEvent


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def normalize_coordinates_to_right(events: List[PlayEvent], 
                                   attack_direction: str) -> List[PlayEvent]:
    """
    Normalize play coordinates so all plays attack from left to right.
    
    Args:
        events: List of PlayEvent objects
        attack_direction: 'L' or 'R' indicating original attack direction
    
    Returns:
        List of normalized PlayEvent objects
    """
    if attack_direction == 'R':
        return events
    
    # Flip coordinates for left-attacking plays
    normalized = []
    for event in events:
        normalized.append(PlayEvent(
            event_type=event.event_type,
            time=event.time,
            ball_x=-event.ball_x,
            ball_y=-event.ball_y,
            attacking_players_ahead=event.attacking_players_ahead,
            defending_players_ahead=event.defending_players_ahead,
            team_id=event.team_id,
            player_name=event.player_name
        ))
    
    return normalized


def get_feature_vector(play: Play) -> np.ndarray:
    """
    Extract feature vector for clustering.
    
    Focus on structural patterns:
    - Event sequence and types
    - Spatial progression
    - Tactical positioning
    
    Returns:
        Feature vector as numpy array
    """
    features = []
    
    # Event type sequence (one-hot encoded common types)
    event_types = ['PA', 'SH', 'CR', 'IT', 'LO', 'CA', 'DR', 'TC']
    event_counts = {et: 0 for et in event_types}
    
    for event in play.normalized_events:
        if event.event_type in event_counts:
            event_counts[event.event_type] += 1
    
    features.extend([event_counts[et] for et in event_types])
    
    # Spatial features
    features.extend([
        play.delta_x,
        play.delta_y,
        play.max_x,
        play.total_distance,
        play.num_events,
        play.duration
    ])
    
    # Tactical features
    features.extend([
        play.avg_attackers_ahead,
        play.avg_defenders_ahead,
        1.0 if play.wing_side == 'WING' else 0.0
    ])
    
    return np.array(features)


def determine_attack_direction(is_home: bool, period: int, 
                               stadium_meta: dict = None) -> str:
    """
    Determine team's attack direction.
    
    Args:
        is_home: Whether team is home team
        period: Match period (1 or 2)
        stadium_meta: Optional stadium metadata
    
    Returns:
        'L' or 'R' indicating attack direction
    """
    if stadium_meta and 'teamAttackingDirection' in stadium_meta:
        return stadium_meta['teamAttackingDirection']
    
    # Default: home attacks right in first half
    if period == 1:
        return 'R' if is_home else 'L'
    else:
        return 'L' if is_home else 'R'
