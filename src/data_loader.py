"""
Data loading and parsing for football tactical analysis.
Handles JSON event data and extracts plays.
"""
import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from .models import MatchMetadata, PlayEvent, Play
from .config import analysis_config
from .utils import determine_attack_direction


class EventParser:
    """Parses raw event data into structured PlayEvent objects."""
    
    def __init__(self, config=None):
        self.config = config or analysis_config
    
    def parse_event(self, event: Dict, stadium_meta: Dict, 
                   period: int, is_home: bool,
                   previous_target_player: Optional[str] = None) -> Optional[PlayEvent]:
        """
        Parse a single event into a PlayEvent object.
        
        Args:
            event: Raw event dictionary
            stadium_meta: Stadium metadata
            period: Match period
            is_home: Whether team is home
            previous_target_player: Target player from previous event (for IT events)
        
        Returns:
            PlayEvent object or None if invalid
        """
        poss_event = event.get('possessionEvents', {})
        event_type = poss_event.get('possessionEventType')
        
        ball_data = event.get('ball', [])
        if not ball_data:
            return None
        
        ball_x = ball_data[0].get('x', 0)
        ball_y = ball_data[0].get('y', 0)
        
        # Get team info
        game_event = event.get('gameEvents', {})
        team_id = game_event.get('teamId')
        
        # Count players ahead
        attackers_ahead, defenders_ahead = self._count_players_ahead(
            event, ball_x, is_home, period, stadium_meta
        )
        
        # Extract player name
        player_name = self._extract_player_name(event_type, poss_event, previous_target_player)
        
        return PlayEvent(
            event_type=event_type,
            time=event.get('eventTime', 0),
            ball_x=ball_x,
            ball_y=ball_y,
            attacking_players_ahead=attackers_ahead,
            defending_players_ahead=defenders_ahead,
            team_id=team_id,
            player_name=player_name
        )
    
    def _count_players_ahead(self, event: Dict, ball_x: float, 
                            is_home: bool, period: int, 
                            stadium_meta: Dict) -> tuple:
        """Count attacking and defending players ahead of ball."""
        home_players = event.get('homePlayers', [])
        away_players = event.get('awayPlayers', [])
        
        if is_home:
            attacking_players = home_players
            defending_players = away_players
        else:
            attacking_players = away_players
            defending_players = home_players
        
        attack_dir = determine_attack_direction(is_home, period, stadium_meta)
        threshold = self.config.ahead_threshold
        
        attackers_ahead = sum(
            1 for p in attacking_players
            if (attack_dir == 'R' and p.get('x', 0) > ball_x + threshold) or
               (attack_dir == 'L' and p.get('x', 0) < ball_x - threshold)
        )
        
        defenders_ahead = sum(
            1 for p in defending_players
            if (attack_dir == 'R' and p.get('x', 0) > ball_x + threshold) or
               (attack_dir == 'L' and p.get('x', 0) < ball_x - threshold)
        )
        
        return attackers_ahead, defenders_ahead
    
    def _extract_player_name(self, event_type: str, poss_event: Dict,
                            previous_target_player: Optional[str]) -> Optional[str]:
        """Extract player name based on event type."""
        player_name = None
        
        # Event-specific player fields
        player_field_map = {
            'PA': 'passerPlayerName',
            'SH': 'shooterPlayerName',
            'CA': 'clearerPlayerName',
            'TA': 'challengerPlayerName',
            'CR': 'crosserPlayerName',
            'LO': 'loserPlayerName',
            'RE': 'receiverPlayerName',
            'IT': 'touchPlayerName',
            'DR': 'dribblerPlayerName',
            'TC': 'touchPlayerName',
            'CH': 'challengerPlayerName'
        }
        
        if event_type in player_field_map:
            player_name = poss_event.get(player_field_map[event_type])
        
        # Special handling for IT events - use previous target player
        if event_type == 'IT' and not player_name and previous_target_player:
            player_name = previous_target_player
        
        # Fallback to common fields
        if not player_name:
            player_name = (poss_event.get('carrierPlayerName') or
                          poss_event.get('touchPlayerName') or
                          poss_event.get('receiverPlayerName') or
                          poss_event.get('targetPlayerName'))
        
        return player_name


class PlayExtractor:
    """Extracts plays from match events."""
    
    def __init__(self, config=None):
        self.config = config or analysis_config
        self.event_parser = EventParser(config)
    
    def extract_plays(self, events: List[Dict], metadata: MatchMetadata) -> List[Play]:
        """
        Extract attacking plays from match events.
        Play definition: 2+ passes (PA, CR) of any kind by same team,
        followed by a terminal event (possession lost, shot).
        
        Args:
            events: List of raw event dictionaries
            metadata: Match metadata
        
        Returns:
            List of Play objects
        """
        plays = []
        stadium_meta = self._get_stadium_metadata(events)
        
        i = 0
        while i < len(events):
            # Try to find a play starting from this position
            play_data = self._try_extract_play(events, i, metadata, stadium_meta)
            
            if play_data:
                play, next_idx = play_data
                plays.append(play)
                i = next_idx
            else:
                i += 1
        
        return plays
    
    def _try_extract_play(self, events: List[Dict], start_idx: int,
                         metadata: MatchMetadata, stadium_meta: Dict) -> Optional[tuple]:
        """
        Try to extract a play starting from the given index.
        
        Returns:
            Tuple of (Play, next_index) if successful, None otherwise
        """
        if start_idx >= len(events):
            return None
        
        first_event = events[start_idx]
        poss_event = first_event.get('possessionEvents', {})
        event_type = poss_event.get('possessionEventType')
        
        # Must start with a forward pass
        if event_type != 'PA':
            return None
        
        game_event = first_event.get('gameEvents', {})
        team_id = game_event.get('teamId')
        is_home = game_event.get('homeTeam', False)
        period = game_event.get('period', 1)
        
        # Check if first pass is forward
        is_forward = self._is_pass_forward(
            first_event, events, start_idx, stadium_meta, team_id, period, is_home
        )
        
        if not is_forward:
            return None
        
        # Collect all events for this play
        play_events = []
        pass_events = []  # Track pass events separately
        current_idx = start_idx
        last_target_player = poss_event.get('targetPlayerName')
        first_event_data = first_event
        last_event_data = first_event
        
        # Process first pass
        play_event = self.event_parser.parse_event(
            first_event, stadium_meta, period, is_home, None
        )
        if play_event:
            play_events.append(play_event)
            pass_events.append(first_event)
        
        current_idx += 1
        
        # Continue collecting events from the same team
        while current_idx < len(events):
            event = events[current_idx]
            event_poss = event.get('possessionEvents', {})
            event_type = event_poss.get('possessionEventType')
            event_game = event.get('gameEvents', {})
            event_team = event_game.get('teamId')
            
            # Stop if team changed
            if event_team != team_id:
                # Check if we have a valid play (2+ passes but no terminal event)
                if len(pass_events) >= 2:
                    # Team change = possession lost (terminal event)
                    last_event_data = events[current_idx - 1]
                    break
                else:
                    return None
            
            # Parse and add event
            play_event = self.event_parser.parse_event(
                event, stadium_meta, period, is_home, last_target_player
            )
            if play_event:
                play_events.append(play_event)
                last_event_data = event
            
            # Track passes (PA or CR)
            if event_type in ['PA', 'CR']:
                pass_events.append(event)
                last_target_player = event_poss.get('targetPlayerName')
            
            # Check for terminal events
            if event_type == 'SH':
                # Shot - terminal event
                if len(pass_events) >= 2:
                    current_idx += 1
                    break
                else:
                    return None
            elif event_type in ['LO', 'CA', 'TA']:
                # Possession lost - terminal event
                if len(pass_events) >= 2:
                    current_idx += 1
                    break
                else:
                    return None
            
            current_idx += 1
        
        # Validate: must have at least 2 passes
        if len(pass_events) < 2:
            return None
        
        # Create play
        play = self._create_play(
            play_events, first_event_data, last_event_data,
            metadata, stadium_meta
        )
        
        if play:
            return (play, current_idx)
        
        return None
    
    def _get_stadium_metadata(self, events: List[Dict]) -> Optional[Dict]:
        """Extract stadium metadata from events."""
        for event in events:
            if event.get('stadiumMetadata'):
                return event['stadiumMetadata']
        return None
    
    def _is_pass_forward(self, event: Dict, all_events: List[Dict], 
                        event_idx: int, stadium_meta: Dict,
                        team_id: int, period: int, is_home: bool) -> bool:
        """Determine if a pass is forward."""
        ball_data = event.get('ball', [])
        if not ball_data:
            return False
        
        start_x = ball_data[0].get('x', 0)
        
        # Find next ball position
        end_x = None
        for next_event in all_events[event_idx + 1:event_idx + 5]:
            next_ball = next_event.get('ball', [])
            if next_ball and len(next_ball) > 0:
                end_x = next_ball[0].get('x', 0)
                break
        
        if end_x is None:
            return False
        
        attack_dir = determine_attack_direction(is_home, period, stadium_meta)
        threshold = self.config.forward_threshold
        
        if attack_dir == 'R':
            return end_x > start_x + threshold
        else:
            return end_x < start_x - threshold
    
    def _create_play(self, play_events: List[PlayEvent],
                    first_event: Dict, last_event: Dict,
                    metadata: MatchMetadata,
                    stadium_meta: Dict) -> Optional[Play]:
        """Create a Play object from events."""
        if not play_events or len(play_events) < 2:
            return None
        
        # Extract metadata
        game_event = first_event.get('gameEvents', {})
        team_id = game_event.get('teamId')
        team_name = game_event.get('teamName', 'Unknown')
        video_url = game_event.get('videoUrl', '')
        start_game_clock = game_event.get('startGameClock', 0)
        is_home = game_event.get('homeTeam', False)
        period = game_event.get('period', 1)
        
        last_game_event = last_event.get('gameEvents', {})
        end_game_clock = last_game_event.get('startGameClock', start_game_clock)
        
        # Determine attack direction
        original_attack_dir = determine_attack_direction(is_home, period, stadium_meta)
        
        # Timing
        start_time = play_events[0].time
        end_time = play_events[-1].time
        duration = end_time - start_time
        
        # Filter by duration
        if duration < self.config.min_play_duration or duration > self.config.max_play_duration:
            return None
        
        # Determine outcome
        last_poss_event = last_event.get('possessionEvents', {})
        last_event_type = last_poss_event.get('possessionEventType')
        
        outcome = 'POSSESSION_LOST'
        is_goal = False
        
        if last_event_type == 'SH':
            shot_outcome = last_poss_event.get('shotOutcomeType')
            if shot_outcome == 'G':
                outcome = 'GOAL'
                is_goal = True
            else:
                outcome = 'SHOT'
        elif last_event_type == 'LO':
            outcome = 'LOSS'
        elif last_event_type in ['CA', 'TA']:
            outcome = 'TURNOVER'
        
        play_id = f"M{metadata.match_id}_T{team_id}_T{int(start_time)}"
        
        return Play(
            play_id=play_id,
            match_id=metadata.match_id,
            match_name=metadata.match_name,
            team_id=team_id,
            team_name=team_name,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            events=play_events,
            normalized_events=[],
            num_events=len(play_events),
            outcome=outcome,
            is_goal=is_goal,
            delta_x=0,
            delta_y=0,
            max_x=0,
            total_distance=0,
            avg_attackers_ahead=0,
            avg_defenders_ahead=0,
            wing_side='CENTER',
            video_url=video_url,
            original_attack_direction=original_attack_dir,
            start_game_clock=start_game_clock,
            end_game_clock=end_game_clock
        )


class DataLoader:
    """Loads match data from JSON files."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.play_extractor = PlayExtractor()
    
    def load_all_matches(self) -> tuple[List[Play], Dict[int, MatchMetadata]]:
        """
        Load all matches from data directory.
        
        Returns:
            Tuple of (all_plays, match_metadata_dict)
        """
        json_files = sorted(self.data_dir.glob('*.json'))
        
        all_plays = []
        match_metadata = {}
        
        for json_path in json_files:
            match_id = int(json_path.stem)
            
            with open(json_path, 'r', encoding='utf-8') as f:
                events = json.load(f)
            
            # Extract match metadata from events
            if events:
                home_team = 'Unknown'
                away_team = 'Unknown'
                
                # Find home and away team names from events
                for event in events[:50]:  # Check first 50 events
                    game_event = event.get('gameEvents', {})
                    team_name = game_event.get('teamName')
                    is_home = game_event.get('homeTeam', False)
                    
                    if team_name:
                        if is_home:
                            home_team = team_name
                        else:
                            away_team = team_name
                    
                    if home_team != 'Unknown' and away_team != 'Unknown':
                        break
                
                match_name = f"{home_team} vs {away_team}"
                
                metadata = MatchMetadata(
                    match_id=match_id,
                    match_name=match_name,
                    home_team=home_team,
                    away_team=away_team,
                    file_path=str(json_path)
                )
                
                match_metadata[match_id] = metadata
                
                # Extract plays
                plays = self.play_extractor.extract_plays(events, metadata)
                all_plays.extend(plays)
        
        return all_plays, match_metadata
