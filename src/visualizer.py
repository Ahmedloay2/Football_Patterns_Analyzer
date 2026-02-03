"""
Visualization for play comparisons.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from typing import List, Optional

from .models import Play


class FieldVisualizer:
    """Visualizes plays on a football field."""
    
    FIELD_LENGTH = 105.0  # meters
    FIELD_WIDTH = 68.0  # meters
    
    def __init__(self):
        self.colors = {
            'field': '#2d8f2d',
            'lines': 'white',
            'play1': '#FFD700',
            'play2': '#4169E1'
        }
    
    def draw_field(self, ax, field_color='#2d8f2d'):
        """Draw a football field on the given axes."""
        ax.set_xlim(-self.FIELD_LENGTH/2 - 5, self.FIELD_LENGTH/2 + 5)
        ax.set_ylim(-self.FIELD_WIDTH/2 - 5, self.FIELD_WIDTH/2 + 5)
        
        # Field background
        field_rect = patches.Rectangle(
            (-self.FIELD_LENGTH/2, -self.FIELD_WIDTH/2),
            self.FIELD_LENGTH, self.FIELD_WIDTH,
            linewidth=2, edgecolor='white', facecolor=field_color
        )
        ax.add_patch(field_rect)
        
        # Center line
        ax.plot([0, 0], [-self.FIELD_WIDTH/2, self.FIELD_WIDTH/2], 
                'white', linewidth=2)
        
        # Center circle
        center_circle = plt.Circle((0, 0), 9.15, fill=False, 
                                  color='white', linewidth=2)
        ax.add_patch(center_circle)
        
        # Penalty areas
        for side in [-1, 1]:
            x_pos = side * self.FIELD_LENGTH/2
            
            # Penalty area
            penalty_box = patches.Rectangle(
                (x_pos - side*16.5, -20.15),
                side*16.5, 40.3,
                linewidth=2, edgecolor='white', facecolor='none'
            )
            ax.add_patch(penalty_box)
            
            # Goal area
            goal_box = patches.Rectangle(
                (x_pos - side*5.5, -9.16),
                side*5.5, 18.32,
                linewidth=2, edgecolor='white', facecolor='none'
            )
            ax.add_patch(goal_box)
        
        ax.set_aspect('equal')
        ax.axis('off')
    
    def plot_play(self, ax, play: Play, color='#FFD700', 
                 alpha=0.9, show_annotations=True):
        """Plot a single play on the field."""
        events = play.normalized_events
        
        if not events:
            return
        
        # Extract coordinates
        x_coords = [e.ball_x for e in events]
        y_coords = [e.ball_y for e in events]
        
        # Plot path
        ax.plot(x_coords, y_coords, color=color, linewidth=3, 
                alpha=alpha, marker='o', markersize=8, zorder=10)
        
        # Mark start and end
        ax.scatter(x_coords[0], y_coords[0], color=color, s=200, 
                  marker='o', edgecolor='white', linewidth=2, 
                  zorder=15, label='START')
        
        outcome_marker = '⚽' if play.is_goal else '🎯' if play.outcome == 'SHOT' else '❌'
        ax.scatter(x_coords[-1], y_coords[-1], color=color, s=300,
                  marker='X', edgecolor='white', linewidth=2,
                  zorder=15)
        
        if show_annotations:
            ax.text(x_coords[-1] + 2, y_coords[-1] + 2, outcome_marker,
                   fontsize=16, color='white', weight='bold',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    def compare_plays(self, play1: Play, play2: Play, 
                     show_details=True) -> plt.Figure:
        """
        Create comprehensive comparison with field plots and detailed stats.
        
        Args:
            play1: First play
            play2: Second play
            show_details: Whether to show detailed comparison table
        
        Returns:
            Matplotlib figure with integrated visualization and details
        """
        # Create figure with field plots and text area
        fig = plt.figure(figsize=(22, 12))
        fig.patch.set_facecolor('#1a1a1a')
        
        # Create grid: 2 fields on top, details below
        gs = fig.add_gridspec(2, 2, height_ratios=[2.5, 1], hspace=0.25, wspace=0.15,
                             left=0.05, right=0.95, top=0.92, bottom=0.05)
        
        # Field plots
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Details text area (spanning both columns)
        ax_details = fig.add_subplot(gs[1, :])
        ax_details.axis('off')
        
        # Plot play 1
        self.draw_field(ax1, field_color=self.colors['field'])
        self.plot_play(ax1, play1, color=self.colors['play1'], 
                      alpha=0.95, show_annotations=True)
        
        outcome1 = 'GOAL' if play1.is_goal else 'SHOT' if play1.outcome == 'SHOT' else 'LOST'
        title1 = (f'PLAY 1 (GOLD) - {outcome1}\n{play1.team_name}\n'
                 f'{play1.match_name}\n'
                 f'Time: {play1.time_range_display} ({play1.duration:.1f}s) | '
                 f'{play1.num_events} events | Forward: {play1.delta_x:.1f}m | '
                 f'{play1.wing_side}\n'
                 f'Pattern: #{play1.cluster_id} - {play1.cluster_name}')
        
        ax1.set_title(title1, fontsize=10, fontweight='bold', pad=15,
                     color='white', bbox=dict(boxstyle='round', 
                     facecolor='#333333', alpha=0.9))
        
        # Plot play 2
        self.draw_field(ax2, field_color=self.colors['field'])
        self.plot_play(ax2, play2, color=self.colors['play2'],
                      alpha=0.95, show_annotations=True)
        
        outcome2 = 'GOAL' if play2.is_goal else 'SHOT' if play2.outcome == 'SHOT' else 'LOST'
        title2 = (f'PLAY 2 (BLUE) - {outcome2}\n{play2.team_name}\n'
                 f'{play2.match_name}\n'
                 f'Time: {play2.time_range_display} ({play2.duration:.1f}s) | '
                 f'{play2.num_events} events | Forward: {play2.delta_x:.1f}m | '
                 f'{play2.wing_side}\n'
                 f'Pattern: #{play2.cluster_id} - {play2.cluster_name}')
        
        ax2.set_title(title2, fontsize=10, fontweight='bold', pad=15,
                     color='white', bbox=dict(boxstyle='round',
                     facecolor='#333333', alpha=0.9))
        
        # Overall title
        if play1.cluster_id == play2.cluster_id:
            main_title = (f'TACTICAL PATTERN COMPARISON: {play1.cluster_name} '
                         f'(Cluster #{play1.cluster_id})')
        else:
            main_title = f'COMPARING DIFFERENT PATTERNS'
        
        fig.suptitle(main_title, fontsize=14, fontweight='bold', 
                    color='white')
        
        # Add detailed comparison text in bottom area
        if show_details:
            details_text = self._create_details_text(play1, play2)
            ax_details.text(0.02, 0.95, details_text, 
                          fontsize=8, fontfamily='monospace',
                          verticalalignment='top', color='white',
                          bbox=dict(boxstyle='round', facecolor='#2a2a2a', 
                                  alpha=0.95, pad=15))
        
        return fig
    
    def _create_details_text(self, play1: Play, play2: Play) -> str:
        """Create formatted text with play details."""
        text_lines = []
        text_lines.append("TACTICAL COMPARISON DETAILS")
        text_lines.append("=" * 150)
        text_lines.append("")
        
        # Comparison table
        text_lines.append(f"{'Metric':<35} {'Play 1 (Gold)':<25} {'Play 2 (Blue)':<25} {'Difference':<20}")
        text_lines.append("-" * 150)
        
        metrics = [
            ('Team', play1.team_name, play2.team_name, ''),
            ('Duration (s)', f'{play1.duration:.1f}', f'{play2.duration:.1f}', 
             f'{abs(play1.duration - play2.duration):.1f}s'),
            ('Forward Progress (m)', f'{play1.delta_x:.1f}', f'{play2.delta_x:.1f}',
             f'{abs(play1.delta_x - play2.delta_x):.1f}m'),
            ('Total Distance (m)', f'{play1.total_distance:.1f}', f'{play2.total_distance:.1f}',
             f'{abs(play1.total_distance - play2.total_distance):.1f}m'),
            ('Number of Events', str(play1.num_events), str(play2.num_events),
             str(abs(play1.num_events - play2.num_events))),
            ('Attackers Ahead (avg)', f'{play1.avg_attackers_ahead:.1f}', 
             f'{play2.avg_attackers_ahead:.1f}',
             f'{abs(play1.avg_attackers_ahead - play2.avg_attackers_ahead):.1f}'),
            ('Outcome', play1.outcome, play2.outcome,
             'GOAL' if (play1.is_goal or play2.is_goal) else ''),
        ]
        
        for metric, val1, val2, diff in metrics:
            text_lines.append(f"{metric:<35} {val1:<25} {val2:<25} {diff:<20}")
        
        text_lines.append("")
        text_lines.append("EVENT SEQUENCES")
        text_lines.append("-" * 150)
        
        # Play 1 events
        text_lines.append(f"Play 1: {play1.team_name} - {play1.outcome}")
        for i, event in enumerate(play1.events, 1):
            player = event.player_name if event.player_name else "Unknown"
            text_lines.append(f"  {i}. {event.event_type:<10} {player:<30} @ ({event.ball_x:5.1f}, {event.ball_y:5.1f})")
        
        text_lines.append("")
        
        # Play 2 events
        text_lines.append(f"Play 2: {play2.team_name} - {play2.outcome}")
        for i, event in enumerate(play2.events, 1):
            player = event.player_name if event.player_name else "Unknown"
            text_lines.append(f"  {i}. {event.event_type:<10} {player:<30} @ ({event.ball_x:5.1f}, {event.ball_y:5.1f})")
        
        return '\n'.join(text_lines)


class ComparisonPrinter:
    """Prints detailed play comparison information."""
    
    def print_comparison(self, play1: Play, play2: Play, similarity: float):
        """Print detailed comparison table."""
        print("\n" + "=" * 100)
        print("DETAILED PLAY COMPARISON")
        print("=" * 100)
        
        # Use ASCII alternatives for emojis to ensure compatibility
        comparison_data = {
            'Metric': [
                'Team', 'Match', 'Time Range', 'Duration (s)',
                'Forward Progress (m)', 'Lateral Movement (m)',
                'Total Distance (m)', 'Number of Events',
                'Avg Attackers Ahead', 'Avg Defenders Ahead',
                'Position (Wing/Center)', 'Outcome', 'Goal?',
                'Pattern Cluster'
            ],
            'Play 1 (Gold)': [
                play1.team_name, play1.match_name, play1.time_range_display,
                f'{play1.duration:.1f}', f'{play1.delta_x:.1f}', 
                f'{play1.delta_y:.1f}', f'{play1.total_distance:.1f}',
                play1.num_events, f'{play1.avg_attackers_ahead:.1f}',
                f'{play1.avg_defenders_ahead:.1f}', play1.wing_side,
                play1.outcome, 'YES' if play1.is_goal else 'NO',
                f"#{play1.cluster_id} - {play1.cluster_name}" if play1.cluster_id else 'N/A'
            ],
            'Play 2 (Blue)': [
                play2.team_name, play2.match_name, play2.time_range_display,
                f'{play2.duration:.1f}', f'{play2.delta_x:.1f}',
                f'{play2.delta_y:.1f}', f'{play2.total_distance:.1f}',
                play2.num_events, f'{play2.avg_attackers_ahead:.1f}',
                f'{play2.avg_defenders_ahead:.1f}', play2.wing_side,
                play2.outcome, 'YES' if play2.is_goal else 'NO',
                f"#{play2.cluster_id} - {play2.cluster_name}" if play2.cluster_id else 'N/A'
            ]
        }
        
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        
        print(f"\nPattern Similarity Score: {similarity:.3f}")
        print(f"   (0.0 = very different, 1.0 = identical patterns)")
        
        # Tactical Analysis
        print(f"\n" + "-" * 100)
        print(f"TACTICAL ANALYSIS")
        print("-" * 100)
        
        print(f"\nPlay 1 (Gold): {play1.team_name}")
        print(f"  Attack Type: {play1.wing_side}")
        print(f"  Build-up Speed: {play1.duration:.1f} seconds")
        print(f"  Vertical Penetration: {play1.delta_x:.1f} meters forward")
        print(f"  Horizontal Width: {play1.delta_y:.1f} meters lateral")
        print(f"  Ball Movement: {play1.total_distance:.1f} meters total")
        print(f"  Offensive Support: {play1.avg_attackers_ahead:.1f} players ahead on average")
        print(f"  Defensive Pressure: {play1.avg_defenders_ahead:.1f} defenders ahead on average")
        print(f"  Result: {play1.outcome} {'[GOAL]' if play1.is_goal else ''}")
        
        print(f"\nPlay 2 (Blue): {play2.team_name}")
        print(f"  Attack Type: {play2.wing_side}")
        print(f"  Build-up Speed: {play2.duration:.1f} seconds")
        print(f"  Vertical Penetration: {play2.delta_x:.1f} meters forward")
        print(f"  Horizontal Width: {play2.delta_y:.1f} meters lateral")
        print(f"  Ball Movement: {play2.total_distance:.1f} meters total")
        print(f"  Offensive Support: {play2.avg_attackers_ahead:.1f} players ahead on average")
        print(f"  Defensive Pressure: {play2.avg_defenders_ahead:.1f} defenders ahead on average")
        print(f"  Result: {play2.outcome} {'[GOAL]' if play2.is_goal else ''}")
        
        # Event sequences
        print(f"\n" + "-" * 100)
        print(f"EVENT SEQUENCES WITH PLAYER DETAILS")
        print("-" * 100)
        
        self._print_event_sequence(play1, "Play 1 (Gold)")
        self._print_event_sequence(play2, "Play 2 (Blue)")
        
        print("\n" + "=" * 100 + "\n")
    
    def _print_event_sequence(self, play: Play, label: str):
        """Print event sequence for a play."""
        outcome_marker = '[GOAL]' if play.is_goal else '[SHOT]' if play.outcome == 'SHOT' else '[LOST]'
        print(f"\n>> {label}: {play.team_name} - {play.outcome} {outcome_marker}")
        print(f"   Match: {play.match_name} | Time: {play.time_range_display} ({play.duration:.1f}s)")
        print(f"   Forward Progress: {play.delta_x:.1f}m | Position: {play.wing_side}")
        print(f"\n   Event Sequence ({play.num_events} events):")
        print(f"   {'#':<3} {'Event Type':<15} {'Player Name':<30} {'Position':<15} {'Context'}")
        print(f"   {'-'*3} {'-'*15} {'-'*30} {'-'*15} {'-'*20}")
        
        for i, event in enumerate(play.events, 1):
            player_name = event.player_name if event.player_name else "Unknown"
            position = f"({event.ball_x:5.1f}, {event.ball_y:5.1f})"
            context = f"{event.attacking_players_ahead} ahead"
            
            print(f"   {i:<3} {event.event_type:<15} {player_name:<30} {position:<15} {context}")
