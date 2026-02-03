"""
Interactive browser for exploring clustered plays.
"""
import matplotlib.pyplot as plt
from typing import List, Optional
from collections import OrderedDict

from .models import Play
from .clustering import PlayClusterer
from .visualizer import FieldVisualizer, ComparisonPrinter


class PlayBrowser:
    """Interactive browser for navigating through clustered plays."""
    
    def __init__(self, cluster_id: int, clusters: OrderedDict, 
                 cluster_analysis: dict):
        """
        Initialize browser for a specific cluster.
        
        Args:
            cluster_id: Cluster to browse
            clusters: All clusters
            cluster_analysis: Analysis results
        """
        self.cluster_id = cluster_id
        self.plays = clusters.get(cluster_id, [])
        self.cluster_info = cluster_analysis.get(cluster_id, {})
        
        self.visualizer = FieldVisualizer()
        self.printer = ComparisonPrinter()
        self.clusterer = PlayClusterer()
        
        cluster_name = self.cluster_info.get('name', 'Unknown')
        print(f"✅ Browser initialized for Cluster #{cluster_id} - {cluster_name}")
        print(f"   {len(self.plays)} plays in this pattern")
        print(f"   Use browser.list() to see all plays")
        print(f"   Use browser.compare(1, 2) to compare play #1 with play #2")
        print(f"   Use browser.summary() to see statistics")
    
    def list(self):
        """List all plays in the cluster."""
        cluster_name = self.cluster_info.get('name', 'Unknown')
        print(f"\n{'='*100}")
        print(f"CLUSTER #{self.cluster_id}: {cluster_name}")
        print(f"{'='*100}")
        print(f"Total Plays: {len(self.plays)}\n")
        
        print(f"{'#':<4} {'Team':<20} {'Match':<35} {'Time':<15} "
              f"{'Outcome':<15} {'Events':<7} {'Forward':<8}")
        print("-" * 100)
        
        for i, play in enumerate(self.plays, 1):
            outcome_emoji = '⚽' if play.is_goal else '🎯' if play.outcome == 'SHOT' else '❌'
            print(f"{i:<4} {play.team_name:<20} {play.match_name:<35} "
                  f"{play.time_range_display:<15} "
                  f"{outcome_emoji} {play.outcome:<12} "
                  f"{play.num_events:<7} {play.delta_x:<8.1f}")
    
    def compare(self, play1_num: int, play2_num: int):
        """
        Compare two plays in the cluster.
        
        Args:
            play1_num: First play number (1-indexed)
            play2_num: Second play number (1-indexed)
        """
        if not (1 <= play1_num <= len(self.plays)):
            print(f"❌ Play #{play1_num} not found (valid range: 1-{len(self.plays)})")
            return
        
        if not (1 <= play2_num <= len(self.plays)):
            print(f"❌ Play #{play2_num} not found (valid range: 1-{len(self.plays)})")
            return
        
        play1 = self.plays[play1_num - 1]
        play2 = self.plays[play2_num - 1]
        
        cluster_name = self.cluster_info.get('name', 'Unknown')
        print(f"\n🔄 Comparing Play #{play1_num} vs Play #{play2_num}")
        print(f"   Pattern: {cluster_name}")
        print(f"   {play1.team_name} ({play1.outcome}) vs "
              f"{play2.team_name} ({play2.outcome})\n")
        
        # Visualize
        fig = self.visualizer.compare_plays(play1, play2, show_details=True)
        plt.show()
        
        # Calculate similarity
        similarity = self.clusterer.calculate_similarity(play1, play2)
        
        # Print detailed comparison
        self.printer.print_comparison(play1, play2, similarity)
    
    def summary(self):
        """Display cluster statistics."""
        print(f"\n{'='*80}")
        print(f"CLUSTER #{self.cluster_id} SUMMARY")
        print(f"{'='*80}")
        
        cluster_name = self.cluster_info.get('name', 'Unknown')
        print(f"Pattern Name: {cluster_name}")
        print(f"\n📊 Statistics:")
        print(f"   Total Plays: {self.cluster_info.get('total', 0)}")
        print(f"   Goals: {self.cluster_info.get('goals', 0)}")
        print(f"   Shots: {self.cluster_info.get('shots', 0)}")
        print(f"   Possession Lost: {self.cluster_info.get('losses', 0)}")
        print(f"\n📈 Averages:")
        print(f"   Duration: {self.cluster_info.get('avg_duration', 0):.1f} seconds")
        print(f"   Forward Progress: {self.cluster_info.get('avg_forward', 0):.1f} meters")
        print(f"   Events per Play: {self.cluster_info.get('avg_events', 0):.1f}")
        print(f"\n📍 Position:")
        wing_plays = self.cluster_info.get('wing_plays', 0)
        total = self.cluster_info.get('total', 1)
        print(f"   Wing Plays: {wing_plays} ({100*wing_plays/total:.1f}%)")
        print(f"   Central Plays: {total-wing_plays} ({100*(total-wing_plays)/total:.1f}%)")
        print(f"\n{'='*80}\n")


# Import matplotlib for browser
import matplotlib.pyplot as plt
