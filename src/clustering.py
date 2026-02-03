"""
Clustering plays by tactical patterns.
Uses hierarchical clustering based on structural similarity.
"""
import numpy as np
from typing import List, Dict
from collections import OrderedDict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from .models import Play
from .utils import get_feature_vector
from .config import analysis_config


class PlayClusterer:
    """Clusters plays based on tactical patterns."""
    
    def __init__(self, config=None):
        self.config = config or analysis_config
    
    def cluster_plays(self, plays: List[Play]) -> OrderedDict:
        """
        Cluster plays by tactical pattern.
        
        Args:
            plays: List of Play objects with features
        
        Returns:
            OrderedDict mapping cluster_id to list of plays
        """
        # Filter plays with sufficient forward progress
        valid_plays = [
            p for p in plays 
            if p.delta_x >= self.config.min_forward_progress
        ]
        
        if not valid_plays:
            return OrderedDict()
        
        # Extract feature vectors
        feature_matrix = np.array([get_feature_vector(p) for p in valid_plays])
        
        # Perform hierarchical clustering
        distance_matrix = pdist(feature_matrix, metric='euclidean')
        linkage_matrix = linkage(distance_matrix, method='ward')
        
        # Cut dendrogram to form clusters
        cluster_labels = fcluster(
            linkage_matrix, 
            t=self.config.clustering_threshold,
            criterion='distance'
        )
        
        # Assign cluster IDs to plays
        for play, cluster_id in zip(valid_plays, cluster_labels):
            play.cluster_id = int(cluster_id)
        
        # Group plays by cluster
        clusters_dict = {}
        for play in valid_plays:
            cluster_id = play.cluster_id
            if cluster_id not in clusters_dict:
                clusters_dict[cluster_id] = []
            clusters_dict[cluster_id].append(play)
        
        # Filter clusters: Remove clusters with less than 2 plays
        filtered_clusters = {
            cid: plays_list 
            for cid, plays_list in clusters_dict.items() 
            if len(plays_list) >= 2
        }
        
        # Sort by play count (descending - most plays first)
        sorted_clusters = sorted(
            filtered_clusters.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        # Renumber clusters sequentially
        result = OrderedDict()
        for new_id, (old_id, plays_list) in enumerate(sorted_clusters, 1):
            # Update cluster IDs
            for play in plays_list:
                play.cluster_id = new_id
            result[new_id] = plays_list
        
        return result
    
    def calculate_similarity(self, play1: Play, play2: Play) -> float:
        """
        Calculate similarity score between two plays.
        
        Args:
            play1: First play
            play2: Second play
        
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        vec1 = get_feature_vector(play1)
        vec2 = get_feature_vector(play2)
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(vec1 - vec2)
        
        # Convert to similarity (inverse)
        max_distance = 100.0  # Normalization constant
        similarity = 1.0 - min(distance / max_distance, 1.0)
        
        return similarity


class ClusterAnalyzer:
    """Analyzes cluster characteristics."""
    
    def analyze_clusters(self, clusters: OrderedDict) -> Dict:
        """
        Analyze tactical patterns in clusters.
        
        Args:
            clusters: OrderedDict of cluster_id -> plays
        
        Returns:
            Dictionary with detailed cluster analysis
        """
        analysis = {}
        
        for cluster_id, plays in clusters.items():
            cluster_analysis = self._analyze_single_cluster(cluster_id, plays)
            analysis[cluster_id] = cluster_analysis
        
        return analysis
    
    def _analyze_single_cluster(self, cluster_id: int, 
                                plays: List[Play]) -> Dict:
        """Analyze a single cluster."""
        total_plays = len(plays)
        goals = sum(1 for p in plays if p.is_goal)
        shots = sum(1 for p in plays if p.outcome in ['SHOT', 'GOAL'])
        losses = sum(1 for p in plays if p.outcome == 'POSSESSION_LOST')
        
        # Calculate averages
        avg_duration = np.mean([p.duration for p in plays])
        avg_forward = np.mean([p.delta_x for p in plays])
        avg_events = np.mean([p.num_events for p in plays])
        
        # Determine cluster name
        cluster_name = self._generate_cluster_name(plays)
        
        # Update cluster names in plays
        for play in plays:
            play.cluster_name = cluster_name
        
        return {
            'cluster_id': cluster_id,
            'name': cluster_name,
            'total': total_plays,
            'goals': goals,
            'shots': shots,
            'losses': losses,
            'avg_duration': avg_duration,
            'avg_forward': avg_forward,
            'avg_events': avg_events,
            'wing_plays': sum(1 for p in plays if p.wing_side == 'WING')
        }
    
    def _generate_cluster_name(self, plays: List[Play]) -> str:
        """Generate descriptive name for cluster based on tactical patterns."""
        # Analyze cluster characteristics
        total = len(plays)
        
        # Wing preference
        wing_plays = sum(1 for p in plays if p.wing_side == 'WING')
        wing_pct = wing_plays / total if total > 0 else 0
        
        # Outcome analysis
        goals = sum(1 for p in plays if p.is_goal)
        shots = sum(1 for p in plays if p.outcome in ['SHOT', 'GOAL'])
        
        # Distance metrics
        avg_forward = np.mean([p.delta_x for p in plays])
        avg_duration = np.mean([p.duration for p in plays])
        avg_events = np.mean([p.num_events for p in plays])
        
        # Build cluster name
        name_parts = []
        
        # Attack position
        if wing_pct >= 0.7:
            name_parts.append("Wing Attack")
        elif wing_pct <= 0.3:
            name_parts.append("Central Attack")
        else:
            name_parts.append("Mixed Attack")
        
        # Speed & length
        if avg_duration < 5:
            name_parts.append("Fast")
        elif avg_duration > 10:
            name_parts.append("Slow Build")
        else:
            name_parts.append("Medium")
        
        # Penetration
        if avg_forward > 30:
            name_parts.append("Deep")
        elif avg_forward > 20:
            name_parts.append("Mid")
        else:
            name_parts.append("Short")
        
        # Success rate
        if total >= 2:
            success_rate = goals / total
            if success_rate >= 0.3:
                name_parts.append("High-Conv")
            elif success_rate > 0:
                name_parts.append("Low-Conv")
        
        return " ".join(name_parts)
