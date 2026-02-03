"""
Main entry point for Football Tactical Pattern Analysis.
"""
import pandas as pd
from pathlib import Path

from .config import path_config, analysis_config
from .data_loader import DataLoader
from .feature_engineer import FeatureEngineer
from .clustering import PlayClusterer, ClusterAnalyzer
from .browser import PlayBrowser


class TacticalAnalyzer:
    """Main orchestrator for tactical analysis pipeline."""
    
    def __init__(self, data_dir: Path = None, config=None):
        """
        Initialize the tactical analyzer.
        
        Args:
            data_dir: Directory containing event JSON files
            config: Analysis configuration
        """
        self.config = config or analysis_config
        self.paths = path_config if data_dir is None else type('obj', (object,), {
            'data_dir': Path(data_dir),
            'cluster_csv': Path(data_dir).parent / 'output' / 'cluster_analysis.csv'
        })()
        
        # Initialize components
        self.data_loader = DataLoader(self.paths.data_dir)
        self.feature_engineer = FeatureEngineer(self.config)
        self.clusterer = PlayClusterer(self.config)
        self.cluster_analyzer = ClusterAnalyzer()
        
        # State
        self.all_plays = []
        self.match_metadata = {}
        self.clusters = None
        self.cluster_analysis = None
        self.total_extracted_plays = 0
    
    def run_analysis(self) -> dict:
        """
        Run complete analysis pipeline.
        
        Returns:
            Dictionary with analysis results
        """
        print("="*80)
        print("FOOTBALL TACTICAL PATTERN ANALYSIS")
        print("="*80)
        
        # Step 1: Load data
        print("\n📂 Loading match data...")
        self.all_plays, self.match_metadata = self.data_loader.load_all_matches()
        print(f"   ✅ Extracted {len(self.all_plays)} plays from {len(self.match_metadata)} matches")
        print(f"   (Each play = 2+ passes followed by shot/loss)")
        
        # Step 2: Engineer features
        print("\n⚙️ Engineering features...")
        self.all_plays = self.feature_engineer.engineer_features(self.all_plays)
        print(f"   Features calculated for {len(self.all_plays)} plays")
        
        # Step 3: Cluster plays
        print("\n🎯 Clustering plays by tactical pattern...")
        self.clusters = self.clusterer.cluster_plays(self.all_plays)
        print(f"   Found {len(self.clusters)} distinct tactical patterns")
        
        # Step 4: Analyze clusters
        print("\n📊 Analyzing tactical patterns...")
        self.cluster_analysis = self.cluster_analyzer.analyze_clusters(self.clusters)
        
        # Step 5: Export results
        print("\n💾 Exporting results...")
        self._export_cluster_analysis()
        print(f"   Saved to: {self.paths.cluster_csv}")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
        return {
            'plays': self.all_plays,
            'clusters': self.clusters,
            'analysis': self.cluster_analysis,
            'match_metadata': self.match_metadata
        }
    
    def reanalyze_with_threshold(self, threshold: float) -> dict:
        """
        Re-run clustering with new threshold (keeps loaded data).
        
        Args:
            threshold: New clustering threshold value
        
        Returns:
            Dictionary with updated analysis results
        """
        if not self.all_plays:
            raise ValueError("No data loaded - run full analysis first")
        
        print("\n" + "="*80)
        print(f"RE-CLUSTERING with threshold = {threshold}")
        print("="*80)
        
        # Update config
        self.config.clustering_threshold = threshold
        self.clusterer = PlayClusterer(self.config)
        
        # Re-cluster
        print("\n🎯 Re-clustering plays...")
        self.clusters = self.clusterer.cluster_plays(self.all_plays)
        print(f"   Found {len(self.clusters)} distinct tactical patterns")
        
        # Re-analyze
        print("\n📊 Analyzing patterns...")
        self.cluster_analysis = self.cluster_analyzer.analyze_clusters(self.clusters)
        
        print("\n" + "="*80)
        print("RE-ANALYSIS COMPLETE")
        print("="*80)
        
        return {
            'clusters': self.clusters,
            'analysis': self.cluster_analysis
        }
    
    def create_browser(self, cluster_id: int) -> PlayBrowser:
        """
        Create an interactive browser for a cluster.
        
        Args:
            cluster_id: Cluster to browse
        
        Returns:
            PlayBrowser instance
        """
        if self.clusters is None:
            raise ValueError("Run analysis first before creating browser")
        
        return PlayBrowser(cluster_id, self.clusters, self.cluster_analysis)
    
    def print_cluster_summary(self):
        """Print summary of all clusters."""
        if self.cluster_analysis is None:
            print("❌ No analysis results. Run analysis first.")
            return
        
        print("\n" + "="*80)
        print("TACTICAL PATTERN SUMMARY")
        print("="*80)
        
        for cluster_id, analysis in self.cluster_analysis.items():
            print(f"\nCluster #{cluster_id}: {analysis['name']}")
            print(f"  Total: {analysis['total']} plays")
            print(f"  Goals: {analysis['goals']}, Shots: {analysis['shots']}, "
                  f"Losses: {analysis['losses']}")
            print(f"  Avg Duration: {analysis['avg_duration']:.1f}s, "
                  f"Avg Forward: {analysis['avg_forward']:.1f}m")
    
    def _export_cluster_analysis(self):
        """Export cluster analysis to CSV."""
        if not self.cluster_analysis:
            return
        
        rows = []
        for cluster_id, analysis in self.cluster_analysis.items():
            rows.append({
                'cluster_id': cluster_id,
                'pattern_name': analysis['name'],
                'total_plays': analysis['total'],
                'goals': analysis['goals'],
                'shots': analysis['shots'],
                'losses': analysis['losses'],
                'avg_duration': analysis['avg_duration'],
                'avg_forward_progress': analysis['avg_forward'],
                'avg_events': analysis['avg_events'],
                'wing_plays': analysis['wing_plays']
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(self.paths.cluster_csv, index=False)


def main():
    """Main execution function."""
    # Create analyzer
    analyzer = TacticalAnalyzer()
    
    # Run analysis
    results = analyzer.run_analysis()
    
    # Print summary
    analyzer.print_cluster_summary()
    
    # Example: Create browser for cluster 1
    print("\n" + "="*80)
    print("Creating browser for Cluster #1...")
    print("="*80)
    browser = analyzer.create_browser(1)
    
    # Show how to use
    print("\n💡 Example usage:")
    print("   browser.list()          # List all plays")
    print("   browser.summary()       # Show statistics")
    print("   browser.compare(1, 2)   # Compare plays")
    
    return analyzer, browser


if __name__ == '__main__':
    analyzer, browser = main()
