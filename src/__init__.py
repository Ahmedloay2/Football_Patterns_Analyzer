"""
Football Tactical Pattern Analysis System

A comprehensive OOP-based system for analyzing football/soccer tactical patterns
from event data, using hierarchical clustering and interactive visualization.

Modules:
    - config: Configuration settings
    - models: Data models (MatchMetadata, PlayEvent, Play)
    - utils: Utility functions
    - data_loader: Data loading and parsing
    - feature_engineer: Feature calculation
    - clustering: Play clustering and analysis
    - visualizer: Visualization components
    - browser: Interactive play browser
    - main: Main analysis pipeline

Usage:
    from src.main import TacticalAnalyzer
    
    analyzer = TacticalAnalyzer()
    results = analyzer.run_analysis()
    browser = analyzer.create_browser(cluster_id=1)
    browser.compare(1, 2)
"""

__version__ = '1.0.0'
__author__ = 'Football Analytics Team'

from .config import analysis_config, path_config
from .models import MatchMetadata, PlayEvent, Play
from .data_loader import DataLoader
from .feature_engineer import FeatureEngineer
from .clustering import PlayClusterer, ClusterAnalyzer
from .visualizer import FieldVisualizer, ComparisonPrinter
from .browser import PlayBrowser
from .main import TacticalAnalyzer

__all__ = [
    'analysis_config',
    'path_config',
    'MatchMetadata',
    'PlayEvent',
    'Play',
    'DataLoader',
    'FeatureEngineer',
    'PlayClusterer',
    'ClusterAnalyzer',
    'FieldVisualizer',
    'ComparisonPrinter',
    'PlayBrowser',
    'TacticalAnalyzer'
]
