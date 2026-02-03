# Football Tactical Pattern Analysis System

A comprehensive data analysis system for identifying and clustering similar tactical plays in football (soccer) matches using hierarchical clustering and advanced feature engineering. Built with Object-Oriented Programming principles and featuring an interactive GUI for exploration.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Project Architecture](#project-architecture)
- [Play Definition & Extraction Algorithm](#play-definition--extraction-algorithm)
- [Feature Engineering](#feature-engineering)
- [Clustering Algorithm](#clustering-algorithm)
- [Configuration Parameters](#configuration-parameters)
- [GUI Usage](#gui-usage)
- [Code Structure](#code-structure)
- [Customization Guide](#customization-guide)
- [Algorithm Tuning](#algorithm-tuning)
- [Output Files](#output-files)
- [Requirements](#requirements)

---

## 🎯 Overview

This system analyzes football match event data to:
1. **Extract tactical plays** from raw event sequences
2. **Engineer features** that capture tactical characteristics
3. **Cluster similar plays** using hierarchical clustering
4. **Generate descriptive names** for each tactical pattern
5. **Provide interactive exploration** through a GUI
6. **Visualize plays** on football field diagrams
7. **Compare plays** side-by-side with detailed metrics

### What is a "Play"?

A **play** is defined as a sequence of events by a single team that:
- **Starts** with a forward pass (PA event)
- **Contains** at least 2 passes (PA or CR - cross events)
- **Ends** with a terminal event:
  - Possession lost (team changes, LO, CA, TA events)
  - Shot taken (SH event - may result in GOAL)
- **Same team** maintains possession throughout
- **Intermediate events** (dribbles, touches, etc.) are included but don't count toward the pass requirement

**Example Play:**
```
Pass (PA) → Dribble (DR) → Pass (PA) → Touch (IT) → Shot (SH) ✓ VALID (2 passes + shot)
Pass (PA) → Pass (PA) → Team Change ✓ VALID (2 passes + possession lost)
Pass (PA) → Shot (SH) ✗ INVALID (only 1 pass)
```

---

## ✨ Features

### Core Analysis
- **Automatic play extraction** from JSON event data
- **Multi-dimensional feature engineering** (13+ features per play)
- **Hierarchical clustering** using Ward's linkage method
- **Automatic cluster naming** based on tactical characteristics
- **Dynamic re-clustering** with adjustable threshold

### Visualization
- **Field plots** showing ball movement paths
- **Side-by-side comparison** of plays
- **Color-coded outcomes** (goal, shot, possession lost)
- **Player position tracking** throughout plays

### Interactive GUI
- **Browse clusters** with descriptive names
- **List all plays** in a cluster with details
- **View cluster statistics** (goals, shots, averages)
- **Compare any two plays** with similarity scores
- **Adjust clustering threshold** and re-analyze in real-time

---

## 🚀 Quick Start

### Installation

1. **Install required packages:**
```bash
pip install -r requirements.txt
```

2. **Ensure your data is in the correct location:**
   - Place JSON event files in: `Event Data/` folder
   - JSON files should follow StatsBomb-style event format

### Running the Application

**Option 1: GUI Application (Recommended)**
```bash
python run_gui.py
```

**Option 2: Command Line**
```python
from src.main import TacticalAnalyzer

analyzer = TacticalAnalyzer()
results = analyzer.run_analysis()
browser = analyzer.create_browser(cluster_id=1)
browser.list()
browser.compare(1, 2)
```

---

## 🏗️ Project Architecture

The system follows **SOLID principles** for maintainability and extensibility:

### File Structure

```
📦 Project Root
├── 📁 Event Data/           # Input JSON files (match event data)
├── 📁 output/               # Generated analysis results
│   ├── all_plays.csv
│   ├── cluster_analysis.csv
│   ├── cluster_summary.csv
│   └── detailed_clusters.json
├── 📁 src/                  # Core source code
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── models.py           # Data models (Play, PlayEvent, etc.)
│   ├── utils.py            # Utility functions
│   ├── data_loader.py      # JSON parsing & play extraction
│   ├── feature_engineer.py # Feature calculation
│   ├── clustering.py       # Clustering algorithms
│   ├── visualizer.py       # Field visualization
│   ├── browser.py          # Interactive exploration
│   └── main.py             # Main analysis pipeline
├── gui_app.py              # Tkinter GUI application
├── run_gui.py              # GUI launcher
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

### SOLID Principles Implementation

#### 1. **Single Responsibility Principle (SRP)**
Each module has one clear purpose:
- `config.py` → Manages configuration settings
- `models.py` → Defines data structures
- `data_loader.py` → Loads and parses data
- `feature_engineer.py` → Calculates features
- `clustering.py` → Performs clustering
- `visualizer.py` → Creates visualizations
- `browser.py` → Provides interactive interface

#### 2. **Open/Closed Principle (OCP)**
- Easy to add new event types without modifying existing code
- New clustering algorithms can be added by extending `PlayClusterer`
- New features can be added in `feature_engineer.py` without breaking existing features

#### 3. **Liskov Substitution Principle (LSP)**
- All components work with abstract interfaces
- `PlayExtractor`, `PlayClusterer`, `FeatureEngineer` can be replaced with alternative implementations

#### 4. **Interface Segregation Principle (ISP)**
- Small, focused classes with minimal public methods
- GUI components separate from core analysis logic

#### 5. **Dependency Inversion Principle (DIP)**
- Components depend on configuration objects, not hardcoded values
- Easy to test with mock configurations
- Database/storage layer abstracted through `PathConfig`

---

## 🔍 Play Definition & Extraction Algorithm

### Algorithm Overview (`src/data_loader.py`)

The play extraction uses a **sliding window** approach:

```python
def extract_plays(events):
    """
    Scan through all events looking for valid play sequences.
    """
    plays = []
    i = 0
    
    while i < len(events):
        # Try to extract a play starting from position i
        play_data = try_extract_play(events, i)
        
        if play_data:
            play, next_idx = play_data
            plays.append(play)
            i = next_idx  # Jump to end of this play
        else:
            i += 1  # Move to next event
    
    return plays
```

### Detailed Extraction Logic

**Step 1: Find Starting Point**
- Scan for forward pass (PA event)
- Verify pass direction matches team's attack direction
- Calculate attack direction based on:
  - Stadium metadata
  - Team (home/away)
  - Period (1st/2nd half)

**Step 2: Collect Same-Team Events**
```python
while current_event.team_id == starting_team_id:
    # Add event to play
    
    if event_type in ['PA', 'CR']:
        pass_count += 1
        
    if event_type in ['SH', 'LO', 'CA', 'TA']:
        # Terminal event - check if valid play
        if pass_count >= 2:
            create_play()
        break
```

**Step 3: Validation**
- **Minimum passes:** 2 (configurable in `config.py`)
- **Duration:** 3-30 seconds (configurable)
- **Forward progress:** ≥5 meters (configurable)

**Step 4: Play Creation**
- Extract all metadata (team, match, time)
- Calculate features (see Feature Engineering)
- Normalize coordinates to standard field orientation
- Determine outcome (GOAL, SHOT, POSSESSION_LOST, etc.)

### Why This Definition?

This definition captures **meaningful attacking sequences**:
- **2+ passes** filters out simple turnovers
- **Terminal event** ensures plays have clear outcomes
- **Same team** requirement maintains tactical coherence
- **Intermediate events** preserve full context (dribbles, touches)

---

## 🧮 Feature Engineering

### Features Calculated (`src/feature_engineer.py`)

The system calculates **13 features** per play grouped into 4 categories:

#### 1. **Spatial Features**
| Feature | Description | Calculation | Tactical Meaning |
|---------|-------------|-------------|------------------|
| `delta_x` | Forward progress | `final_x - initial_x` | Penetration depth |
| `delta_y` | Lateral movement | `final_y - initial_y` | Width of attack |
| `max_x` | Deepest penetration | `max(all_x_coords)` | Threat level |
| `total_distance` | Ball travel distance | `Σ√(Δx² + Δy²)` | Play complexity |

#### 2. **Temporal Features**
| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| `duration` | Play length | Seconds | 3-30 |
| `num_events` | Event count | Integer | 2+ |

#### 3. **Tactical Features**
| Feature | Description | Calculation | Use |
|---------|-------------|-------------|-----|
| `avg_attackers_ahead` | Offensive support | `mean(attackers ahead of ball)` | Formation analysis |
| `avg_defenders_ahead` | Defensive pressure | `mean(defenders ahead of ball)` | Resistance level |
| `wing_side` | Attack position | `'WING' if abs(y) > 15 else 'CENTER'` | Positional categorization |

#### 4. **Outcome Features**
| Feature | Description | Values |
|---------|-------------|--------|
| `outcome` | Final result | GOAL, SHOT, POSSESSION_LOST, LOSS, TURNOVER |
| `is_goal` | Scoring flag | Boolean |

### Feature Vector Construction

For clustering, features are normalized and combined:

```python
def get_feature_vector(play):
    """
    13-dimensional feature vector for clustering.
    """
    return np.array([
        play.delta_x,              # Forward progress
        play.delta_y,              # Lateral movement
        play.total_distance,       # Total distance
        play.duration,             # Time length
        play.num_events,           # Event count
        play.avg_attackers_ahead,  # Offensive support
        play.avg_defenders_ahead,  # Defensive pressure
        play.max_x,                # Maximum penetration
        1 if play.is_goal else 0,  # Scoring outcome
        1 if play.wing_side == 'WING' else 0,  # Position
        # ... (normalized values)
    ])
```

---

## 🎯 Clustering Algorithm

### Method: Hierarchical Clustering (`src/clustering.py`)

The system uses **Agglomerative Hierarchical Clustering** with **Ward's linkage**:

```python
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

# Calculate pairwise distances
distance_matrix = pdist(feature_matrix, metric='euclidean')

# Build linkage tree using Ward's method
linkage_matrix = linkage(distance_matrix, method='ward')

# Cut tree at threshold
cluster_labels = fcluster(linkage_matrix, 
                         t=clustering_threshold,
                         criterion='distance')
```

### Why Hierarchical Clustering?

✅ **Advantages:**
- No need to specify number of clusters beforehand
- Produces dendrogram showing hierarchical relationships
- Works well with Euclidean distance in tactical feature space
- Ward's method minimizes within-cluster variance

❌ **Limitations:**
- O(n² log n) time complexity
- Sensitive to outliers
- Can't undo merges

### Clustering Pipeline

**Step 1: Filter Plays**
```python
valid_plays = [p for p in plays if p.delta_x >= min_forward_progress]
```

**Step 2: Extract Features**
```python
feature_matrix = np.array([get_feature_vector(p) for p in valid_plays])
```

**Step 3: Cluster**
```python
cluster_labels = hierarchical_clustering(feature_matrix, threshold)
```

**Step 4: Filter Small Clusters**
```python
# Remove clusters with < 2 plays
filtered = {cid: plays for cid, plays in clusters.items() if len(plays) >= 2}
```

**Step 5: Renumber & Sort**
```python
# Sort by cluster size (largest first)
# Renumber sequentially: 1, 2, 3, ...
```

### Cluster Naming Algorithm

Clusters are automatically named based on characteristics:

```python
def generate_cluster_name(plays):
    """
    Create descriptive name from play statistics.
    
    Format: [Position] [Speed] [Depth] [Conversion]
    Example: "Wing Attack Fast Deep High-Conv"
    """
    # Position (wing_side >= 70% threshold)
    if wing_pct >= 0.7:
        position = "Wing Attack"
    elif wing_pct <= 0.3:
        position = "Central Attack"
    else:
        position = "Mixed Attack"
    
    # Speed (duration thresholds)
    if avg_duration < 5:
        speed = "Fast"
    elif avg_duration > 10:
        speed = "Slow Build"
    else:
        speed = "Medium"
    
    # Depth (forward progress)
    if avg_forward > 30:
        depth = "Deep"
    elif avg_forward > 20:
        depth = "Mid"
    else:
        depth = "Short"
    
    # Conversion (goal rate)
    if goal_rate >= 0.3:
        conversion = "High-Conv"
    elif goal_rate > 0:
        conversion = "Low-Conv"
    
    return f"{position} {speed} {depth} {conversion}"
```

**Example Names:**
- "Wing Attack Fast Deep High-Conv" → Quick wing plays that score
- "Central Attack Slow Build Mid" → Patient buildup through center
- "Mixed Attack Medium Short Low-Conv" → Varied short attacks

---

## ⚙️ Configuration Parameters

### Core Settings (`src/config.py`)

```python
@dataclass
class AnalysisConfig:
    # Play Duration Filters
    min_play_duration: float = 3.0    # Minimum seconds
    max_play_duration: float = 30.0   # Maximum seconds
    
    # Spatial Filters
    min_forward_progress: float = 5.0  # Minimum meters forward
    
    # Clustering
    clustering_threshold: float = 12.0  # Distance threshold
    
    # Position Thresholds
    ahead_threshold: float = 1.0        # Meters to count as "ahead"
    forward_threshold: float = 1.0      # Meters to count as "forward"
```

### Effect of Each Parameter

#### `min_play_duration` (default: 3.0 seconds)
- **Increase (e.g., 5.0):** 
  - ✅ Filters quick turnovers
  - ✅ Focuses on sustained attacks
  - ❌ May miss quick counter-attacks
  
- **Decrease (e.g., 1.0):**
  - ✅ Captures rapid transitions
  - ❌ Includes more noise

#### `max_play_duration` (default: 30.0 seconds)
- **Increase (e.g., 60.0):**
  - ✅ Includes long possession plays
  - ❌ May merge multiple distinct sequences
  
- **Decrease (e.g., 15.0):**
  - ✅ Focuses on direct attacks
  - ❌ Misses patient buildup

#### `min_forward_progress` (default: 5.0 meters)
- **Increase (e.g., 10.0):**
  - ✅ Only penetrating attacks
  - ❌ Misses lateral/possession plays
  
- **Decrease (e.g., 2.0):**
  - ✅ Includes all forward movement
  - ❌ More plays to cluster

#### `clustering_threshold` (default: 12.0)
- **Increase (e.g., 20.0):**
  - ✅ Fewer, broader clusters
  - ✅ Merges similar patterns
  - ❌ May group distinct tactics
  
- **Decrease (e.g., 8.0):**
  - ✅ More specific clusters
  - ✅ Finer tactical distinctions
  - ❌ More clusters to analyze

**Recommended Values by Use Case:**

| Use Case | Duration | Progress | Threshold | Result |
|----------|----------|----------|-----------|--------|
| **Counter-attacks** | 1-10s | 15m | 8.0 | Fast, direct plays |
| **Possession play** | 10-60s | 3m | 15.0 | Patient buildup |
| **General analysis** | 3-30s | 5m | 12.0 | Balanced coverage |
| **High-level patterns** | 5-45s | 8m | 20.0 | Broad categories |

---

## 🖥️ GUI Usage

### Main Window

**Controls Section:**
- **Cluster Dropdown:** Select pattern to explore
  - Format: `Cluster 1: Wing Attack Fast Deep (15 plays)`
- **List Plays:** Show all plays in cluster with details
- **Summary:** Display cluster statistics
- **Compare:** Enter two play numbers to compare
- **Cluster Threshold:** Adjust and re-analyze

**Output Section:**
- Displays results of commands
- Scrollable text area
- Monospace font for alignment

### Comparison Window

When comparing two plays, a single window opens with:

**Top Section:**
- Cluster name and similarity score (0.0-1.0)

**Middle Section (Field Plots):**
- **Left:** Play #1 in gold
- **Right:** Play #2 in blue
- Ball path with markers
- Start (circle) and end (X) positions
- Outcome icons (⚽ goal, 🎯 shot, ❌ lost)

**Bottom Section (Details):**
- **Left Panel:** Play #1 metrics and events
- **Right Panel:** Play #2 metrics and events
- Complete event sequences with:
  - Event number
  - Event type (PA, IT, DR, SH, etc.)
  - Player name
  - Ball position
  - Attackers ahead

### Keyboard Shortcuts

- **Enter:** Execute selected command
- **Tab:** Navigate between fields
- **Ctrl+C:** Copy selected text from output

---

## 📁 Code Structure

### Core Modules

#### `src/config.py` - Configuration
```python
# Global settings
analysis_config = AnalysisConfig()
path_config = PathConfig()

# Usage
from src.config import analysis_config
threshold = analysis_config.clustering_threshold
```

#### `src/models.py` - Data Models
```python
@dataclass
class PlayEvent:
    """Single event in a play."""
    event_type: str
    time: float
    ball_x: float
    ball_y: float
    attacking_players_ahead: int
    defending_players_ahead: int
    team_id: int
    player_name: Optional[str]

@dataclass
class Play:
    """Complete play sequence."""
    play_id: str
    match_id: int
    team_name: str
    events: List[PlayEvent]
    # ... 25+ fields
```

#### `src/data_loader.py` - Data Loading
```python
class EventParser:
    """Parse raw JSON events."""
    
class PlayExtractor:
    """Extract plays from events."""
    def extract_plays(events, metadata) -> List[Play]
    def _try_extract_play(events, start_idx) -> Optional[tuple]
    
class DataLoader:
    """Load all JSON files."""
    def load_all_matches() -> tuple[List[Play], Dict]
```

#### `src/feature_engineer.py` - Feature Engineering
```python
class FeatureEngineer:
    """Calculate play features."""
    def engineer_features(plays: List[Play]) -> List[Play]
    def _calculate_spatial_features(play: Play)
    def _calculate_tactical_features(play: Play)
```

#### `src/clustering.py` - Clustering
```python
class PlayClusterer:
    """Cluster plays by pattern."""
    def cluster_plays(plays) -> OrderedDict
    def calculate_similarity(play1, play2) -> float

class ClusterAnalyzer:
    """Analyze clusters."""
    def analyze_clusters(clusters) -> Dict
    def _generate_cluster_name(plays) -> str
```

#### `src/visualizer.py` - Visualization
```python
class FieldVisualizer:
    """Draw plays on fields."""
    def draw_field(ax)
    def plot_play(ax, play, color)
    def compare_plays(play1, play2) -> Figure

class ComparisonPrinter:
    """Print comparison tables."""
    def print_comparison(play1, play2, similarity)
```

#### `src/browser.py` - Interactive Browser
```python
class PlayBrowser:
    """Browse and compare plays."""
    def list()              # List all plays
    def compare(n1, n2)     # Compare two plays
    def summary()           # Show statistics
```

#### `src/main.py` - Main Pipeline
```python
class TacticalAnalyzer:
    """Main analysis pipeline."""
    def run_analysis() -> dict
    def reanalyze_with_threshold(threshold) -> dict
    def create_browser(cluster_id) -> PlayBrowser
    def save_results()
```

---

## 🔧 Customization Guide

### Adding New Features

**In `src/feature_engineer.py`:**

```python
def _calculate_custom_feature(self, play: Play):
    """Add your custom feature."""
    # Example: Calculate average pass distance
    pass_distances = []
    for i in range(len(play.events) - 1):
        if play.events[i].event_type == 'PA':
            dx = play.events[i+1].ball_x - play.events[i].ball_x
            dy = play.events[i+1].ball_y - play.events[i].ball_y
            dist = np.sqrt(dx**2 + dy**2)
            pass_distances.append(dist)
    
    play.avg_pass_distance = np.mean(pass_distances) if pass_distances else 0
```

**In `src/utils.py`:**

```python
def get_feature_vector(play: Play) -> np.ndarray:
    """Add to feature vector."""
    return np.array([
        # ... existing features ...
        play.avg_pass_distance,  # NEW FEATURE
    ])
```

### Adding New Event Types

**In `src/data_loader.py`:**

```python
# Add to terminal events list
if event_type in ['SH', 'LO', 'CA', 'TA', 'NEW_EVENT']:
    # Handle new terminal event
```

### Changing Cluster Naming

**In `src/clustering.py`:**

```python
def _generate_cluster_name(self, plays: List[Play]) -> str:
    """Custom naming logic."""
    # Add your own naming criteria
    if avg_passes > 5:
        return "Long Possession Pattern"
    # ...
```

### Custom Distance Metrics

**In `src/clustering.py`:**

```python
# Replace Euclidean with custom metric
from scipy.spatial.distance import pdist

def custom_distance(u, v):
    # Your custom distance calculation
    return np.sum(np.abs(u - v))  # Manhattan distance example

distance_matrix = pdist(feature_matrix, metric=custom_distance)
```

---

## 📊 Algorithm Tuning

### Optimizing Cluster Quality

#### Problem: Too Many Small Clusters
**Solution:**
- ✅ Increase `clustering_threshold` (e.g., 15.0 → 20.0)
- ✅ Increase `min_forward_progress` (filter more plays)
- ✅ Reduce feature dimensionality (fewer features)

#### Problem: Clusters Too Broad
**Solution:**
- ✅ Decrease `clustering_threshold` (e.g., 12.0 → 8.0)
- ✅ Add more discriminative features
- ✅ Use different linkage method (e.g., 'complete' instead of 'ward')

#### Problem: No Goals in Any Cluster
**Solution:**
- ✅ Check `min_forward_progress` (may be too high)
- ✅ Verify goal events in source data
- ✅ Lower duration thresholds to capture quick goals

### Performance Optimization

#### For Large Datasets (10,000+ plays)

**Option 1: Faster Clustering**
```python
# Use mini-batch k-means instead
from sklearn.cluster import MiniBatchKMeans

kmeans = MiniBatchKMeans(n_clusters=20, batch_size=1000)
labels = kmeans.fit_predict(feature_matrix)
```

**Option 2: Dimensionality Reduction**
```python
from sklearn.decomposition import PCA

# Reduce 13 features to 5
pca = PCA(n_components=5)
reduced_features = pca.fit_transform(feature_matrix)
```

**Option 3: Sampling**
```python
# Analyze subset of plays
import random
sample_plays = random.sample(all_plays, 5000)
```

### Validation Techniques

#### Silhouette Score (Cluster Quality)
```python
from sklearn.metrics import silhouette_score

score = silhouette_score(feature_matrix, cluster_labels)
# Score: -1 (poor) to 1 (excellent)
# Good clusters: > 0.5
```

#### Elbow Method (Optimal Threshold)
```python
# Try different thresholds
thresholds = [5, 10, 15, 20, 25]
for t in thresholds:
    clusters = cluster_plays(plays, threshold=t)
    print(f"Threshold {t}: {len(clusters)} clusters")
```

---

## 📤 Output Files

### Generated Files (in `output/`)

#### `all_plays.csv`
Complete play database with all features:
```csv
play_id,match_id,team_name,duration,delta_x,delta_y,num_events,outcome,cluster_id
M3812_T1_T123,3812,Poland,5.2,12.3,-3.4,4,SHOT,1
...
```

#### `cluster_analysis.csv`
Cluster summaries:
```csv
cluster_id,name,total,goals,shots,avg_duration,avg_forward
1,Wing Attack Fast Deep,15,3,8,4.2,18.5
...
```

#### `cluster_summary.csv`
Statistical overview:
```csv
metric,value
total_plays,247
total_clusters,8
avg_plays_per_cluster,30.9
...
```

#### `detailed_clusters.json`
Full cluster details in JSON:
```json
{
  "1": {
    "name": "Wing Attack Fast Deep",
    "plays": [...],
    "statistics": {...}
  }
}
```

---

## 📦 Requirements

### Python Version
- **Python 3.8+** required
- Tested on Python 3.9, 3.10, 3.11

### Dependencies (`requirements.txt`)

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scipy>=1.7.0
```

**Optional:**
```txt
scikit-learn>=0.24.0  # For advanced clustering
seaborn>=0.11.0       # For enhanced visualizations
```

### Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🎓 Technical Details

### Coordinate System

**Field Dimensions:**
- Length: 105 meters
- Width: 68 meters
- Origin: Center of field (0, 0)
- X-axis: Length (-52.5 to +52.5)
- Y-axis: Width (-34 to +34)

**Normalization:**
All plays normalized to attack left-to-right regardless of actual direction.

### Event Types Reference

| Code | Event Type | Description |
|------|------------|-------------|
| PA | Pass | Player passes ball to teammate |
| CR | Cross | Cross from wing into box |
| SH | Shot | Shot at goal |
| IT | Interception/Touch | Ball touched/intercepted |
| DR | Dribble | Player dribbles with ball |
| LO | Loss | Possession lost |
| CA | Clearance | Defensive clearance |
| TA | Tackle | Defensive tackle |
| TC | Touch | General touch |
| CH | Challenge | Challenge for ball |

### Similarity Calculation

```python
def calculate_similarity(play1, play2) -> float:
    """
    Calculate similarity score between plays.
    
    Returns: 0.0 (very different) to 1.0 (identical)
    """
    vec1 = get_feature_vector(play1)
    vec2 = get_feature_vector(play2)
    
    # Euclidean distance
    distance = np.linalg.norm(vec1 - vec2)
    
    # Normalize to 0-1 similarity
    max_distance = 100.0
    similarity = 1.0 - min(distance / max_distance, 1.0)
    
    return similarity
```

---

## 🐛 Troubleshooting

### Common Issues

#### "No plays extracted"
- ✅ Check JSON format matches StatsBomb schema
- ✅ Verify `min_forward_progress` not too high
- ✅ Check duration thresholds

#### "Only 1 cluster found"
- ✅ Decrease `clustering_threshold`
- ✅ Check feature variance (may need more features)
- ✅ Verify plays have diverse characteristics

#### GUI not responding
- ✅ Check for long-running analysis (large datasets)
- ✅ Verify matplotlib backend compatibility
- ✅ Try running analysis in command line first

#### Field plots not showing
- ✅ Update matplotlib: `pip install -U matplotlib`
- ✅ Check TkAgg backend: `matplotlib.use('TkAgg')`

---

## 📈 Future Enhancements

### Potential Improvements

1. **Machine Learning**
   - Neural network embeddings for plays
   - Supervised classification by play type
   - Outcome prediction models

2. **Advanced Clustering**
   - DBSCAN for arbitrary shapes
   - HDBSCAN for hierarchical density
   - Fuzzy clustering for overlapping patterns

3. **Extended Features**
   - Player formation analysis
   - Passing network metrics
   - Pressure indicators
   - Space creation metrics

4. **Visualization**
   - 3D plots of feature space
   - Interactive web dashboard
   - Animation of play sequences
   - Heatmaps of player positions

5. **Analysis Tools**
   - Compare across matches
   - Team tactical signatures
   - Evolution of patterns over season
   - Success rate prediction

