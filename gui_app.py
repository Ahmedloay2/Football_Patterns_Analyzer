"""
Simple GUI for Football Tactical Pattern Analysis.
Select cluster and run browser commands with visual output.
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import sys
from io import StringIO

from src.main import TacticalAnalyzer


class TacticalAnalysisGUI:
    """Simple GUI for tactical analysis."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Football Tactical Pattern Analysis")
        self.root.geometry("1200x800")
        
        self.analyzer = None
        self.browser = None
        self.current_cluster = None
        
        self.setup_ui()
        self.load_data()
    
    def setup_ui(self):
        """Setup the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Top panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Cluster selection
        ttk.Label(control_frame, text="Select Cluster:").grid(row=0, column=0, padx=5)
        self.cluster_var = tk.StringVar()
        self.cluster_combo = ttk.Combobox(control_frame, textvariable=self.cluster_var, 
                                         state='readonly', width=80)
        self.cluster_combo.grid(row=0, column=1, padx=5)
        self.cluster_combo.bind('<<ComboboxSelected>>', self.on_cluster_selected)
        
        # Command buttons
        ttk.Button(control_frame, text="List Plays", 
                  command=self.cmd_list).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Summary", 
                  command=self.cmd_summary).grid(row=0, column=3, padx=5)
        
        # Compare inputs
        ttk.Label(control_frame, text="Compare:").grid(row=1, column=0, padx=5, pady=5)
        self.play1_var = tk.StringVar(value="1")
        self.play2_var = tk.StringVar(value="2")
        
        ttk.Entry(control_frame, textvariable=self.play1_var, 
                 width=15).grid(row=1, column=1, padx=5, sticky=tk.W)
        ttk.Label(control_frame, text="vs").grid(row=1, column=1, padx=50)
        ttk.Entry(control_frame, textvariable=self.play2_var, 
                 width=15).grid(row=1, column=1, padx=5, sticky=tk.E)
        
        ttk.Button(control_frame, text="Compare Plays", 
                  command=self.cmd_compare).grid(row=1, column=2, padx=5)
        
        # Threshold adjustment
        ttk.Label(control_frame, text="Cluster Threshold:").grid(row=2, column=0, padx=5, pady=5)
        self.threshold_var = tk.StringVar(value="12.0")
        ttk.Entry(control_frame, textvariable=self.threshold_var, 
                 width=10).grid(row=2, column=1, padx=5, sticky=tk.W)
        ttk.Button(control_frame, text="Re-analyze", 
                  command=self.cmd_reanalyze).grid(row=2, column=2, padx=5)
        
        # Output panel - Text output
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="10")
        output_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        
        # Scrolled text widget with better formatting
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.NONE, 
                                                     font=('Consolas', 9), width=120, height=30)
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
    
    def load_data(self):
        """Load and analyze data."""
        self.print_output("Loading data and running analysis...\n")
        self.status_var.set("Loading data...")
        self.root.update()
        
        try:
            # Create analyzer and run analysis
            self.analyzer = TacticalAnalyzer()
            results = self.analyzer.run_analysis()
            
            # Populate cluster dropdown
            cluster_options = []
            for cluster_id, analysis in self.analyzer.cluster_analysis.items():
                total = analysis['total']
                name = analysis['name']
                cluster_options.append(f"Cluster {cluster_id}: {name} ({total} plays)")
            
            self.cluster_combo['values'] = cluster_options
            if cluster_options:
                self.cluster_combo.current(0)
                self.on_cluster_selected(None)
            
            self.print_output("\n✅ Analysis complete!\n")
            self.print_output(f"Total plays extracted: {self.analyzer.total_extracted_plays}\n")
            self.print_output(f"Found {len(self.analyzer.clusters)} tactical patterns\n\n")
            self.status_var.set("Ready - Select a cluster and run commands")
            
        except Exception as e:
            self.print_output(f"\n❌ Error: {str(e)}\n")
            self.status_var.set("Error loading data")
            messagebox.showerror("Error", f"Failed to load data:\n{str(e)}")
    
    def on_cluster_selected(self, event):
        """Handle cluster selection."""
        selection = self.cluster_var.get()
        if not selection:
            return
        
        # Extract cluster number
        cluster_id = int(selection.split(':')[0].replace('Cluster ', ''))
        self.current_cluster = cluster_id
        
        # Create browser
        self.browser = self.analyzer.create_browser(cluster_id)
        
        # Get cluster name
        cluster_name = self.analyzer.cluster_analysis.get(cluster_id, {}).get('name', 'Unknown')
        
        # Show initial info
        self.clear_output()
        self.print_output(f"Selected: Cluster {cluster_id}: {cluster_name}\n")
        self.print_output("=" * 80 + "\n")
        self.status_var.set(f"Cluster {cluster_id}: {cluster_name} selected")
    
    def cmd_list(self):
        """Execute list command."""
        if not self.browser:
            messagebox.showwarning("Warning", "Please select a cluster first")
            return
        
        self.clear_output()
        self.status_var.set("Listing plays...")
        
        # Capture output
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        
        try:
            self.browser.list()
            output = mystdout.getvalue()
            self.print_output(output)
            self.status_var.set("List complete")
        except Exception as e:
            self.print_output(f"❌ Error: {str(e)}\n")
            self.status_var.set("Error")
        finally:
            sys.stdout = old_stdout
    
    def cmd_summary(self):
        """Execute summary command."""
        if not self.browser:
            messagebox.showwarning("Warning", "Please select a cluster first")
            return
        
        self.clear_output()
        self.status_var.set("Generating summary...")
        
        # Capture output
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        
        try:
            self.browser.summary()
            output = mystdout.getvalue()
            self.print_output(output)
            self.status_var.set("Summary complete")
        except Exception as e:
            self.print_output(f"❌ Error: {str(e)}\n")
            self.status_var.set("Error")
        finally:
            sys.stdout = old_stdout
    
    def cmd_compare(self):
        """Execute compare command - single window with both plays."""
        if not self.browser:
            messagebox.showwarning("Warning", "Please select a cluster first")
            return
        
        try:
            play1_idx = int(self.play1_var.get()) - 1
            play2_idx = int(self.play2_var.get()) - 1
        except ValueError:
            messagebox.showerror("Error", "Please enter valid play numbers")
            return
        
        if not (0 <= play1_idx < len(self.browser.plays)):
            messagebox.showerror("Error", f"Play {play1_idx + 1} not found (valid range: 1-{len(self.browser.plays)})")
            return
        
        if not (0 <= play2_idx < len(self.browser.plays)):
            messagebox.showerror("Error", f"Play {play2_idx + 1} not found (valid range: 1-{len(self.browser.plays)})")
            return
        
        self.status_var.set(f"Comparing plays {play1_idx + 1} vs {play2_idx + 1}...")
        
        # Get plays
        play1 = self.browser.plays[play1_idx]
        play2 = self.browser.plays[play2_idx]
        
        # Calculate similarity
        similarity = self.browser.clusterer.calculate_similarity(play1, play2)
        
        # Create single comparison window
        compare_window = tk.Toplevel(self.root)
        compare_window.title(f"Play Comparison: #{play1_idx + 1} vs #{play2_idx + 1}")
        compare_window.geometry("1400x820")
        compare_window.configure(bg='#1a1a1a')
        
        # Main container
        main_frame = ttk.Frame(compare_window)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title
        cluster_name = self.analyzer.cluster_analysis.get(self.current_cluster, {}).get('name', 'Unknown')
        title_label = tk.Label(
            main_frame,
            text=f"PATTERN: {cluster_name} (Cluster #{self.current_cluster}) | Similarity: {similarity:.3f}",
            font=('Arial', 12, 'bold'),
            bg='#1a1a1a',
            fg='white',
            pady=8
        )
        title_label.pack()
        
        # Field plots frame (side by side)
        fields_frame = ttk.Frame(main_frame)
        fields_frame.pack(fill='x', pady=5)
        
        # Create matplotlib figure with 2 subplots
        fig = Figure(figsize=(13, 4), facecolor='#1a1a1a')
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        # Plot play 1
        self.browser.visualizer.draw_field(ax1, field_color='#2d8f2d')
        self.browser.visualizer.plot_play(ax1, play1, color='#FFD700', alpha=0.95, show_annotations=True)
        outcome1 = 'GOAL' if play1.is_goal else 'SHOT' if play1.outcome == 'SHOT' else 'LOST'
        ax1.set_title(f'PLAY #{play1_idx + 1} (GOLD) - {outcome1}\n{play1.team_name}\n'
                     f'{play1.duration:.1f}s | {play1.num_events} events | {play1.delta_x:.1f}m fwd',
                     fontsize=8, fontweight='bold', pad=8, color='white',
                     bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.9))
        
        # Plot play 2
        self.browser.visualizer.draw_field(ax2, field_color='#2d8f2d')
        self.browser.visualizer.plot_play(ax2, play2, color='#4169E1', alpha=0.95, show_annotations=True)
        outcome2 = 'GOAL' if play2.is_goal else 'SHOT' if play2.outcome == 'SHOT' else 'LOST'
        ax2.set_title(f'PLAY #{play2_idx + 1} (BLUE) - {outcome2}\n{play2.team_name}\n'
                     f'{play2.duration:.1f}s | {play2.num_events} events | {play2.delta_x:.1f}m fwd',
                     fontsize=8, fontweight='bold', pad=8, color='white',
                     bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.9))
        
        fig.tight_layout()
        
        # Embed fields
        canvas = FigureCanvasTkAgg(fig, master=fields_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
        
        # Details frame (side by side)
        details_container = ttk.Frame(main_frame)
        details_container.pack(fill='both', expand=True, pady=5)
        
        # Left details (Play 1)
        left_frame = ttk.LabelFrame(details_container, text=f"Play #{play1_idx + 1} Details", padding=5)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        details1_text = tk.Text(
            left_frame,
            wrap=tk.NONE,
            font=('Consolas', 10),
            bg='#2a2a2a',
            fg='white',
            height=18,
            width=65
        )
        details1_text.pack(fill='both', expand=True)
        
        details1_content = self._build_compact_play_details(play1, play1_idx + 1)
        details1_text.insert('1.0', details1_content)
        details1_text.configure(state='disabled')
        
        # Right details (Play 2)
        right_frame = ttk.LabelFrame(details_container, text=f"Play #{play2_idx + 1} Details", padding=5)
        right_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        details2_text = tk.Text(
            right_frame,
            wrap=tk.NONE,
            font=('Consolas', 10),
            bg='#2a2a2a',
            fg='white',
            height=18,
            width=65
        )
        details2_text.pack(fill='both', expand=True)
        
        details2_content = self._build_compact_play_details(play2, play2_idx + 1)
        details2_text.insert('1.0', details2_content)
        details2_text.configure(state='disabled')
        
        self.status_var.set("Comparison complete")
    
    def _build_compact_play_details(self, play, play_num):
        """Build compact detailed text for a single play."""
        lines = []
        lines.append(f"{play.team_name} - {play.outcome} {'[GOAL]' if play.is_goal else ''}")
        lines.append("=" * 62)
        lines.append(f"Match: {play.match_name[:45]}")
        lines.append(f"Time: {play.time_range_display} ({play.duration:.1f}s)")
        lines.append("")
        lines.append("METRICS:")
        lines.append(f"  Fwd: {play.delta_x:.1f}m  Lat: {play.delta_y:.1f}m  Total: {play.total_distance:.1f}m")
        lines.append(f"  Events: {play.num_events}  Position: {play.wing_side}")
        lines.append(f"  Att Ahead: {play.avg_attackers_ahead:.1f}  Def: {play.avg_defenders_ahead:.1f}")
        lines.append("")
        lines.append("EVENTS:")
        lines.append("-" * 62)
        lines.append(f"{'#':<2} {'Type':<8} {'Player':<24} {'Pos':<13} {'Ah':<4}")
        lines.append("-" * 62)
        
        for i, event in enumerate(play.events, 1):
            player = event.player_name if event.player_name else "Unknown"
            if len(player) > 23:
                player = player[:20] + "..."
            position = f"({event.ball_x:4.1f},{event.ball_y:4.1f})"
            ahead = f"{event.attacking_players_ahead}"
            lines.append(f"{i:<2} {event.event_type:<8} {player:<24} {position:<13} {ahead:<4}")
        
        lines.append("=" * 62)
        
        return '\n'.join(lines)
    
    def print_output(self, text):
        """Print text to output widget."""
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.root.update()
    
    def cmd_reanalyze(self):
        """Re-run clustering with new threshold."""
        if not self.analyzer:
            messagebox.showwarning("Warning", "Load data first")
            return
        
        try:
            threshold = float(self.threshold_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid threshold number")
            return
        
        self.clear_output()
        self.print_output(f"Re-analyzing with threshold = {threshold}...\n")
        self.status_var.set("Re-analyzing...")
        self.root.update()
        
        # Capture output
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        
        try:
            # Re-run clustering
            self.analyzer.reanalyze_with_threshold(threshold)
            
            # Get output
            output = mystdout.getvalue()
            self.print_output(output)
            
            # Update cluster dropdown
            cluster_options = []
            for cluster_id, analysis in self.analyzer.cluster_analysis.items():
                total = analysis['total']
                name = analysis['name']
                cluster_options.append(f"Cluster {cluster_id}: {name} ({total} plays)")
            
            self.cluster_combo['values'] = cluster_options
            if cluster_options:
                self.cluster_combo.current(0)
                self.on_cluster_selected(None)
            
            self.print_output("\n✅ Re-analysis complete!\n")
            self.print_output(f"Found {len(self.analyzer.clusters)} tactical patterns\n\n")
            self.status_var.set("Ready - Select a cluster and run commands")
            
        except Exception as e:
            self.print_output(f"\n❌ Error: {str(e)}\n")
            self.status_var.set("Error")
            messagebox.showerror("Error", f"Re-analysis failed:\n{str(e)}")
        finally:
            sys.stdout = old_stdout
    
    def clear_output(self):
        """Clear output widget."""
        self.output_text.delete(1.0, tk.END)
    
    def run(self):
        """Start the GUI."""
        self.root.mainloop()


def main():
    """Run the GUI application."""
    app = TacticalAnalysisGUI()
    app.run()


if __name__ == '__main__':
    main()
