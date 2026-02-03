"""
Launch the Tactical Analysis GUI.
Simple double-click to run.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui_app import main

if __name__ == '__main__':
    print("Starting Football Tactical Analysis GUI...")
    print("Please wait while data is loading...\n")
    main()
