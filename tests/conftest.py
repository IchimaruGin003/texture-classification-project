"""
Pytest configuration file - sets up Python path for all tests
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

# Verify imports work
try:
    from src.models.knn_trainer import KNNTrainer

    print("✓ KNNTrainer import successful in conftest")
except ImportError as e:
    print(f"✗ KNNTrainer import failed in conftest: {e}")
