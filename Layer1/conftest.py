# Layer1/conftest.py
import sys
from pathlib import Path

# Add Layer1/ to the Python path so tests can import schema, agents, etc.
sys.path.insert(0, str(Path(__file__).parent))