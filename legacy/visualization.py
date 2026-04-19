"""
Task 3: Visualization
Generates 4-panel visualization of simulation results.

Panels:
1. SoCal Coast Map - Geographic context
2. Environmental Variables - Nutrients, oxygen, temp, habitat, fishing over time
3. Ecosystem Populations - 6 functional groups over time
4. Impact Summary - Winners/Losers/Stable
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
from typing import Dict, List
from simulation_engine import SimulationResult


# Color scheme (professional, accessible)
COLORS = {
    'nutrient': '#3b82f6',      # Blue
    'oxygen': '#10b981',         # Green
    'temperature': '#f59e0b',    # Amber
    'habitat': '#14b8a6',        # Teal
    'fishing': '#ef4444',        # Red
    'phytoplankton': '#86efac',  # Light green
    'zooplankton': '#4ade80',    # Medium green
    'forage_fish': '#3b82f6',    # Blue
    'predator_fish': '#1e40af',  # Dark blue
    'seabirds': '#f97316',       # Orange
    'marine_mammals': '#a855f7', # Purple
    'baseline': '#6b7280',       # Gray
    'policy_line': '#dc2626',    # Red dashed
}


def plot_socal_map(ax):
    """
    Panel 1: Simple map of Southern California coast
    Shows policy impact zone
    """
    # Simple coastline outline (rough coordinates)
    # From San Diego (32.7°N) to Point Conception (34.4°N)
    coastline_lat = [32.7, 32.9, 33.1, 33.4, 33.7, 34.0, 34.2, 34.4]
    coastline_lon = [-117.2, -117.3, -117.9, -118.4, -118.5, -119.5, -120.0, -120.5]
    
    # Plot coastline
    ax.plot(coastline_lon, coastline_lat, 'k-', linewidth=2, label='California Coast')
    
    # Shade policy impact zone (CalCOFI lines 80-93 coverage)
    impact_zone_lon = [-117.2, -120.5, -120.8, -117.5, -117.2]
    impact_zone_lat = [32.7, 34.4, 34.6, 32.9, 32.7]
    ax.fill(impact_zone_lon, impact_zone_lat, alpha=0.2, color=COLORS['habitat'], 
            label='Policy Impact Zone')
    
    # Add city markers
    cities = {
        'San Diego': (-117.16, 32.72),
        'Los Angeles': (-118.24, 34.05),
        'Santa Barbara': (-119.70, 34.42)
    }
    
    for city, (lon, lat) in cities.items():
        ax.plot(lon, lat, 'ko', markersize=6)
        ax.text(lon - 0.3, lat + 0.1, city, fontsize=9)
    
    # Add CalCOFI station representation
    # Simplified grid of stations
    station_lons = np.linspace(-117.5, -120.0, 8)
    station_lats = np.linspace(32.9, 34.3, 6)
    for slon in station_lons:
        for slat in station_lats:
            ax.plot(slon, slat, 'o', color=COLORS['baseline'], 
                   markersize=3, alpha=0.3)
    
    ax.set_xlim(-121, -116.5)
    ax.set_ylim(32.5, 34.7)
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_title('Southern California Coastal Ecosystem\nCalCOFI Study Region', 
                fontsize=11, fontweight='bold', pad=10)
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')


def plot_environmental_variables(ax, timeline: List[Dict]):
    """
    Panel 2: Environmental variables over time
    """
    years = [state['year'] for state in timeline]
    
    variables = {
        'Nutrient Load': ('nutrient_load', COLORS['nutrient']),
        'Dissolved Oxygen': ('dissolved_oxygen', COLORS['oxygen']),
        'Water Temperature': ('water_temperature', COLORS['temperature']),
        'Habitat Quality': ('habitat_quality', COLORS['habitat']),
        'Fishing Pressure': ('fishing_pressure', COLORS['fishing']),
    }
    
    for label, (key, color) in variables.items():
        values = [state[key] for state in timeline]
        ax.plot(years, values, linewidth=2, label=label, color=color)
    
    # Add baseline reference line
    ax.axhline(y=0.5, color=COLORS['baseline'], linestyle='--', 
              linewidth=1, alpha=0.5, label='Baseline (pre-policy)')
    
    # Mark policy application point
    ax.axvline(x=0, color=COLORS['policy_line'], linestyle=':', 
              linewidth=1.5, alpha=0.7, label='Policy Applied')
    
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel('Normalized Value (0-1 scale)', fontsize=10)
    ax.set_title('Environmental Variables Over Time', 
                fontsize=11, fontweight='bold', pad=10)
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 1)


def plot_ecosystem_populations(ax, timeline: List[Dict]):
    """
    Panel 3: Ecosystem population dynamics
    """
    years = [state['year'] for state in timeline]
    
    agents = {
        'Phytoplankton': ('phytoplankton', COLORS['phytoplankton']),
        'Zooplankton': ('zooplankton', COLORS['zooplankton']),
        'Forage Fish': ('forage_fish', COLORS['forage_fish']),
        'Predator Fish': ('predator_fish', COLORS['predator_fish']),
        'Seabirds': ('seabirds', COLORS['seabirds']),
        'Marine Mammals': ('marine_mammals', COLORS['marine_mammals']),
    }
    
    for label, (key, color) in agents.items():
        values = [state[key] for state in timeline]
        ax.plot(years, values, linewidth=2, label=label, color=color)
    
    # Add baseline reference line
    ax.axhline(y=0.5, color=COLORS['baseline'], linestyle='--', 
              linewidth=1, alpha=0.5, label='Baseline')
    
    # Mark policy application point
    ax.axvline(x=0, color=COLORS['policy_line'], linestyle=':', 
              linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel('Population Index (0-1 scale)', fontsize=10)
    ax.set_title('Ecosystem Population Dynamics', 
                fontsize=11, fontweight='bold', pad=10)
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 1)


def plot_impact_summary(ax, impacts: Dict[str, str], summary: str, timeline: List[Dict]):
    """
    Panel 4: Winners/Losers summary with visual cards
    """
    ax.axis('off')
    
    # Calculate actual percentage changes
    baseline = {
        'phytoplankton': 0.5,
        'zooplankton': 0.5,
        'forage_fish': 0.5,
        'predator_fish': 0.5,
        'seabirds': 0.5,
        'marine_mammals': 0.5,
    }
    
    final_state = timeline[-1]
    changes = {
        agent: ((final_state[agent] - baseline[agent]) / baseline[agent] * 100)
        for agent in baseline.keys()
    }
    
    # Group by impact
    winners = [(a.replace('_', ' ').title(), changes[a]) 
               for a, impact in impacts.items() if impact == 'increase']
    losers = [(a.replace('_', ' ').title(), changes[a]) 
              for a, impact in impacts.items() if impact == 'decrease']
    stable = [(a.replace('_', ' ').title(), changes[a]) 
              for a, impact in impacts.items() if impact == 'stable']
    
    # Sort by magnitude
    winners.sort(key=lambda x: x[1], reverse=True)
    losers.sort(key=lambda x: x[1])
    
    # Title
    ax.text(0.5, 0.95, 'Ecosystem Impact Summary (10-Year Projection)', 
           ha='center', va='top', fontsize=12, fontweight='bold',
           transform=ax.transAxes)
    
    # Draw three columns
    col_width = 0.3
    col_starts = [0.05, 0.37, 0.69]
    colors_bg = ['#d1fae5', '#fee2e2', '#f3f4f6']  # Light green, light red, light gray
    titles = ['🟢 Winners (Population Increase)', '🔴 Losers (Population Decrease)', '⚪ Stable']
    data_lists = [winners, losers, stable]
    
    for col_x, bg_color, title, data in zip(col_starts, colors_bg, titles, data_lists):
        # Column background
        rect = FancyBboxPatch((col_x, 0.05), col_width, 0.75,
                              boxstyle="round,pad=0.01", 
                              transform=ax.transAxes,
                              facecolor=bg_color, edgecolor='gray', 
                              linewidth=0.5, alpha=0.3)
        ax.add_patch(rect)
        
        # Column title
        ax.text(col_x + col_width/2, 0.77, title, 
               ha='center', va='top', fontsize=9, fontweight='bold',
               transform=ax.transAxes)
        
        # List items
        y_start = 0.70
        y_step = 0.08
        for i, (agent, pct_change) in enumerate(data):
            y_pos = y_start - i * y_step
            
            # Agent name
            ax.text(col_x + 0.02, y_pos, agent, 
                   ha='left', va='top', fontsize=8,
                   transform=ax.transAxes)
            
            # Percentage change
            sign = '+' if pct_change > 0 else ''
            color = '#059669' if pct_change > 0 else '#dc2626' if pct_change < 0 else '#6b7280'
            ax.text(col_x + col_width - 0.02, y_pos, f'{sign}{pct_change:.0f}%', 
                   ha='right', va='top', fontsize=8, fontweight='bold',
                   color=color, transform=ax.transAxes)
    
    # Summary text at bottom
    # Wrap text if too long
    import textwrap
    wrapped_summary = textwrap.fill(summary, width=80)
    ax.text(0.5, 0.02, wrapped_summary,
           ha='center', va='bottom', fontsize=9, style='italic',
           transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))


def generate_visualization(result: SimulationResult, 
                          policy_description: str = "",
                          output_path: str = "ecosystem_simulation.png"):
    """
    Generate complete 4-panel visualization.
    
    Args:
        result: SimulationResult from Task 2
        policy_description: Text description of the policy
        output_path: Where to save the figure
    """
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Custom layout: 2x2 grid with different sizes
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3,
                         left=0.08, right=0.95, top=0.93, bottom=0.07)
    
    ax1 = fig.add_subplot(gs[0, 0])  # Top left: Map
    ax2 = fig.add_subplot(gs[0, 1])  # Top right: Environmental vars
    ax3 = fig.add_subplot(gs[1, 0])  # Bottom left: Populations
    ax4 = fig.add_subplot(gs[1, 1])  # Bottom right: Summary
    
    # Generate each panel
    plot_socal_map(ax1)
    plot_environmental_variables(ax2, result['timeline'])
    plot_ecosystem_populations(ax3, result['timeline'])
    plot_impact_summary(ax4, result['impacts'], result['summary'], result['timeline'])
    
    # Main title
    if policy_description:
        fig.suptitle(f'EcoPolicy Simulator: {policy_description}', 
                    fontsize=14, fontweight='bold', y=0.97)
    else:
        fig.suptitle('EcoPolicy Simulator: Southern California Coastal Ecosystem', 
                    fontsize=14, fontweight='bold', y=0.97)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Visualization saved to: {output_path}")
    
    return fig


# Example usage
if __name__ == "__main__":
    from task2_simulation_engine import run_simulation
    
    print("=" * 70)
    print("TASK 3: VISUALIZATION TEST")
    print("=" * 70)
    
    # Test policy
    test_levers = {
        "nutrient_load": -0.3,
        "habitat_quality": 0.1,
    }
    
    policy_desc = "30% Agricultural Runoff Reduction + Habitat Restoration"
    
    print(f"\nPolicy: {policy_desc}")
    print(f"Levers: {test_levers}")
    print("\nRunning 10-year simulation...")
    
    # Run simulation
    result = run_simulation(test_levers, years=10)
    
    print(f"\nGenerating visualization...")
    
    # Generate visualization
    fig = generate_visualization(result, policy_desc, "ecosystem_simulation.png")
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print("\nOpen 'ecosystem_simulation.png' to view results.")
    print("\nSummary:", result['summary'])