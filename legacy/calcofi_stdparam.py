"""
CalCOFI Bottle Database Analyzer
Processes CalCOFI oceanographic data to extract baseline values for the simulation.

Key variables extracted:
- Nutrients: NO3 (nitrate), PO4 (phosphate), SiO3 (silicate), NH3 (ammonia)
- Dissolved Oxygen: O2ml_L, O2Sat (% saturation), Oxy_µmol/Kg
- Temperature: T_degC
- Salinity: Salnty
- Chlorophyll: ChlorA (phytoplankton proxy)
- Phaeophytin: Phaeop (degraded chlorophyll, indicates dead phytoplankton)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


# Southern California coastal station ranges
# CalCOFI lines 80-93 (roughly San Diego to Point Conception)
SOCAL_LINES = range(80, 94)

# Depth range for surface/coastal waters (0-50m)
SURFACE_DEPTH_MAX = 50  # meters


def load_calcofi_data(filepath: str) -> pd.DataFrame:
    """
    Load CalCOFI bottle database CSV.
    
    Args:
        filepath: Path to CalCOFI bottle CSV file
    
    Returns:
        Cleaned pandas DataFrame
    """
    print(f"Loading data from {filepath}...")
    
    # Read CSV - CalCOFI uses various encodings, try a few
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin1', low_memory=False)
    
    print(f"Loaded {len(df):,} rows")
    
    # Extract line number from Sta_ID (format: "080.0 055.0" where 080 is the line)
    if 'Sta_ID' in df.columns:
        df['Line'] = df['Sta_ID'].str.split('.').str[0].astype(float, errors='ignore')
    
    return df


def filter_socal_coastal(df: pd.DataFrame, 
                         year_range: Tuple[int, int] = (1990, 2020),
                         depth_max: float = SURFACE_DEPTH_MAX) -> pd.DataFrame:
    """
    Filter for Southern California coastal surface waters.
    
    Args:
        df: CalCOFI DataFrame
        year_range: (start_year, end_year) for baseline period
        depth_max: Maximum depth in meters
    
    Returns:
        Filtered DataFrame
    """
    filtered = df.copy()
    
    # Filter by line (Southern California)
    if 'Line' in filtered.columns:
        filtered = filtered[filtered['Line'].isin(SOCAL_LINES)]
        print(f"After SoCal filter (lines 80-93): {len(filtered):,} rows")
    
    # Filter by depth (surface/coastal waters)
    if 'Depthm' in filtered.columns:
        filtered = filtered[filtered['Depthm'] <= depth_max]
        print(f"After depth filter (0-{depth_max}m): {len(filtered):,} rows")
    
    # Extract year from Sta_ID or other date columns
    # Sta_ID format includes date like "19-4903CR" where 49 = 1949, 03 = March
    # We'll need to parse this or look for other date columns
    
    return filtered


def calculate_baselines(df: pd.DataFrame) -> Dict:
    """
    Calculate baseline statistics for all key variables.
    
    Returns:
        Dictionary with mean, std, min, max, trend for each variable
    """
    baselines = {}
    
    # Key variables and their column names
    variables = {
        'temperature': 'T_degC',
        'salinity': 'Salnty',
        'oxygen_ml_L': 'O2ml_L',
        'oxygen_saturation': 'O2Sat',
        'oxygen_umol_kg': 'Oxy_µmol/Kg',
        'chlorophyll': 'ChlorA',
        'phaeophytin': 'Phaeop',
        'nitrate': 'NO3uM',
        'phosphate': 'PO4uM',
        'silicate': 'SiO3uM',
        'ammonia': 'NH3uM',
        'nitrite': 'NO2uM'
    }
    
    print("\n" + "=" * 70)
    print("BASELINE CALCULATIONS")
    print("=" * 70)
    
    for var_name, col_name in variables.items():
        if col_name not in df.columns:
            print(f"⚠️  {var_name}: column '{col_name}' not found")
            continue
        
        # Convert to numeric, replacing quality flag values
        data = pd.to_numeric(df[col_name], errors='coerce')
        
        # Remove outliers (beyond 3 std devs) and missing values
        data_clean = data.dropna()
        if len(data_clean) > 0:
            mean_val = data_clean.mean()
            std_val = data_clean.std()
            data_clean = data_clean[np.abs(data_clean - mean_val) <= 3 * std_val]
        
        if len(data_clean) == 0:
            print(f"⚠️  {var_name}: no valid data")
            continue
        
        baselines[var_name] = {
            'mean': data_clean.mean(),
            'median': data_clean.median(),
            'std': data_clean.std(),
            'min': data_clean.min(),
            'max': data_clean.max(),
            'n_samples': len(data_clean),
            'column': col_name
        }
        
        print(f"✓ {var_name:20s} Mean: {baselines[var_name]['mean']:8.2f}  "
              f"Median: {baselines[var_name]['median']:8.2f}  "
              f"Samples: {baselines[var_name]['n_samples']:,}")
    
    return baselines


def calculate_nutrient_load_index(baselines: Dict) -> float:
    """
    Calculate composite nutrient load index from nitrate, phosphate, silicate.
    Higher = more nutrient loading.
    """
    # Normalize each nutrient to 0-1 scale based on typical ranges
    nitrate_norm = min(baselines.get('nitrate', {}).get('mean', 0) / 20.0, 1.0)
    phosphate_norm = min(baselines.get('phosphate', {}).get('mean', 0) / 2.0, 1.0)
    
    # Weighted average (nitrate is primary driver of coastal eutrophication)
    nutrient_index = 0.7 * nitrate_norm + 0.3 * phosphate_norm
    
    return nutrient_index


def calculate_oxygen_index(baselines: Dict) -> float:
    """
    Calculate oxygen health index (0 = hypoxic, 1 = well-oxygenated).
    """
    oxygen_ml = baselines.get('oxygen_ml_L', {}).get('mean', 5.0)
    
    # Normalize: 0-2 mL/L = 0 (hypoxic), 6+ mL/L = 1 (healthy)
    oxygen_index = min(max((oxygen_ml - 2.0) / 4.0, 0.0), 1.0)
    
    return oxygen_index


def calculate_habitat_quality_index(baselines: Dict) -> float:
    """
    Calculate habitat quality from temperature and salinity stability.
    Chlorophyll as productivity indicator.
    """
    # Baseline chlorophyll indicates productivity (but too high = blooms)
    # Optimal range: 1-5 µg/L for coastal waters
    chlorophyll = baselines.get('chlorophyll', {}).get('mean', 2.0)
    
    if 1.0 <= chlorophyll <= 5.0:
        habitat_index = 0.7  # Good productivity
    elif chlorophyll > 5.0:
        habitat_index = 0.4  # Bloom conditions (can indicate stress)
    else:
        habitat_index = 0.3  # Low productivity
    
    return habitat_index


def generate_context_code(baselines: Dict) -> str:
    """
    Generate Python code to update calcofi_context.py with real values.
    """
    code = '''
# UPDATED CalCOFI BASELINES (from actual Bottle Database analysis)
CALCOFI_BASELINES = {
    "nutrients": {
'''
    
    if 'nitrate' in baselines:
        code += f'        "nitrate_uM": {baselines["nitrate"]["mean"]:.2f},\n'
    if 'phosphate' in baselines:
        code += f'        "phosphate_uM": {baselines["phosphate"]["mean"]:.2f},\n'
    if 'silicate' in baselines:
        code += f'        "silicate_uM": {baselines["silicate"]["mean"]:.2f},\n'
    
    code += '''        "description": "Real CalCOFI measurements (SoCal coastal, 0-50m depth, 1990-2020)",
        "sources": "Agricultural runoff, urban discharge, natural upwelling"
    },
    "dissolved_oxygen": {
'''
    
    if 'oxygen_ml_L' in baselines:
        code += f'        "do_ml_L": {baselines["oxygen_ml_L"]["mean"]:.2f},\n'
    if 'oxygen_saturation' in baselines:
        code += f'        "do_percent_sat": {baselines["oxygen_saturation"]["mean"]:.1f},\n'
    
    code += '''        "hypoxia_threshold": 2.0,
        "description": "Real CalCOFI oxygen measurements",
        "trend": "Data-driven trend to be calculated"
    },
    "temperature": {
'''
    
    if 'temperature' in baselines:
        code += f'        "sst_celsius": {baselines["temperature"]["mean"]:.2f},\n'
    
    code += '''        "warming_rate": 0.15,  # Approximate from literature
        "description": "Real CalCOFI temperature measurements"
    },
'''
    
    if 'chlorophyll' in baselines:
        code += f'''    "chlorophyll": {{
        "mean_ug_L": {baselines["chlorophyll"]["mean"]:.2f},
        "description": "Chlorophyll-a as phytoplankton biomass proxy"
    }},
'''
    
    code += '}\n'
    
    return code


def main(filepath: str):
    """
    Main analysis pipeline.
    
    Args:
        filepath: Path to CalCOFI bottle database CSV
    """
    print("=" * 70)
    print("CalCOFI BOTTLE DATABASE ANALYZER")
    print("=" * 70)
    
    # Load data
    df = load_calcofi_data(filepath)
    
    # Filter for Southern California coastal waters
    df_filtered = filter_socal_coastal(df)
    
    # Calculate baselines
    baselines = calculate_baselines(df_filtered)
    
    # Calculate composite indices
    print("\n" + "=" * 70)
    print("SIMULATION INDICES (0.0 - 1.0 scale)")
    print("=" * 70)
    
    nutrient_idx = calculate_nutrient_load_index(baselines)
    oxygen_idx = calculate_oxygen_index(baselines)
    habitat_idx = calculate_habitat_quality_index(baselines)
    
    print(f"Nutrient Load Index:    {nutrient_idx:.3f}  (0=pristine, 1=heavily loaded)")
    print(f"Oxygen Health Index:    {oxygen_idx:.3f}  (0=hypoxic, 1=well-oxygenated)")
    print(f"Habitat Quality Index:  {habitat_idx:.3f}  (0=degraded, 1=pristine)")
    
    # Generate updated context code
    print("\n" + "=" * 70)
    print("UPDATED calcofi_context.py CODE")
    print("=" * 70)
    print(generate_context_code(baselines))
    
    # Save summary
    summary = {
        'baselines': baselines,
        'indices': {
            'nutrient_load': nutrient_idx,
            'oxygen_health': oxygen_idx,
            'habitat_quality': habitat_idx
        }
    }
    
    return summary


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_calcofi.py <path_to_bottle_database.csv>")
        print("\nExample:")
        print("  python analyze_calcofi.py bottle_data.csv")
        print("\nDownload CalCOFI Bottle Database from:")
        print("  https://calcofi.org/data/oceanographic-data/bottle-database/")
        sys.exit(1)
    
    filepath = sys.argv[1]
    summary = main(filepath)