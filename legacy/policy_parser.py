"""
Task 1: Policy Parser (Ollama version) - Refactored
Converts natural language environmental policies into absolute environmental parameters.
Uses local Ollama models - no API keys required!
"""

import json
import requests
from typing import Dict, List, TypedDict


class Action(TypedDict):
    label: str
    type: str  # "reduction" or "improvement"


class PolicyParseResult(TypedDict):
    environment: Dict[str, float]
    confidence: float
    actions: List[Action]
    summary: str


# Baseline environment (pre-policy conditions)
BASELINE_ENVIRONMENT = {
    "temperature": 16.2,      # celsius, optimal 12-16
    "nutrients": 0.6,         # 0-1 normalized
    "pH": 8.05,               # optimal 8.1-8.3
    "salinity": 33.4,         # PSU, optimal 32-34
    "fishing_pressure": 0.2,  # 0-1 normalized
    "pollution_index": 0.3    # 0-1 normalized
}


def parse_policy(
    policy_text: str,
    baseline: Dict[str, float] = None,
    model: str = "llama3.1",
    ollama_url: str = "http://localhost:11434"
) -> PolicyParseResult:
    """
    Converts natural language policy into absolute environmental parameters using Ollama.
    
    Args:
        policy_text: Natural language description of an environmental policy
        baseline: Baseline environment dict (uses BASELINE_ENVIRONMENT if None)
        model: Ollama model to use (default: "llama3.1")
        ollama_url: Ollama server URL (default: "http://localhost:11434")
    
    Returns:
        Dictionary containing:
        - environment: Dict with absolute parameter values after policy application
        - confidence: Float 0.0-1.0 indicating parse confidence
        - actions: List of identified policy actions
        - summary: One sentence mechanism explanation
    
    Example:
        >>> result = parse_policy("Reduce agricultural runoff by 30%")
        >>> print(result["environment"]["nutrients"])
        0.42  # 30% reduction from baseline 0.6
    """
    
    if baseline is None:
        baseline = BASELINE_ENVIRONMENT.copy()
    
    system_prompt = f"""You are the policy interpretation engine for EcoPolicy Simulator, a Southern California coastal ecosystem simulation tool based on CalCOFI oceanographic data (1990-2020, Lines 80-93, 0-50m depth).

Your job: convert natural language environmental policies into absolute environmental parameter values.

CURRENT BASELINE ENVIRONMENT (CalCOFI measured values):
{json.dumps(baseline, indent=2)}

PARAMETER DEFINITIONS:
- temperature: Sea surface temperature in °C (CalCOFI baseline: 15.13°C, optimal range: 12-18°C)
- nutrients: Composite nutrient index 0-1 (CalCOFI baseline: 0.120)
  * Derived from: nitrate (1.55 μM), phosphate (0.44 μM), silicate (3.42 μM)
  * 0 = oligotrophic (pristine), 1 = hypereutrophic (polluted)
- pH: Ocean acidity (typical range: 7.9-8.3, optimal: 8.1-8.2)
- salinity: Salinity in PSU (CalCOFI baseline: 33.42, optimal: 32-34)
- fishing_pressure: Commercial/recreational fishing intensity 0-1 (0 = marine reserve, 1 = maximum sustainable yield)
- pollution_index: General pollution/habitat degradation 0-1 (inverse of habitat quality, CalCOFI baseline: 0.700)

POLICY IMPACT MAPPING (based on Southern California coastal dynamics):
- Nutrient runoff reduction → decrease "nutrients" (affects nitrate/phosphate/silicate), decrease "pollution_index", may increase "chlorophyll" initially
- Wastewater treatment upgrades → decrease "nutrients" and "pollution_index", increase "dissolved_oxygen"
- Agricultural BMPs → decrease "nutrients" (especially nitrate), decrease "pollution_index"
- Fishing restrictions/MPAs → decrease "fishing_pressure", gradual increase in "habitat_quality_index"
- Kelp forest restoration → decrease "pollution_index", stabilize "temperature" (via canopy shading), increase local "dissolved_oxygen"
- Stormwater management → decrease "pollution_index" and "nutrients"
- Ocean acidification mitigation → increase "pH" (very slow response)
- Climate warming → increase "temperature", decrease "dissolved_oxygen" (warm water holds less O₂)

SCIENTIFIC CONSTRAINTS:
- Temperature: 10-22°C (based on CalCOFI SoCal range)
- Nutrients (composite): 0-1 (CalCOFI pristine=0.05, current=0.120, degraded=0.4+)
- pH: 7.8-8.4 (current ocean range)
- Salinity: 30-36 PSU (SoCal coastal range)
- Dissolved oxygen: 0-10 ml/L (hypoxia <2.0, CalCOFI baseline 5.65)
- Oxygen health index: 0-1 (CalCOFI baseline 0.912)
- Fishing pressure: 0-1 (0=fully protected, 0.3=moderate, 1=overfished)
- Pollution index: 0-1 (CalCOFI baseline 0.700, pristine <0.2)
- Chlorophyll: 0-20 μg/L (CalCOFI baseline 0.61, bloom >5)

CALIBRATION RULES:
1. Start from CalCOFI baseline values (shown above)
2. Apply policy changes incrementally - ecosystem response is gradual
3. Maintain parameter coupling:
   - Higher nutrients → higher chlorophyll → lower dissolved oxygen (eutrophication)
   - Higher temperature → lower dissolved oxygen (solubility effect)
   - Lower pollution → higher oxygen health index
4. Only modify parameters directly affected by the policy
5. Conservative estimates: real ecosystem response takes years/decades
6. Confidence reflects policy specificity and scientific certainty (0.0-1.0)

OUTPUT FORMAT:
Return ONLY valid JSON, no markdown, no explanation:
{{
  "environment": {{
    "temperature": 15.13,
    "nutrients": 0.120,
    "pH": 8.10,
    "salinity": 33.42,
    "fishing_pressure": 0.30,
    "pollution_index": 0.700,
  }},
  "confidence": 0.85,
  "actions": [
    {{"label": "30% agricultural runoff reduction", "type": "reduction"}},
    {{"label": "coastal water quality monitoring", "type": "monitoring"}}
  ],
  "summary": "One-sentence mechanistic explanation of how the policy affects the ecosystem (e.g., 'Reduced nutrient loading decreases eutrophication risk, improving oxygen levels and habitat quality over 3-5 years')."
}}"""
    
    # Construct the full prompt
    full_prompt = f"{system_prompt}\n\nPolicy to parse:\n{policy_text}\n\nJSON output:"
    
    # Call Ollama API
    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "format": "json"  # Force JSON output
            },
            timeout=60  # 60 second timeout for generation
        )
        response.raise_for_status()
        
        # Extract the generated text
        result_data = response.json()
        raw_response = result_data.get("response", "")
        
        # Strip markdown code fences if present
        clean = raw_response.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON
        result = json.loads(clean)
        
        # Validate structure
        if "environment" not in result or "confidence" not in result:
            raise ValueError("Invalid response structure from model")
        
        # Ensure all baseline parameters are present (fill in missing ones)
        for param, value in baseline.items():
            if param not in result["environment"]:
                result["environment"][param] = value
        
        return result
        
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Could not connect to Ollama. Make sure it's running:\n"
            "  1. Run 'ollama serve' in a terminal\n"
            "  2. Pull a model: 'ollama pull llama3.1'"
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON. Raw response: {raw_response[:200]}")


def apply_policy_manually(
    policy_text: str,
    baseline: Dict[str, float] = None
) -> Dict[str, float]:
    """
    Fallback function: applies simple policy rules without LLM.
    Useful for testing or when Ollama is unavailable.
    
    Args:
        policy_text: Policy description
        baseline: Baseline environment
    
    Returns:
        Modified environment dict
    """
    if baseline is None:
        baseline = BASELINE_ENVIRONMENT.copy()
    
    env = baseline.copy()
    text_lower = policy_text.lower()
    
    # Simple keyword-based rules
    if "runoff" in text_lower or "nutrient" in text_lower:
        # Extract percentage if present
        import re
        match = re.search(r'(\d+)%', policy_text)
        if match:
            reduction = float(match.group(1)) / 100
            env["nutrients"] *= (1 - reduction)
            env["pollution_index"] *= (1 - reduction)
    
    if "fishing" in text_lower or "no-fishing" in text_lower:
        if "zone" in text_lower or "ban" in text_lower:
            env["fishing_pressure"] *= 0.5  # 50% reduction
    
    if "water quality" in text_lower or "monitoring" in text_lower:
        env["pollution_index"] *= 0.9  # 10% improvement
    
    return env


# Example usage and testing
if __name__ == "__main__":
    # Test cases from the spec
    test_policies = [
        "Reduce agricultural runoff by 30%",
        "Reduce agricultural runoff by 30% and improve coastal water quality monitoring",
        "Implement coastal no-fishing zones and improve water quality monitoring",
        "Mandate 50% reduction in fertilizer use near coastal watersheds by 2028"
    ]
    
    print("=" * 60)
    print("TASK 1: POLICY PARSER TEST (Ollama - Refactored)")
    print("=" * 60)
    print("\nBaseline Environment:")
    for param, value in BASELINE_ENVIRONMENT.items():
        print(f"  {param}: {value}")
    print("\nMake sure Ollama is running: ollama serve")
    print("Using model: llama3.1")
    print("=" * 60)
    
    for i, policy in enumerate(test_policies, 1):
        print(f"\n[Test {i}] Policy: {policy}")
        print("-" * 60)
        
        try:
            result = parse_policy(policy)
            
            print(f"Confidence: {result['confidence']:.0%}")
            print(f"\nEnvironment Parameters (post-policy):")
            for param, value in result['environment'].items():
                baseline_val = BASELINE_ENVIRONMENT[param]
                change = value - baseline_val
                if abs(change) > 0.001:
                    print(f"  {param}: {value:.3f} (Δ {change:+.3f})")
                else:
                    print(f"  {param}: {value:.3f}")
            
            print(f"\nActions identified:")
            for action in result['actions']:
                print(f"  - [{action['type']}] {action['label']}")
            
            print(f"\nSummary: {result['summary']}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            print("\nFalling back to rule-based parser:")
            env = apply_policy_manually(policy)
            print("Environment Parameters (rule-based):")
            for param, value in env.items():
                baseline_val = BASELINE_ENVIRONMENT[param]
                change = value - baseline_val
                if abs(change) > 0.001:
                    print(f"  {param}: {value:.3f} (Δ {change:+.3f})")
                else:
                    print(f"  {param}: {value:.3f}")
        
        print("=" * 60)