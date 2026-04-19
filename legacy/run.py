from simulation_engine import run_simulation
from visualization import generate_visualization

# Levers from Task 1
levers = {
    "nutrient_load": -0.3,    # 30% reduction
    "habitat_quality": 0.1     # 10% improvement
}

result = run_simulation(levers, years=10)

# Access results
generate_visualization(
    result, 
    policy_description="30% Runoff Reduction",
    output_path="my_policy_result.png"
)