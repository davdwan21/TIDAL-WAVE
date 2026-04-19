import asyncio
from dotenv import load_dotenv
load_dotenv()

from agents.parser import parse_policy
from agents.literature import literature_agent
from agents.historical import historical_agent
from agents.dataset import dataset_agent

async def main():
    policy = "Ban commercial trawling within 50 miles of the California coast"
    print(f"Testing: {policy}\n")
    
    parsed = parse_policy(policy)
    print(f"✅ Parsed: {parsed}\n")
    
    import time
    
    print("--- Literature Agent ---")
    start = time.time()
    try:
        result = await literature_agent(parsed)
        print(f"✅ {time.time()-start:.1f}s — {result}\n")
    except Exception as e:
        print(f"❌ {time.time()-start:.1f}s — {type(e).__name__}: {e}\n")
    
    print("--- Historical Agent ---")
    start = time.time()
    try:
        result = await historical_agent(parsed)
        print(f"✅ {time.time()-start:.1f}s — {result}\n")
    except Exception as e:
        print(f"❌ {time.time()-start:.1f}s — {type(e).__name__}: {e}\n")
    
    print("--- Dataset Agent ---")
    start = time.time()
    try:
        result = await dataset_agent(parsed)
        print(f"✅ {time.time()-start:.1f}s — {result}\n")
        n_calcofi = len(result.get("baseline_values") or [])
        n_inat = len(result.get("trend_indicators") or [])
        print(f"Dataset match counts: CalCOFI={n_calcofi}, iNaturalist={n_inat}")
        assert 2 <= n_calcofi <= 4, f"Expected 2-4 CalCOFI matches, got {n_calcofi}"
        assert 1 <= n_inat <= 2, f"Expected 1-2 iNaturalist matches, got {n_inat}"
    except Exception as e:
        print(f"❌ {time.time()-start:.1f}s — {type(e).__name__}: {e}\n")

asyncio.run(main())