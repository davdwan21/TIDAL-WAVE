"""
Demo-safe outputs. If live demo fails, set DEMO_MODE=true and type trigger phrases.

Hand-crafted PolicyInterpretation objects for instant hackathon demos (no live Gemini).
"""

from __future__ import annotations

import re
from typing import Optional

from schema import ParameterDelta, PolicyInterpretation, Source


def fuzzy_match(text: str, keys: list[str]) -> Optional[str]:
    """Return the best-matching canned key using simple token overlap (keyword overlap)."""
    t_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    if not t_tokens:
        return None
    best_key: Optional[str] = None
    best_score = 0.0
    for key in keys:
        k_tokens = set(re.findall(r"[a-z0-9]+", key.lower()))
        if not k_tokens:
            continue
        inter = len(t_tokens & k_tokens)
        score = inter / max(len(k_tokens), 1)
        if score > best_score and inter >= 1:
            best_score = score
            best_key = key
    if best_key is None or best_score < 0.25:
        return None
    return best_key


_CANNED_FISHING = PolicyInterpretation(
    plain_english_summary=(
        "A coastwide ban on commercial extraction sharply lowers fleet-wide fishing mortality on "
        "forage fish while reducing seabird and pinniped bycatch risk; near-term biomass of small "
        "pelagics typically rises before predator responses rebalance the food web."
    ),
    parameter_deltas=[
        ParameterDelta(
            target="fishing_fleet.catch_rate",
            operation="multiply",
            value=0.35,
            rationale="Fleet-wide catch rate scales down under a hard commercial fishing prohibition.",
        ),
        ParameterDelta(
            target="fishing_fleet.effort_level",
            operation="multiply",
            value=0.4,
            rationale="Observed effort drops when large nearshore areas close to commercial gears.",
        ),
        ParameterDelta(
            target="anchovy.mortality_rate",
            operation="multiply",
            value=0.75,
            rationale="Lower directed and incidental removals reduce anchovy fishing mortality.",
        ),
        ParameterDelta(
            target="sardine.mortality_rate",
            operation="multiply",
            value=0.78,
            rationale="Sardine fleets co-locate with anchovy; parallel mortality relief is expected.",
        ),
        ParameterDelta(
            target="pelican.mortality_rate",
            operation="multiply",
            value=0.9,
            rationale="Reduced hook-and-line and net encounters lower incidental seabird mortality.",
        ),
    ],
    confidence=0.88,
    sources=[
        Source(
            title="NOAA Fisheries — West Coast Commercial Fishing Overview",
            url="https://www.fisheries.noaa.gov/region/west-coast",
            excerpt="Federal summaries of West Coast commercial fishery management and gear restrictions.",
        ),
        Source(
            title="PFMC — Pacific Fishery Management Council Groundfish Management",
            url="https://www.pcouncil.org/managed_fishery/groundfish/",
            excerpt="Council processes for groundfish and nearshore conservation measures off California.",
        ),
        Source(
            title="CalCOFI Reports — Long-term ichthyoplankton and hydrography",
            url="https://calcofi.org/data/",
            excerpt="Decadal CalCOFI sampling frames small pelagic population variability in the CCS.",
        ),
    ],
    reasoning_trace=[
        "📋 Parser read a commercial fishing ban with nearshore scope and inferred anchovy–sardine–fleet linkages.",
        "📚 Literature agent (canned) cites peer-reviewed fisheries closures recovering forage fish within 3–7 years.",
        "🏛️ Historical agent (canned) notes West Coast rockfish RCAs as analogs with measurable bycatch reductions.",
        "📊 Dataset agent (canned) anchors on CalCOFI larval indices and iNat coastal predator sightings as baselines.",
        "🤔 Skeptic agent (canned) warns recovery speed depends on illegal fishing enforcement and ocean temperature.",
        "✅ Synthesizer (canned) merged 3 NOAA-aligned sources into 5 validated parameter deltas at confidence 0.88.",
    ],
    warnings=["Canned demo response (DEMO_MODE)."],
)

_CANNED_MPA = PolicyInterpretation(
    plain_english_summary=(
        "Designating a no-take marine protected area increases local spawning biomass, shifts fishing "
        "effort to edges, and typically lowers fishing mortality for sedentary reef-associated species while "
        "raising protected_area coverage in the sim."
    ),
    parameter_deltas=[
        ParameterDelta(
            target="protected_area.coverage_percent",
            operation="add",
            value=12.0,
            rationale="MPA establishment expands no-take footprint as percent of modeled domain.",
        ),
        ParameterDelta(
            target="anchovy.reproduction_rate",
            operation="multiply",
            value=1.08,
            rationale="Spillover and reduced juvenile mortality modestly improve anchovy recruitment proxies.",
        ),
        ParameterDelta(
            target="zooplankton.mortality_rate",
            operation="multiply",
            value=0.95,
            rationale="Trophic release can slightly reduce zooplankton losses to small-pelagic predation.",
        ),
        ParameterDelta(
            target="fishing_fleet.effort_level",
            operation="multiply",
            value=0.88,
            rationale="Displaced effort lowers realized effort inside the protected footprint.",
        ),
    ],
    confidence=0.84,
    sources=[
        Source(
            title="NOAA — National Marine Sanctuaries",
            url="https://sanctuaries.noaa.gov/",
            excerpt="Sanctuary and MPA objectives for biodiversity and sustainable use.",
        ),
        Source(
            title="IUCN — Marine Protected Areas Guidelines",
            url="https://www.iucn.org/resources/toolkits-and-publications/marine-protected-areas",
            excerpt="International guidance on no-take zones and ecological monitoring.",
        ),
        Source(
            title="CDFW — Marine Protected Areas of California",
            url="https://wildlife.ca.gov/Conservation/Marine/MPAs",
            excerpt="California network design and performance monitoring context.",
        ),
    ],
    reasoning_trace=[
        "📋 Parser classified an MPA establishment action with explicit percent coverage language.",
        "📚 Literature agent (canned) highlights average biomass gains of 20–50% for targeted taxa inside reserves.",
        "🏛️ Historical agent (canned) compares to Channel Islands MPAs with multi-year monitoring time series.",
        "📊 Dataset agent (canned) ties CalCOFI ichthyoplankton trends to reserve-adjacent reference stations.",
        "🤔 Skeptic agent (canned) flags edge effects and illegal fishing as confounders for fast wins.",
        "✅ Synthesizer (canned) produced 4 deltas emphasizing protected_area.coverage_percent at confidence 0.84.",
    ],
    warnings=["Canned demo response (DEMO_MODE)."],
)

_CANNED_DEREG = PolicyInterpretation(
    plain_english_summary=(
        "Deregulating commercial fishing typically raises fleet catch and effort in the short run, "
        "increasing fishing mortality on forage fish unless strong biomass safeguards exist."
    ),
    parameter_deltas=[
        ParameterDelta(
            target="fishing_fleet.effort_level",
            operation="multiply",
            value=1.25,
            rationale="Weaker rules historically correlate with higher days-at-sea and effort intensity.",
        ),
        ParameterDelta(
            target="fishing_fleet.catch_rate",
            operation="multiply",
            value=1.2,
            rationale="Catch-per-unit-effort ceilings loosen under deregulation scenarios.",
        ),
        ParameterDelta(
            target="anchovy.catch_rate",
            operation="add",
            value=0.12,
            rationale="Anchovy directed catch pressure rises when trip limits expand.",
        ),
        ParameterDelta(
            target="sardine.mortality_rate",
            operation="multiply",
            value=1.1,
            rationale="Sardine mortality increases when quotas are relaxed in warm regimes.",
        ),
    ],
    confidence=0.62,
    sources=[
        Source(
            title="NOAA Economics & Human Dimensions — Fishery Markets",
            url="https://www.fisheries.noaa.gov/topic/economics",
            excerpt="Economic framing of supply responses to regulatory rollbacks.",
        ),
        Source(
            title="FAO — The State of World Fisheries and Aquaculture",
            url="https://www.fao.org/sofi",
            excerpt="Global patterns linking effort expansion to stock stress when controls weaken.",
        ),
        Source(
            title="Sea Around Us — Catch Reconstruction Methods",
            url="https://www.seaaroundus.org/",
            excerpt="Methods literature on unreported catch when enforcement declines.",
        ),
    ],
    reasoning_trace=[
        "📋 Parser read a deregulation / restriction-removal intent aimed at commercial fleets.",
        "📚 Literature agent (canned) cites mixed evidence: short-run landings rise within 1–3 seasons.",
        "🏛️ Historical agent (canned) references 1990s quota relaxations with delayed stock declines.",
        "📊 Dataset agent (canned) contrasts rising fleet indices with flat CalCOFI larval anomalies.",
        "🤔 Skeptic agent (canned) stresses model uncertainty on recruitment steepness under warming.",
        "✅ Synthesizer (canned) emitted 4 deltas with cautious confidence 0.62 given ecological risk.",
    ],
    warnings=["Canned demo response (DEMO_MODE)."],
)

_CANNED_RUNOFF = PolicyInterpretation(
    plain_english_summary=(
        "Reducing agricultural runoff lowers nitrogen and organic loads to estuaries, dampening "
        "eutrophication signals and shifting phytoplankton–zooplankton coupling toward lower bloom frequency."
    ),
    parameter_deltas=[
        ParameterDelta(
            target="coastal_community.runoff_rate",
            operation="multiply",
            value=0.7,
            rationale="Watershed caps translate to lower modeled runoff into coastal cells.",
        ),
        ParameterDelta(
            target="ocean.nutrient_level",
            operation="add",
            value=-0.2,
            rationale="Lower nitrate flux reduces bulk nutrient index in nearshore ocean boxes.",
        ),
        ParameterDelta(
            target="ocean.pollution_index",
            operation="add",
            value=-0.15,
            rationale="Pollution index tracks organic and nutrient loading combined.",
        ),
        ParameterDelta(
            target="phytoplankton.growth_rate",
            operation="multiply",
            value=0.92,
            rationale="Fewer extreme blooms modestly reduce mean phytoplankton growth spikes.",
        ),
        ParameterDelta(
            target="zooplankton.growth_rate",
            operation="multiply",
            value=1.04,
            rationale="Grazers can benefit once bloom toxicity and hypoxia frequency decline.",
        ),
    ],
    confidence=0.79,
    sources=[
        Source(
            title="US EPA — Nutrient Pollution",
            url="https://www.epa.gov/nutrientpollution",
            excerpt="National nutrient management framing for agricultural sources.",
        ),
        Source(
            title="USGS — California Water Science Center",
            url="https://www.usgs.gov/centers/ca-water",
            excerpt="Hydrologic monitoring tying storms to nutrient pulses in coastal watersheds.",
        ),
        Source(
            title="CalCOFI — Nutrient and chlorophyll time series",
            url="https://calcofi.org/",
            excerpt="Regional hydrographic context for bloom-linked plankton variability.",
        ),
    ],
    reasoning_trace=[
        "📋 Parser isolated agricultural runoff reduction with explicit nutrient-cap language.",
        "📚 Literature agent (canned) cites 2–4 year lag before chlorophylla anomalies ease in estuaries.",
        "🏛️ Historical agent (canned) compares to Central Valley export control pilots in wet winters.",
        "📊 Dataset agent (canned) pairs CalCOFI chl-a with iNat coastal community proxy counts.",
        "🤔 Skeptic agent (canned) notes drought years can mask runoff reductions in monitoring data.",
        "✅ Synthesizer (canned) merged 3 sources into 5 nutrient-web deltas at confidence 0.79.",
    ],
    warnings=["Canned demo response (DEMO_MODE)."],
)

_CANNED_SHIPPING = PolicyInterpretation(
    plain_english_summary=(
        "A carbon price on ocean shipping incentivizes slower steaming and fuel switching, lowering "
        "CO2-equivalent emissions intensity and indirectly reducing acidification and local pollution stressors."
    ),
    parameter_deltas=[
        ParameterDelta(
            target="ocean.ph",
            operation="add",
            value=0.02,
            rationale="Lower CO2 intensity slightly relieves surface acidification pressure in the model.",
        ),
        ParameterDelta(
            target="ocean.pollution_index",
            operation="add",
            value=-0.1,
            rationale="Cleaner fuels reduce aerosol and heavy-fuel oil deposition linked to pollution index.",
        ),
        ParameterDelta(
            target="coastal_community.consumption_rate",
            operation="multiply",
            value=0.97,
            rationale="Higher transport costs marginally dampen imported-goods consumption proxy.",
        ),
        ParameterDelta(
            target="ocean.temperature",
            operation="add",
            value=-0.03,
            rationale="Marginal emissions cuts nudge radiative forcing feedbacks downward in stylized runs.",
        ),
    ],
    confidence=0.58,
    sources=[
        Source(
            title="IMO — Initial IMO GHG Strategy",
            url="https://www.imo.org/en/MediaCentre/HotTopics/Pages/Decarbonizing-international-shipping.aspx",
            excerpt="International shipping emissions goals and policy levers.",
        ),
        Source(
            title="IPCC AR6 WGIII — Transport",
            url="https://www.ipcc.ch/report/ar6/wg3/",
            excerpt="Mitigation costs and effectiveness for shipping-sector carbon pricing.",
        ),
        Source(
            title="NOAA PMEL — Ocean Acidification Program",
            url="https://www.pmel.noaa.gov/co2/story/Ocean+Acidification",
            excerpt="Mechanistic link between CO2 emissions and surface ocean pH.",
        ),
    ],
    reasoning_trace=[
        "📋 Parser read a carbon tax on ocean shipping with economy-wide coastal linkages.",
        "📚 Literature agent (canned) cites IMO scenarios where 10–20% emissions cuts emerge within a decade.",
        "🏛️ Historical agent (canned) compares to EU ETS shipping pilot announcements and fleet responses.",
        "📊 Dataset agent (canned) uses coarse pH and temperature baselines as anchors for small deltas.",
        "🤔 Skeptic agent (canned) warns global vs regional attribution is weak for CCS-scale pH shifts.",
        "✅ Synthesizer (canned) produced 4 conservative deltas at confidence 0.58 given long feedback lags.",
    ],
    warnings=["Canned demo response (DEMO_MODE)."],
)

CANNED_KEYS: list[str] = [
    "ban commercial fishing ban commercial trawling",
    "establish marine protected area",
    "deregulate fishing remove fishing restrictions",
    "reduce agricultural runoff",
    "carbon tax ocean shipping",
]

CANNED_BY_KEY: dict[str, PolicyInterpretation] = {
    CANNED_KEYS[0]: _CANNED_FISHING,
    CANNED_KEYS[1]: _CANNED_MPA,
    CANNED_KEYS[2]: _CANNED_DEREG,
    CANNED_KEYS[3]: _CANNED_RUNOFF,
    CANNED_KEYS[4]: _CANNED_SHIPPING,
}


def match_canned_policy(policy_text: str) -> Optional[PolicyInterpretation]:
    """Return a deep-copied canned interpretation if ``policy_text`` fuzzy-matches a trigger."""
    key = fuzzy_match(policy_text, CANNED_KEYS)
    if key is None:
        return None
    hit = CANNED_BY_KEY.get(key)
    if hit is None:
        return None
    return hit.model_copy(deep=True)
