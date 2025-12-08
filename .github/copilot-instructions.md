# Optizoignons Codebase Instructions

## Project Overview
**Optizoignons** is a research/learning project on dynamic pricing optimization in a competitive duopoly setting. The goal is to develop pricing algorithms that maximize revenue while competing with an opponent in an online marketplace with 80-unit inventory and 100-day selling seasons.

### Core Problem
- Participants dynamically set prices to compete for demand
- Demand is a function of both own price and competitor's price: `d(p₁, p₂) = β₀ + β₁p₁ + β₂p₂`
- Key constraints: limited inventory, continuous seasons, observable competitor behavior
- Optimization approaches: reference curve following, linear/LAD regression, strategic pricing models

### Repository Structure
```
Duopoly/          → Final working pricing algorithms per person (Louiza, Quentin, Yann)
Travail_Commun/   → Shared course materials and optimization techniques (OLS, LAD regression, production planning)
Travail_Perso/    → Individual sandboxes organized by person with iterations and experiments
Resultats/        → Competition results organized by date/person: {Date}/{Name}/duopoly_competition_details.csv
```

## Critical Data Structures

### Competition Output CSV Schema
`duopoly_competition_details.csv` contains:
- `selling_season`, `selling_period` - 100 days per season (~30 seasons tested)
- `price` (our algorithm), `price_competitor` - continuous values [5-100]
- `demand` - integer units sold (highly responsive to relative pricing)
- `competitor_has_capacity` - boolean for strategic inventory exhaustion analysis
- `calculation_duration` - algorithm runtime (unused in analysis)

### Reference Curve (Booking Curve)
Embedded in code as `reference` array: normalized daily cumulative sales [0.004, 0.0071...0.9813]. Represents "ideal" sell-through curve to guide revenue optimization. Example usage in `Duopoly/Louiza/duopoly.py` lines 32-33.

## Pricing Algorithm Patterns

### Interface Convention (Required Signature)
All pricing algorithms implement this interface:
```python
def p(
    current_selling_season: int,
    selling_period_in_current_season: int,
    prices_historical_in_current_season: np.ndarray,  # shape (2, period): [[our_prices], [competitor_prices]]
    demand_historical_in_current_season: np.ndarray,  # shape (period,): actual units sold each day
    competitor_has_capacity_current_period_in_current_season: bool,
    information_dump=Optional[Any],
) -> Tuple[float, Any]:  # (new_price, optional_state)
```

### Algorithm Strategies Explored
1. **Random Baseline** (`Duopoly/*/`): Prices [30-90] uniformly random. Establishes demand curve.
2. **Reference Curve Tracking** (`Duopoly/Louiza/duopoly.py`): Compares actual cumulative sales to reference curve, adjusts price with sensitivity factor `α=0.1`. Prices constrained [5-100] and limited to ±20% change per day.
3. **Demand Regression** (`Travail_Perso/Louiza/duopoly.py`): Fits `d ≈ β₀ + β₁p_ours + β₂p_comp` via OLS using last 6+ observations, then optimizes price. Also includes competitor reaction modeling (`p₂ ≈ γ₀ + γ₁p₁`).
4. **State Persistence** (`Travail_Perso/Yann/duopoly.py`): Uses pickle to maintain feedback DataFrame across seasons—stores `(simulation, day, season, demand, own_price, competitor_price, revenue)` for learning.

## Development Workflows

### Analysis Pipeline
1. **Load Results**: `Resultats/fichier_pipeline_analyse.ipynb` loads CSVs from dated folders
2. **Feature Engineering**: Create `revenue = price × demand`, explore correlations
3. **Regression Study**: Use `Travail_Commun/Optimization/{02-ols,02-lad}-regression.ipynb` to understand OLS vs LAD objective functions—key insight from 11.11.25 notes: OLS (sum squared error) vs LAD (sum absolute error/n_obs) yield different sensitivities to outliers
4. **Cross-Date Comparison**: Iterate over `Resultats/{Date}/{Name}/` directories to track algorithm improvements

### Key Dependencies
- **Pyomo**: Non-linear optimization solver for regression (`pyomo.environ as pyo`)
- **NumPy/Pandas**: Numerical computing and CSV handling
- **SciPy**: `minimize_scalar` for demand model fitting (used in advanced versions)
- **Solvers**: ipopt (non-linear), appsi_highs (linear problems)

### Common Issues
- **Demand function float vs. integer**: Real demand is integer, but regression returns floats—account in threshold comparisons
- **Competitor price history indexing**: `prices_historical[0]` = our prices, `prices_historical[1]` = competitor; ensure shape is (2, period)
- **Reference curve boundary**: Array is fixed 100 elements; `last_period_index = period - 1` to avoid off-by-one errors
- **Stock constraint**: Total inventory always 80 units; cumulative demand cannot exceed this

## Collaboration Conventions
- **Code Location**: `Duopoly/{Name}/duopoly.py` for production algorithms
- **Experiments**: Iterate in `Travail_Perso/{Name}/` before promoting
- **Results Sharing**: CSV export to `Resultats/{YYYY.MM.DD}/{Name}/` with consistent naming
- **Notebook Analysis**: Shared learning in `Travail_Commun/Optimization/` (e.g., regression theory, optimization techniques)
- **Documentation**: French comments common in existing code; maintain for team clarity

## Next Priority Areas (From TODO)
1. Demand scenario classification: Detect low/medium/high demand regimes at runtime
2. Dynamic competitor price modeling: Pre-learn on first ~20 days, apply to remaining 80
3. Advanced pricing: Integrate elasticity + sensitivity calculations into optimization
4. Regression validation: Debug why 11.11.25 regressions performed poorly—verify demand model specification
