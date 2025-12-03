# Optizoignons AI Coding Agent Instructions

## Project Overview

**Optizoignons** is a dynamic pricing competition simulator where agents develop strategies to optimize revenue in a duopoly market. The core challenge is implementing a pricing algorithm that competes against competitors in seasonal market scenarios (100 days per season, multiple seasons).

### Key Concepts
- **Capacity**: Each competitor has 80 units per season to sell
- **Booking Curve**: Target cumulative sales profile that agents should follow (reference curve provided)
- **Demand Model**: Linear relationship `d(p₁, p₂, t) = β₀ₜ + β₁ₜ·p₁ + β₂ₜ·p₂` where:
  - `p₁` = our price, `p₂` = competitor price
  - Parameters vary by time `t` within constraints: `|βₖₜ - βₖₜ₋₁| ≤ δ`
  - Demand values must be rounded to integers
- **Time Dimension**: Demand changes slightly day-to-day; models must adapt parameters accordingly

## Architecture & Core Components

### 1. **Algorithm Entry Point: `duopoly.py`**
Every pricing strategy is a Python module with one function:

```python
def p(
    current_selling_season: int,
    selling_period_in_current_season: int,
    prices_historical_in_current_season: Union[np.ndarray, None],
    demand_historical_in_current_season: Union[np.ndarray, None],
    competitor_has_capacity_current_period_in_current_season: bool,
    information_dump: Optional[Any] = None,
) -> Tuple[float, Any]:
    # Returns: (price, updated_information_dump)
```

**Critical Details**:
- Called once per day for 100 days per season across multiple seasons
- `prices_historical` = 2D array: `[our_prices, competitor_prices]`
- `demand_historical` = 1D array of actual demands (integers, 0-80 per day)
- `information_dump` = persistent state dictionary persisted via pickle (survives across calls)
- Day 1 of new season: no historical data available; use this to reset season tracking

### 2. **Demand Estimation Patterns**
Two main estimation strategies exist:

**Approach A: Linear Regression with Time-Varying Parameters** (`Travail_Perso/Quentin/DPC/duopoly.py`)
- Estimates coefficients per season from available data
- Uses 3 demand models, selecting best based on historical performance
- Optimizes prices to hit target daily demand calculated from booking curve

**Approach B: Booking Curve Following** (`Duopoly/Quentin/duopoly.py`)
- Directly compares actual cumulative sales vs. target from `mean_curve.csv`
- Adjusts price by factor: `new_price = last_price × (1 + α·delta)`
- Applies dampening constraints: price changes capped ±20% per day, min 5, max 100

### 3. **Historical Data Management**
- Store persistent state in `information_dump` dict (persisted to `duopoly_feedback.data` pickle file)
- Track cumulative revenue, season-level performance metrics, and season-to-model mappings
- Clean up pickle file at start of new simulations via `run_a_simu.py`

### 4. **Workflow: Running Simulations**
Use `Travail_Perso/Yann/tools/run_a_simu.py`:

```python
from tools.run_a_simu import run_a_simu
info_dump, factors, df_select = run_a_simu(
    csv_of_competition='duopoly_competition_details.csv',
    s=1,
    max_t=20,  # Days to simulate
    is_courbe='target_sales_curve.pkl'
)
```

This loads competition data (actual competitor prices/demand), calls your `duopoly.p()` for each day, and returns full performance details.

## Conventions & Patterns

### **File Organization**
- **`Duopoly/{person}/duopoly.py`**: Latest finalized strategy per team member
- **`Travail_Perso/{person}/`**: Personal sandbox; testing ground for new ideas
- **`Travail_Commun/Optimization/`**: Shared notebooks for regression analysis (OLS, LAD)
- **`Resultats/{date}/{person}/`**: CSV results from competition runs, organized by date
- **`Travail_Perso/{person}/data/`**: Merged/processed datasets from multiple simulations

### **Data Files**
- **`mean_curve.csv`**: Reference booking curve (100-element array of cumulative sales proportions, 0.004 → 0.9813)
- **`modeles_moyens_elasticite.csv`**: Pre-computed demand model coefficients with elasticity data
- **`duopoly_competition_details.csv`**: Competition replay data (columns: competitor price, demand, capacity flag, selling_period)
- **`target_sales_curve.pkl`**: Pickled dict mapping day→capacity utilization target

### **Key Variables & Naming**
- `alpha` (sensitivity factor): Controls price adjustment magnitude (0.2-0.3 typical)
- `delta`: Relative deviation from target (actual−target)/target
- `factor`: Internal state variable tracking model performance
- Parameters `coeff_price`, `coeff_competitor`, `intercept`: Linear demand model coefficients

## Debugging & Analysis

### **Common Issues**
1. **Day 1 calls have no history**: Always guard with `if selling_period_in_current_season <= 1`
2. **Integer demand vs. float estimation**: Apply `round()` or `max(0, int(...))` to demand estimates
3. **Season transitions**: Day 100 data available but new season starts fresh; track accordingly
4. **Pickle persistence**: Delete `duopoly_feedback.data` between simulation runs to start fresh

### **Regression Analysis Notebooks** (Shared)
- **`02-ols-regression.ipynb`**: Ordinary Least Squares for demand coefficients
- **`02-lad-regression.ipynb`**: Least Absolute Deviations (more robust to outliers)
- **`03-production-planning-advanced.ipynb`**: Pyomo optimization for multi-period planning

### **Validation Workflow**
1. Extract competition results to CSV: columns expected are `selling_period`, `demand`, `price_competitor`, `competitor_has_capacity`
2. Calculate cumulative revenue: `sum(demand[t] × our_price[t])` for each season
3. Compare booking curve adherence: expected vs. actual cumulative sales

## Technologies & Tools

- **NumPy/Pandas**: Data handling and demand calculations
- **Pyomo**: Optimization framework for constrained price-setting (emerging pattern; see notebooks)
- **Pickle**: Serializing `information_dump` state across function calls
- **Matplotlib**: Plotting demand curves, capacity utilization, revenue over time

## Next Steps / Evolution

Based on README notes:
- **Scenario learning**: Build separate models for low/medium/high demand scenarios; detect scenario early
- **Competitor price prediction**: Use first 20 days of data to estimate competitor's demand model
- **Time-varying parameters**: Implement smoothness constraints on coefficients across days (Pyomo-based)
- **Elasticity calculation**: Measure price sensitivity dynamically from competition data

---

**Key Files to Study**: `Travail_Perso/Yann/duopoly_strat/12.03/duopoly.py`, `Travail_Perso/Quentin/DPC/duopoly.py`, `Travail_Perso/Yann/tools/run_a_simu.py`
