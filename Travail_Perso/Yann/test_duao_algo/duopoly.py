from typing import Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import uuid
import pickle
from functools import lru_cache

# ===============================
# Load target sales curve
# ===============================
try:
    with open('/root/partcipant_folder/target_sales_curve_quentin.pkl', 'rb') as f:
        courbe = pickle.load(f)
except:
    with open('target_sales_curve_quentin.pkl', 'rb') as f:
        courbe = pickle.load(f)

MAX_CAPACITY = 80
MAX_DAY = 100

# ===============================
# Feedback storage
# ===============================
def _initialize_data_feedback():
    try:
        with open('duopoly_feedback.data', 'rb') as handle:
            return pickle.load(handle)
    except:
        return {
            'history': pd.DataFrame(),
            'current_simulation': '',
            'cumulative_revenue_current_selling_season': 0.0,
        }

# ===============================
# DP CONFIG
# ===============================
PRICE_GRID = np.arange(20, 101, 5)  # actions discrètes
DISCOUNT = 0.985                    # discount (proche de 1 => pense au futur)
DP_HORIZON = 7                      # horizon court mais non-myopic

# Incertitude demande (simple, mais au moins explicite + probabilisée)
SCENARIOS = {"min": 0.85, "med": 1.0, "max": 1.15}
SCENARIO_PROBS = {"min": 0.25, "med": 0.50, "max": 0.25}

# Trust region
TRUST_DELTA = 10  # € max de variation autour du dernier prix (en absolu)

# ===============================
# Discrétisation / état
# ===============================
def _disc_price(p: float) -> int:
    return int(np.round(p / 5.0) * 5)

def _disc_cap(c: float) -> int:
    # on discrétise la capacité pour limiter le nombre d'états
    return int(np.round(c))

def make_state(day: int, remaining_capacity: float, last_price: float, competitor_price: float) -> Tuple[int, int, int, int]:
    return (
        int(day),
        _disc_cap(remaining_capacity),
        _disc_price(last_price),
        _disc_price(competitor_price),
    )

# ===============================
# Competitor reaction model (simple, stable)
# ===============================
def competitor_reaction(price_ours: float, comp_price: float, comp_has_capacity: bool) -> float:
    """
    Réaction concurrente plausible: inertie + légère attraction vers notre prix.
    Si le concurrent n'a plus de capacité, on fige son prix (ou on le rend non pertinent).
    """
    if not comp_has_capacity:
        return comp_price

    # Inertie + réponse partielle
    next_p = 0.75 * comp_price + 0.25 * price_ours

    # Petites bornes + arrondi cohérent avec la grille
    next_p = float(np.clip(next_p, 20, 120))
    return next_p

# ===============================
# Demand model (cohérent + borné capacité)
# ===============================
def demand_model(
    price_ours: float,
    price_comp: float,
    remaining_capacity: float,
    day: int,
    scenario_mult: float,
    comp_has_capacity: bool,
) -> float:
    """
    Demande stylisée:
    - base issue de courbe[day]['cap_util'] (driver exogène)
    - effet prix relatif via exp(-gap/scale)
    - si le concurrent n'a plus de capacité => avantage compétitif (gap amplifié)
    """
    key = day if day in courbe else max(courbe.keys())
    base = float(courbe[key].get("cap_util", 1.0))

    gap = (price_ours - price_comp)

    # si concurrent out-of-capacity, tu peux capter plus de demande => on "réduit" le gap effectif
    if not comp_has_capacity:
        gap = 0.6 * gap  # avantage (moins pénalisé si tu es plus cher)

    scale = 18.0
    price_effect = np.exp(-gap / scale)

    d = base * price_effect * scenario_mult

    # bornes réalistes
    d = max(0.0, min(float(d), float(remaining_capacity)))
    return d

# ===============================
# Trust region helper
# ===============================
def trust_region_candidates(last_price: float) -> np.ndarray:
    lo = last_price - TRUST_DELTA
    hi = last_price + TRUST_DELTA
    cand = PRICE_GRID[(PRICE_GRID >= lo) & (PRICE_GRID <= hi)]
    # sécurité si trop étroit
    if cand.size == 0:
        cand = PRICE_GRID
    return cand

# ===============================
# DP (Bellman) - cache par appel de p via "solver factory"
# ===============================
def build_dp_solver(comp_has_capacity: bool):
    """
    On construit un solver DP avec cache lru_cache "local",
    pour éviter l'explosion de mémoire sur toute la saison.
    """

    @lru_cache(maxsize=200_000)
    def V(state: Tuple[int, int, int, int], anchor_day: int) -> float:
        day, cap, last_p, comp_p = state

        if day > MAX_DAY or cap <= 0:
            return 0.0
        if day >= anchor_day + DP_HORIZON:
            return 0.0

        best = -1e18
        last_price = float(last_p)
        comp_price = float(comp_p)

        actions = trust_region_candidates(last_price)

        for price in actions:
            exp_val = 0.0
            for sname, mult in SCENARIOS.items():
                d = demand_model(
                    price_ours=float(price),
                    price_comp=comp_price,
                    remaining_capacity=float(cap),
                    day=int(day),
                    scenario_mult=float(mult),
                    comp_has_capacity=comp_has_capacity,
                )
                reward = float(price) * d

                next_comp = competitor_reaction(float(price), comp_price, comp_has_capacity)
                next_state = make_state(day + 1, cap - d, float(price), next_comp)

                exp_val += SCENARIO_PROBS[sname] * (reward + DISCOUNT * V(next_state, anchor_day))

            if exp_val > best:
                best = exp_val

        return float(best)

    def optimal_price(day: int, remaining_capacity: float, last_price: float, competitor_price: float) -> float:
        anchor_day = day
        state = make_state(day, remaining_capacity, last_price, competitor_price)

        best_price = float(np.clip(last_price, 20, 100))
        best_val = -1e18

        actions = trust_region_candidates(last_price)

        for price in actions:
            val = 0.0
            for sname, mult in SCENARIOS.items():
                d = demand_model(
                    price_ours=float(price),
                    price_comp=float(competitor_price),
                    remaining_capacity=float(remaining_capacity),
                    day=int(day),
                    scenario_mult=float(mult),
                    comp_has_capacity=comp_has_capacity,
                )
                reward = float(price) * d

                next_comp = competitor_reaction(float(price), float(competitor_price), comp_has_capacity)
                next_state = make_state(day + 1, remaining_capacity - d, float(price), next_comp)

                val += SCENARIO_PROBS[sname] * (reward + DISCOUNT * V(next_state, anchor_day))

            if val > best_val:
                best_val = val
                best_price = float(price)

        return float(np.clip(best_price, 20, 100))

    return optimal_price

# ===============================
# Main pricing function (challenge API)
# ===============================
def p(
    current_selling_season: int,
    selling_period_in_current_season: int,
    prices_historical_in_current_season: Union[np.ndarray, None],
    demand_historical_in_current_season: Union[np.ndarray, None],
    competitor_has_capacity_current_period_in_current_season: bool,
    information_dump: Optional[Any] = None,
) -> Tuple[float, Any]:

    day = selling_period_in_current_season
    season = current_selling_season

    # --- Init ---
    if day == 1:
        information_dump = _initialize_data_feedback()
        if season == 1:
            information_dump['current_simulation'] = str(uuid.uuid4())
        information_dump['cumulative_revenue_current_selling_season'] = 0.0

        # init stable (pas besoin de random)
        price_init = 40.0
        return price_init, information_dump

    prices = prices_historical_in_current_season
    demand = demand_historical_in_current_season

    # Sécurités
    if prices is None or demand is None or prices.shape[1] < 1 or len(demand) < 1:
        # fallback safe
        return 40.0, information_dump if information_dump is not None else _initialize_data_feedback()

    remaining_capacity = float(MAX_CAPACITY - np.sum(demand))
    remaining_capacity = max(0.0, remaining_capacity)

    competitor_price = float(prices[1, -1])
    last_own_price = float(prices[0, -1])

    # DP solver (avec réaction concurrente + trust region)
    optimal_price = build_dp_solver(competitor_has_capacity_current_period_in_current_season)
    price_today = optimal_price(
        day=day,
        remaining_capacity=remaining_capacity,
        last_price=last_own_price,
        competitor_price=competitor_price
    )

    # --- Logging ---
    if information_dump is None:
        information_dump = _initialize_data_feedback()
        if information_dump.get('current_simulation', '') == '':
            information_dump['current_simulation'] = str(uuid.uuid4())

    revenue_yesterday = float(last_own_price) * float(demand[-1])
    cum_rev = float(information_dump.get('cumulative_revenue_current_selling_season', 0.0)) + revenue_yesterday

    row = {
        'simulation': information_dump.get('current_simulation', ''),
        'season': season,
        'day': day - 1,
        'own_price': last_own_price,
        'competitor_price': competitor_price,
        'demand': float(demand[-1]),
        'remaining_capacity': remaining_capacity,
        'revenue': revenue_yesterday,
        'cumulative_revenue': cum_rev,
        'price_today': price_today
    }

    information_dump['history'] = pd.concat(
        [information_dump.get('history', pd.DataFrame()), pd.DataFrame([row])],
        ignore_index=True
    )
    information_dump['cumulative_revenue_current_selling_season'] = cum_rev

    # Save end of season
    if day >= MAX_DAY:
        with open('duopoly_feedback.data', 'wb') as handle:
            pickle.dump(information_dump, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return price_today, information_dump
