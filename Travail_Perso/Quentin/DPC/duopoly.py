from typing import Any, Optional, Tuple, Union, Dict
import numpy as np
from functools import lru_cache

# =========================
# Fitting utilities
# =========================

def _fit_demand_model(
    prices_historical_in_current_season: Union[np.ndarray, None],
    demand_historical_in_current_season: Union[np.ndarray, None],
    min_points: int = 8,
    ridge: float = 1e-3,  # régularisation légère pour stabilité
) -> Optional[np.ndarray]:
    """
    Fit: d ≈ β0 + β1 p_ours + β2 p_comp  (OLS / Ridge)
    Retourne beta (3,)
    """
    if prices_historical_in_current_season is None or demand_historical_in_current_season is None:
        return None

    prices = np.asarray(prices_historical_in_current_season, dtype=float)
    demand = np.asarray(demand_historical_in_current_season, dtype=float)

    if prices.ndim != 2 or prices.shape[0] < 1:
        return None

    n_obs = min(prices.shape[1], demand.shape[0])
    if n_obs < min_points:
        return None

    p_ours = prices[0, -n_obs:]
    if prices.shape[0] >= 2:
        p_comp = prices[1, -n_obs:]
    else:
        p_comp = np.full_like(p_ours, np.mean(p_ours))

    d = demand[-n_obs:]

    # Matrice design
    X = np.column_stack([np.ones_like(p_ours), p_ours, p_comp])

    # Ridge closed-form: beta = (X^T X + λI)^(-1) X^T y
    try:
        XtX = X.T @ X
        reg = ridge * np.eye(XtX.shape[0])
        beta = np.linalg.solve(XtX + reg, X.T @ d)
    except np.linalg.LinAlgError:
        return None

    if beta.shape != (3,):
        return None

    # Petite sanity check: on veut idéalement β1 < 0 (demande décroît avec notre prix)
    # On n'interdit pas, mais si β1 est positif, le modèle est suspect.
    return beta


def _fit_competitor_reaction(
    prices_historical_in_current_season: Union[np.ndarray, None],
    min_points: int = 8,
    ridge: float = 1e-3,
) -> Optional[np.ndarray]:
    """
    Fit: p_comp ≈ γ0 + γ1 p_ours  (OLS / Ridge)
    Retourne gamma (2,)
    """
    if prices_historical_in_current_season is None:
        return None

    prices = np.asarray(prices_historical_in_current_season, dtype=float)
    if prices.ndim != 2 or prices.shape[0] < 2:
        return None

    p_ours = prices[0]
    p_comp = prices[1]
    n_obs = min(len(p_ours), len(p_comp))
    if n_obs < min_points:
        return None

    p_ours = p_ours[-n_obs:]
    p_comp = p_comp[-n_obs:]

    X = np.column_stack([np.ones_like(p_ours), p_ours])

    try:
        XtX = X.T @ X
        reg = ridge * np.eye(XtX.shape[0])
        gamma = np.linalg.solve(XtX + reg, X.T @ p_comp)
    except np.linalg.LinAlgError:
        return None

    if gamma.shape != (2,):
        return None

    return gamma


# =========================
# Simple reactive fallback
# =========================

def _reactive_price(last_price: float, last_demand: Optional[float]) -> float:
    if last_demand is None:
        return float(np.clip(last_price, 5, 100))

    if last_demand >= 3:
        new_p = last_price * 1.05
    elif last_demand <= 0:
        new_p = last_price * 0.95
    else:
        new_p = last_price

    return float(np.clip(new_p, 5, 100))


# =========================
# Dynamic Programming core
# =========================

def _dp_price(
    beta: np.ndarray,
    gamma: np.ndarray,
    day: int,
    remaining_capacity: float,
    last_price: float,
    last_comp_price: float,
    competitor_has_capacity: bool,
    horizon: int = 6,
    discount: float = 0.985,
    price_floor: float = 5.0,
    price_cap: float = 100.0,
    trust_delta: float = 15.0,
) -> float:
    """
    VRAI DP multi-steps :
    - état = (t, cap, p_last, p_comp_last) discretisé
    - action = prix sur une grille restreinte par trust region
    - transition : comp_price évolue via gamma + inertie
    - reward = p * demand_pred (bornée par cap)
    """

    β0, β1, β2 = float(beta[0]), float(beta[1]), float(beta[2])
    γ0, γ1 = float(gamma[0]), float(gamma[1])

    # Grille de prix (dense mais pas trop) + trust region
    base_grid = np.arange(price_floor, price_cap + 1e-9, 2.0)

    def candidates(p_prev: float) -> np.ndarray:
        lo = max(price_floor, p_prev - trust_delta)
        hi = min(price_cap,  p_prev + trust_delta)
        cand = base_grid[(base_grid >= lo) & (base_grid <= hi)]
        if cand.size == 0:
            return base_grid
        return cand

    def predict_comp_price(p_ours: float, p_comp_prev: float) -> float:
        # best-response linéaire + inertie (stabilise)
        br = γ0 + γ1 * p_ours
        next_p = 0.7 * p_comp_prev + 0.3 * br
        return float(np.clip(next_p, price_floor, price_cap))

    def predict_demand(p_ours: float, p_comp: float, cap: float) -> float:
        # si concurrent out-of-capacity, on boost un peu la demande (effet "marché captif")
        # (c’est une approximation raisonnable sans tricher)
        comp_term = p_comp
        boost = 1.0
        if not competitor_has_capacity:
            boost = 1.10
            comp_term = p_comp + 10.0  # revient à dire "concurrent moins pressant"

        d = (β0 + β1 * p_ours + β2 * comp_term) * boost
        d = max(0.0, float(d))
        return min(d, float(cap))

    # Discrétisation d'état pour limiter le nombre d'états
    def disc_price(p: float) -> int:
        return int(np.round(p / 2.0) * 2)

    def disc_cap(c: float) -> int:
        return int(np.round(c))

    anchor_day = day  # pour stopper le DP à day+horizon

    @lru_cache(maxsize=200_000)
    def V(t: int, cap_i: int, p_last_i: int, p_comp_i: int) -> float:
        # t est le jour absolu (comme day), cap_i capacité restante (discrète)
        if cap_i <= 0:
            return 0.0
        if t >= anchor_day + horizon:
            return 0.0

        cap = float(cap_i)
        p_last = float(p_last_i)
        p_comp = float(p_comp_i)

        best = -1e18
        for p in candidates(p_last):
            d = predict_demand(p, p_comp, cap)
            r = p * d
            p_comp_next = predict_comp_price(p, p_comp)
            cap_next = disc_cap(cap - d)
            val = r + discount * V(t + 1, cap_next, disc_price(p), disc_price(p_comp_next))
            if val > best:
                best = val

        return float(best)

    # Choix argmax au jour courant
    cap0 = disc_cap(remaining_capacity)
    p_last0 = disc_price(last_price)
    p_comp0 = disc_price(last_comp_price)

    best_p = float(np.clip(last_price, price_floor, price_cap))
    best_val = -1e18

    for p in candidates(float(p_last0)):
        d = predict_demand(p, float(p_comp0), float(cap0))
        r = p * d
        p_comp_next = predict_comp_price(p, float(p_comp0))
        cap_next = disc_cap(float(cap0) - d)
        val = r + discount * V(day + 1, cap_next, disc_price(p), disc_price(p_comp_next))
        if val > best_val:
            best_val = val
            best_p = float(p)

    return float(np.clip(best_p, price_floor, price_cap))


# =========================
# Main DPC function
# =========================

def p(
    current_selling_season: int,
    selling_period_in_current_season: int,
    prices_historical_in_current_season: Union[np.ndarray, None],
    demand_historical_in_current_season: Union[np.ndarray, None],
    competitor_has_capacity_current_period_in_current_season: bool,
    information_dump: Optional[Any] = None,
) -> Tuple[float, Any]:

    # init info
    if information_dump is None or not isinstance(information_dump, dict):
        information_dump = {
            "last_price": 60.0,
            "beta": None,
            "gamma": None,
        }

    day = int(selling_period_in_current_season)

    # last observed prices
    last_price = float(information_dump.get("last_price", 60.0))
    last_comp_price = 60.0

    if prices_historical_in_current_season is not None:
        arr = np.asarray(prices_historical_in_current_season, dtype=float)
        if arr.ndim == 2 and arr.shape[1] > 0:
            last_price = float(arr[0, -1])
            if arr.shape[0] >= 2:
                last_comp_price = float(arr[1, -1])

    # last observed demand
    last_demand = None
    if demand_historical_in_current_season is not None:
        dem = np.asarray(demand_historical_in_current_season, dtype=float)
        if dem.size > 0:
            last_demand = float(dem[-1])
        else:
            dem = None
    else:
        dem = None

    # capacity estimate (si ton stock total est 80 dans le challenge)
    remaining_capacity = 80.0
    if demand_historical_in_current_season is not None:
        remaining_capacity = float(max(0.0, 80.0 - np.sum(np.asarray(demand_historical_in_current_season, dtype=float))))

    # fit models
    beta = _fit_demand_model(prices_historical_in_current_season, demand_historical_in_current_season, min_points=8)
    gamma = _fit_competitor_reaction(prices_historical_in_current_season, min_points=8)

    information_dump["beta"] = beta
    information_dump["gamma"] = gamma

    # fallback reactive
    price_reactive = _reactive_price(last_price, last_demand)

    price_dp = None
    if beta is not None and gamma is not None:
        # si modèle demande suspect (β1 positif), on réduit la confiance
        β1 = float(beta[1])
        trust_delta = 15.0 if β1 < 0 else 10.0
        horizon = 6 if β1 < 0 else 4

        price_dp = _dp_price(
            beta=beta,
            gamma=gamma,
            day=day,
            remaining_capacity=remaining_capacity,
            last_price=last_price,
            last_comp_price=last_comp_price,
            competitor_has_capacity=competitor_has_capacity_current_period_in_current_season,
            horizon=horizon,
            discount=0.985,
            trust_delta=trust_delta,
        )

        # bonus léger si concurrent out-of-capacity (tu peux pousser un peu)
        if price_dp is not None and not competitor_has_capacity_current_period_in_current_season:
            price_dp = float(np.clip(price_dp * 1.03, 5, 100))

    # mixture DP + reactive (comme tu faisais, mais un peu plus DP)
    if price_dp is not None:
        candidate = 0.8 * price_dp + 0.2 * price_reactive
    else:
        candidate = price_reactive

    # trust region final autour du dernier prix (anti-chaos)
    new_price = float(np.clip(candidate, last_price * 0.75, last_price * 1.25))
    new_price = float(np.clip(new_price, 5, 100))

    information_dump["last_price"] = new_price
    return new_price, information_dump
