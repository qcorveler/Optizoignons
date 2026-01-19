from typing import Any, Optional, Tuple, Union
import numpy as np
import xgboost as xgb
from functools import lru_cache
#typing: pour annoter les types (utile pour le prof, lisibilité).
#numpy as np: tableaux, clips, opérations vectorisées.
#xgboost as xgb: modèle de régression (demand prediction).
#lru_cache: mémoïsation pour accélérer la DP offline (évite de recalculer V(...) 100k fois).

# ============================================================
# Utils
# ============================================================

#bornes du prix final renvoyé à la plateforme
PRICE_FLOOR = 5.0
PRICE_CAP   = 100.0
#bornes pour “nettoyer” le prix concurrent (parfois la donnée est bruitée / extrême / absurde)
PC_CLIP_LO  = 0.1
PC_CLIP_HI  = 120.0

def _clip_pc(pc: float) -> float:
    return float(np.clip(pc, PC_CLIP_LO, PC_CLIP_HI))

def _bool_to_int(b: bool) -> float:
    return 1.0 if bool(b) else 0.0


# ============================================================
# XGBoost demand model (trained occasionally, reused online)
# Features: [p_ours, p_comp_clipped, cap_int]
# ============================================================

#train XGBoost regression model to predict demand
def _fit_demand_model_xgb(
    prices_hist: Union[np.ndarray, None],#prices historiques (2D: our price + competitor price)
    demand_hist: Union[np.ndarray, None],#demand observed (1D)
    cap_hist: Optional[np.ndarray], #competitor capacity history encoded as 0/1 (1D)
    min_points: int = 80,#min points to train
    num_boost_round: int = 80 #number of boosting rounds
) -> Optional[xgb.Booster]: #return trained XGBoost model or None if not enough data

#sanity checks
    #if not enough data, return None
    if prices_hist is None or demand_hist is None:
        return None
    #conert to numpy arrays float
    P = np.asarray(prices_hist, dtype=float)
    d = np.asarray(demand_hist, dtype=float)
    #if shapes invalid, return None
    if P.ndim != 2 or P.shape[1] == 0:
        return None
    #d must be 1D and match number of observations
    n = min(P.shape[1], d.shape[0])
    #if not enough data, return None
    if n < min_points:
        return None
    #get last n observations
    p_ours = P[0, -n:]
    #get competitor prices (clip extreme values)
    if P.shape[0] >= 2:
        p_comp = P[1, -n:]
    else:
        p_comp = np.full_like(p_ours, np.mean(p_ours))#we invent a competitor price equal to our mean price
    #we clip competitor prices (anti-outliers)
    p_comp = np.array([_clip_pc(x) for x in p_comp], dtype=float)
    #get competitor capacity history (0/1)
    if cap_hist is None or len(cap_hist) < n:
        cap_int = np.full(n, 1.0)  #=1 we assume competitor has capacity if no data
    else:
        cap_int = cap_hist[-n:].astype(float) #last n values as float
    #construction dataset and train XGBoost
    #X: n rows, 3 columns (our price, competitor price, competitor capacity )
    X = np.column_stack([p_ours, p_comp, cap_int])
    #corresponding demand values
    y = d[-n:]
    #create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X, label=y)
    #set XGBoost parameters
    params = {
        "objective": "reg:squarederror", #reg MSE
        "eval_metric": "rmse",
        "max_depth": 3,#limite overfitting
        "eta": 0.08, #learning rate 
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 5,#regularization 
        "lambda": 1.0,#L2 regularization 
        "verbosity": 0,
        "seed": 42,
    }
    #train model
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    return model


# ============================================================
# OFFLINE DP 
# DO NOT call this inside p() (too slow).
# goal : choose price to maximize discounted revenue proxy over horizon
# ============================================================

def dp_offline_best_price(
    model_demand: xgb.Booster,
    competitor_has_capacity: bool,
    last_comp_price: float,
    remaining_capacity: float = 80.0,
    last_price: float = 60.0,
    horizon: int = 6,#periods to look ahead
    discount: float = 0.985,
    trust_delta: float = 15.0,#no big jumps from last price
    step: float = 2.0,#2eurs pr step in DP grid
) -> float:
    """
    Offline DP that chooses price to maximize discounted revenue proxy over horizon.
    State: (t, cap, last_price, last_comp_price)
    Transition: competitor price assumed sticky (no gamma), demand by XGB.
    """

    cap_int = _bool_to_int(competitor_has_capacity)
    base_grid = np.arange(PRICE_FLOOR, PRICE_CAP + 1e-9, step)#all possible prices

    #candidate prices within trust region
    #we only consider prices within trust_delta of last price
    def candidates(p_prev: float) -> np.ndarray:
        lo = max(PRICE_FLOOR, p_prev - trust_delta)
        hi = min(PRICE_CAP, p_prev + trust_delta)
        cand = base_grid[(base_grid >= lo) & (base_grid <= hi)]
        return cand if cand.size else base_grid
    #demand prediction function using XGBoost model
    #build a feature row and predict mu mean demand
    #cut at 0 (no negative demand)
    #cut at cap_left(cannot sell more than remaining capacity)
    def predict_demand(p_ours: float, p_comp: float, cap_left: float) -> float:
        X = np.array([[p_ours, _clip_pc(p_comp), cap_int]], dtype=float)
        mu = float(model_demand.predict(xgb.DMatrix(X))[0])
        mu = max(0.0, mu)
        return min(mu, float(cap_left))
    
    #simple competitor price prediction (sticky)
    def predict_comp_next(p_comp_prev: float) -> float:
        # simple sticky competitor (offline simplification)
        return float(_clip_pc(p_comp_prev))

    def disc_price(p: float) -> int:
        return int(np.round(p / step) * step)

    def disc_cap(c: float) -> int:
        return int(np.round(c))

    @lru_cache(maxsize=200_000)
    def V(t: int, cap_i: int, p_last_i: int, p_comp_i: int) -> float:
        if cap_i <= 0 or t >= horizon:
            return 0.0

        cap_left = float(cap_i)
        p_last = float(p_last_i)
        p_comp = float(p_comp_i)

        best = -1e18
        for p in candidates(p_last):
            d = predict_demand(p, p_comp, cap_left)
            r = p * d
            p_comp_next = predict_comp_next(p_comp)
            cap_next = disc_cap(cap_left - d)
            val = r + discount * V(t + 1, cap_next, disc_price(p), disc_price(p_comp_next))
            if val > best:
                best = val
        return float(best)

    cap0 = disc_cap(remaining_capacity)
    p_last0 = disc_price(last_price)
    p_comp0 = disc_price(_clip_pc(last_comp_price))

    best_p, best_val = float(np.clip(last_price, PRICE_FLOOR, PRICE_CAP)), -1e18
    for p in candidates(float(p_last0)):
        d = predict_demand(p, float(p_comp0), float(cap0))
        r = p * d
        cap_next = disc_cap(float(cap0) - d)
        val = r + discount * V(1, cap_next, disc_price(p), p_comp0)
        if val > best_val:
            best_val, best_p = val, float(p)

    return float(np.clip(best_p, PRICE_FLOOR, PRICE_CAP))


# ============================================================
# FAST ONLINE POLICY (for platform request time)
# Derived from your binned revenue curves + can be justified by offline DP.
# ============================================================

def _fast_policy_target(cap: bool, last_comp_price: float) -> float:
    pc = _clip_pc(last_comp_price)

    if not cap:
        # competitor no capacity -> revenue peak around ~40-45
        base = 42.0
        return base

    # competitive -> revenue peak around ~35-40 with mild adjustment
    if pc < 25:
        return 35.0
    if pc > 70:
        return 40.0
    return 37.0


# ============================================================
# Main function required by competition
# ============================================================

def p(
    current_selling_season: int,
    selling_period_in_current_season: int,
    prices_historical_in_current_season: Union[np.ndarray, None],
    demand_historical_in_current_season: Union[np.ndarray, None],
    competitor_has_capacity_current_period_in_current_season: bool,
    information_dump: Optional[Any] = None,
) -> Tuple[float, Any]:

    # ---- init memory ----
    if information_dump is None or not isinstance(information_dump, dict):
        information_dump = {
            "last_price": 60.0,
            "model_demand": None,
            "n_trained": 0,
            "cap_hist": [],
        }

    last_price = float(information_dump.get("last_price", 60.0))
    last_comp_price = 60.0

    # ---- read last observed prices ----
    if prices_historical_in_current_season is not None:
        arr = np.asarray(prices_historical_in_current_season, dtype=float)
        if arr.ndim == 2 and arr.shape[1] > 0:
            last_price = float(arr[0, -1])
            if arr.shape[0] >= 2:
                last_comp_price = float(arr[1, -1])

    last_comp_price = _clip_pc(last_comp_price)

    # ---- update cap history (0/1) ----
    cap_hist = information_dump.get("cap_hist", [])
    if not isinstance(cap_hist, list):
        cap_hist = []
    cap_hist.append(_bool_to_int(competitor_has_capacity_current_period_in_current_season))
    information_dump["cap_hist"] = cap_hist

    # ---- train XGBoost occasionally (NOT every call) ----
    n_obs = 0
    if prices_historical_in_current_season is not None:
        arr = np.asarray(prices_historical_in_current_season, dtype=float)
        if arr.ndim == 2:
            n_obs = arr.shape[1]

    model_demand = information_dump.get("model_demand", None)
    n_trained = int(information_dump.get("n_trained", 0))

    # retrain only if enough new data
    if (n_obs >= 80) and (n_obs - n_trained >= 15):
        cap_hist_arr = np.asarray(cap_hist, dtype=float)
        model_demand = _fit_demand_model_xgb(
            prices_historical_in_current_season,
            demand_historical_in_current_season,
            cap_hist_arr,
            min_points=80,
            num_boost_round=80,
        )
        information_dump["model_demand"] = model_demand
        information_dump["n_trained"] = n_obs

    # ---- FAST decision (no DP online) ----
    target = _fast_policy_target(
        competitor_has_capacity_current_period_in_current_season,
        last_comp_price
    )

    # smooth move towards target (stable + avoids oscillations)
    alpha = 0.30
    candidate = (1 - alpha) * last_price + alpha * target

    # trust region (avoid extreme jumps)
    candidate = float(np.clip(candidate, last_price * 0.75, last_price * 1.25))

    new_price = float(np.clip(candidate, PRICE_FLOOR, PRICE_CAP))

    information_dump["last_price"] = new_price
    return new_price, information_dump
