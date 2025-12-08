from typing import Any, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize_scalar


# Estimer une courbe de demande linéaire d ≈ β0 + β1 p_ours + β2 p_comp
#à partir de l'historique d'une saison (prix, prix conc, demande), on calcule les coeff de la reg lin (moindres carréss)

def _fit_demand_model(
    prices_historical_in_current_season: Union[np.ndarray, None], #tableau 2D L1: nos prix passés, L2: prix conc 
    demand_historical_in_current_season: Union[np.ndarray, None], #tableau 1D demandes
    min_points: int = 6, #nb minimum de points pr faire la reg
):
    if prices_historical_in_current_season is None or demand_historical_in_current_season is None: 
        return None #verif qu'on a bien les données
    prices = np.asarray(prices_historical_in_current_season, dtype=float)
    demand = np.asarray(demand_historical_in_current_season, dtype=float)
    #forcer type float pr calcul matriciel 
    if prices.ndim != 2 or prices.shape[0] < 1: #verif la forme du tab de prix
        return None
    n_obs = min(prices.shape[1], demand.shape[0])
    #prices.shape[1] = nombre de jours où on a un prix
    #demand.shape[0] = nombre de jours où on a une demande
    # On prend le minimum des deux pour être sûr 
    # qu’on a bien un couple (prix, demande) pour chaque jour
    if n_obs < min_points:
        return None
    #extraire nos prix et ceux du concurrent
    p_ours = prices[0, -n_obs:]
    if prices.shape[0] > 1:
        p_comp = prices[1, -n_obs:]
    else:
        p_comp = np.full_like(p_ours, p_ours.mean())
    #extraire la demande correspondante
    d = demand[-n_obs:]
    # construire la matrice de reg
    X = np.column_stack([np.ones_like(p_ours), p_ours, p_comp])
    # notre reg lin (moindres carrés) 
    try:
        beta, *_ = np.linalg.lstsq(X, d, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if len(beta) != 3:
        return None
    return beta  # β0, β1, β2 ça renvoie le triplet 

#en gros : À chaque période, je regarde l’historique récent 
# de la saison ( prix, prix du concurrent, demandes)
#je lance une reg lin pr approximer la demande

# 
# estimer la réaction du concurrent 
# estimer une relation lin entre prix et prix conc : p2 ≈ γ0 + γ1 p1

def _fit_competitor_reaction(
    prices_historical_in_current_season: Union[np.ndarray, None],
    min_points: int = 6,
):
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
        gamma, *_ = np.linalg.lstsq(X, p_comp, rcond=None)
    except np.linalg.LinAlgError:
        return None

    if len(gamma) != 2:
        return None

    return gamma  # γ0, γ1

    #en gros : 
    # si γ1 positif et proche de 1, le conc s’aligne sur moi.
    # si très différent,adapter strat DP 
    # si pas assez d’historique:
    # strat réactive : en si beta et gamma pas fiables 
    #alors ajuster prix en fonction du dernier prix et de derniere demande (pid)

# myopic revenue maximization 
# prix myope: prix opt pr le même jour
#optimisateur court terme 
#on maximise le revenue issue de
#  la prédiction de la demande et prix conc
def _myopic_price(beta, gamma):
    β0, β1, β2 = beta
    γ0, γ1 = gamma

    if β1 >= -1e-4:
        return None

    def neg_rev(p1):
        p2 = γ0 + γ1 * p1
        d = max(β0 + β1*p1 + β2*p2, 0)
        return -(p1 * d)

    try:
        #prix qui maximise le revenu immediat selon modeles 
        sol = minimize_scalar(neg_rev, bounds=(5, 100), method="bounded")
        return float(sol.x)
    except Exception:
        return None
# en gros :
#yope:prix qui maximise le revenu du jour, on s'en foutde demain.
# estime la demande en fonctde notre prix et de celui du conc
#teste tous les prix possibles entre 5,100 pour trouver celui qui maximise p × d(p).



# DP ;one step lookhead 
#là on prend en compte le lendemain aussi contrairement à myopic
#

def _lookahead_price(beta, gamma, delta=0.95):
    β0, β1, β2 = beta
    γ0, γ1 = gamma
    # rmême modele que myopic
    def revenue(p1):
        p2 = γ0 + γ1 * p1 #reaction du conc
        d = max(β0 + β1*p1 + β2*p2, 0)#demande prevue
        return p1 * d #revenu du jour
    #trouver le meilleur prix de demain
    #calcul prix optimal
    try:
        res_future = minimize_scalar(
            lambda p: -revenue(p),
            bounds=(5, 100),
            method="bounded"
        )
        p_future = float(res_future.x)
        V_future = revenue(p_future) #valeur future poss
    except Exception:
        return None
    # fonction objectif DP
    def dp_neg_objective(p1):
        return -(revenue(p1) + delta * V_future)
    try:
        sol = minimize_scalar(
            dp_neg_objective,
            bounds=(5, 100),
            method="bounded"
        )
        return float(sol.x)
    except Exception:
        return None
    #en gros :
    #ne approximation de Bellman (dp simplifié)
    # un agent non-myope
    # qui anticipe la réaction future du marché


# la strat réactive (le thermostat)
def _reactive_price(last_price, last_demand):
    if last_demand is None:
        return last_price

    if last_demand >= 3:
        new_p = last_price * 1.05
    elif last_demand == 0:
        new_p = last_price * 0.95
    else:
        new_p = last_price

    return float(np.clip(new_p, 5, 100))




# Fonct principale DPC

def p(
        current_selling_season: int,
        selling_period_in_current_season: int,
        prices_historical_in_current_season: Union[np.ndarray, None],
        demand_historical_in_current_season: Union[np.ndarray, None],
        competitor_has_capacity_current_period_in_current_season: bool,
        information_dump: Optional[Any] = None,
) -> Tuple[float, Any]:

    # INIT
    if information_dump is None:
        information_dump = {
            "last_price": 60.0,
            "beta": None,
            "gamma": None,
        }

    last_price = float(information_dump.get("last_price", 60.0))

    # récupérer dernier prix
    if prices_historical_in_current_season is not None:
        arr = np.asarray(prices_historical_in_current_season, float)
        if arr.ndim == 2 and arr.shape[1] > 0:
            last_price = float(arr[0, -1])

    # récupérer dernière demande
    last_demand = None
    if demand_historical_in_current_season is not None and len(demand_historical_in_current_season) > 0:
        last_demand = float(demand_historical_in_current_season[-1])

    # MODELES
    beta = _fit_demand_model(prices_historical_in_current_season,
                             demand_historical_in_current_season,
                             min_points=6)
    gamma = _fit_competitor_reaction(prices_historical_in_current_season, min_points=6)

    information_dump["beta"] = beta
    information_dump["gamma"] = gamma

    price_model = None

    # --------- DP LOOKAHEAD SI MODEL ≠ None ---------
    if beta is not None and gamma is not None:
        price_dp = _lookahead_price(beta, gamma)

        if price_dp is not None:
            price_model = price_dp
        else:
            price_model = _myopic_price(beta, gamma)

        # bonus si concurrent out-of-capacity
        if price_model is not None and not competitor_has_capacity_current_period_in_current_season:
            price_model *= 1.03
            price_model = float(np.clip(price_model, 5, 100))

    
    price_reactive = _reactive_price(last_price, last_demand)

    # COMBINAISON : 70% DP / 30% réactif
    if price_model is not None:
        candidate = 0.7*price_model + 0.3*price_reactive
    else:
        candidate = price_reactive

    # TRUST REGION
    new_price = float(np.clip(candidate,
                              last_price * 0.7,
                              last_price * 1.3))

    new_price = float(np.clip(new_price, 5, 100))

    information_dump["last_price"] = new_price

    return new_price, information_dump

