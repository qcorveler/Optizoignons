import numpy as np
import pandas as pd
from typing import Any, Optional, Tuple, Union

def p(
    current_selling_season: int,
    selling_period_in_current_season: int,
    prices_historical_in_current_season: Union[np.ndarray, None],
    demand_historical_in_current_season: Union[np.ndarray, None],
    competitor_has_capacity_current_period_in_current_season: bool,
    information_dump: Optional[Any],
) -> Tuple[float, Any]:

    # -----------------------------
    # 1. Début de saison
    # -----------------------------
    if selling_period_in_current_season <= 1:
        return float(np.random.uniform(30, 90)), None

    # Charger la booking curve cible
    reference_curve = pd.read_csv("mean_curve.csv", header=None).squeeze().to_numpy()
    max_idx = len(reference_curve) - 1
    idx = min(selling_period_in_current_season - 1, max_idx)

    # Données historiques
    demand_hist = np.array(demand_historical_in_current_season, dtype=float)
    sold = np.sum(demand_hist)
    stock_total = 80

    actual_ratio = sold / stock_total
    target_ratio = reference_curve[idx]

    deviation_now = abs(actual_ratio - target_ratio)

    # Récup prix précédent
    prices_hist = np.array(prices_historical_in_current_season, dtype=float)
    last_price = prices_hist[0][idx - 1]

    # --------------------------------------------
    # 2. Définir la fonction objectif φ(p)
    #    → minimise la déviation à la booking curve
    # --------------------------------------------
    def objective(p):
        # modèle simple de demande relative : décroissance linéaire validée dans les slides
        # (on n'utilise pas les paramètres exacts de la demande du prof : interdit)
        elasticity = 0.004
        predicted_sales_next = max(0, 1.0 - elasticity * (p - last_price))

        new_actual_ratio = (sold + predicted_sales_next) / stock_total
        return abs(new_actual_ratio - target_ratio)

    # --------------------------------------------
    # 3. Armijo-like line search sur le prix
    # --------------------------------------------
    candidate_steps = [1.0, 0.5, 0.2, 0.1, -0.1, -0.2, -0.5]  # variations possibles
    best_price = last_price
    best_value = deviation_now

    for s in candidate_steps:

        p_try = last_price * (1 + s)

        # respect des bornes
        p_try = np.clip(p_try, 5, 100)

        val = objective(p_try)

        if val < best_value:  # Armijo-like: φ(α) < φ(0)
            best_value = val
            best_price = p_try

    # --------------------------------------------
    # 4. Ajustement si le concurrent n’a plus de stock
    #    (cf. slide Revenue Curves : si concurrent stock-out → monter prix)
    # --------------------------------------------
    if not competitor_has_capacity_current_period_in_current_season:
        best_price *= 1.10

    # garde-fous finaux
    best_price = np.clip(best_price, 5, 100)

    return float(best_price), None
