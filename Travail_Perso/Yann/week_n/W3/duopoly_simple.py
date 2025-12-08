from typing import Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import uuid
import pickle
import matplotlib.pyplot as plt

file_path = "target_sales_curve_quentin.pkl"
with open(file_path, "rb") as f:
    courbe = pickle.load(f)

def _initialize_data_feedback():
    """Initialize the feedback data object"""

    ## Try to load duopoly feedback data from disk
    try:
        with open('duopoly_feedback.data', 'rb') as handle:
            feedback = pickle.load(handle)
        return feedback
    except:
        return {

            # keep track of historical observables
            'history': (
                pd.DataFrame({
                    'simulation': '',
                    'day': [],
                    'season': [],
                    'demand': [],
                    'own_price': [],
                    'competitor_price': [],
                    'remaining_capacity': [],
                    'revenue': [],
                    'cumulative_revenue': [],
                    'factors' : None,
                })
            ),
            'current_simulation' : '',
            'cumulative_revenue_current_selling_season' : 0
        }

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
    demand = demand_historical_in_current_season
    prices = prices_historical_in_current_season

    # --- Initialisation au premier jour ---
    if day == 1:
        information_dump = _initialize_data_feedback()
        if season == 1:
            information_dump['current_simulation'] = str(uuid.uuid4())
        information_dump['cumulative_revenue_current_selling_season'] = 0
        information_dump['factor'] = 1.0  # facteur initial
        information_dump['last_base_price'] = np.random.uniform(30, 50)  # prix initial
        information_dump['price_today'] = information_dump['last_base_price']
        return information_dump['price_today'], information_dump

    # --- Récupération des infos précédentes ---
    factor = information_dump.get('factor', 1.0)
    last_base_price = information_dump.get('last_base_price', 40.0)
    price_today = information_dump.get('price_today', last_base_price)

    # --- Tous les 5 jours : recalcul du prix de base ---
    if day % 5 == 0:
        # Recalcule le facteur si besoin
        capacity_obj = courbe[day]['cap_util']
        remaining_capacity = 80 - np.sum(demand)
        if remaining_capacity < capacity_obj:
            factor = (remaining_capacity / 80)/capacity_obj
            print(f"→ Facteur mis à jour à {factor:.3f} au jour {day}")

        # Calcul de la moyenne des 5 derniers prix adverses
        n_days = prices.shape[1]
        window = 5 if n_days >= 5 else n_days
        mean_competitor_price = np.mean(prices[1, -window:])

        # Nouveau prix de base
        price_today = mean_competitor_price * factor
        last_base_price = price_today

    else:
        # Autres jours : diminution de 5%
        price_today *= 0.95

    # --- Mise à jour de l'information ---
    revenue = demand[-1] * prices[0, -1]
    rev_sofar = information_dump['cumulative_revenue_current_selling_season']

    info_row = {
        'simulation': information_dump['current_simulation'],
        'day': day - 1,
        'season': season,
        'demand': demand[-1],
        'own_price': prices[0, -1],
        'competitor_price': prices[1, -1],
        'remaining_capacity': 80 - np.sum(demand),
        'revenue': revenue,
        'cumulative_revenue': rev_sofar + revenue,
        'factor': factor,
        'price_today': price_today,
    }

    information_dump['history'] = pd.concat(
        [information_dump['history'], pd.DataFrame([info_row])],
        ignore_index=True
    )

    information_dump['cumulative_revenue_current_selling_season'] = rev_sofar + revenue
    information_dump['factor'] = factor
    information_dump['last_base_price'] = last_base_price
    information_dump['price_today'] = price_today

    # --- Sauvegarde en fin de saison ---
    if day >= 100:
        with open('duopoly_feedback.data', 'wb') as handle:
            pickle.dump(information_dump, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return price_today, information_dump


