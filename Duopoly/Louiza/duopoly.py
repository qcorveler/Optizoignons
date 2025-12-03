from typing import Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import sys
import pyomo.environ as pyo

def p(
        current_selling_season: int,
        selling_period_in_current_season: int,
        prices_historical_in_current_season: Union[np.ndarray, None],
        demand_historical_in_current_season: Union[np.ndarray, None],
        competitor_has_capacity_current_period_in_current_season: bool,
        information_dump=Optional[Any],
) -> Tuple[float, Any] :
    
    """
    Ajuste dynamiquement le prix en fonction des ventes réelles et de la courbe de référence.
    
    Args:
        current_selling_season (int) : id de la saison courante
        selling_period_in_current_season (int): id du jour dans la saison courante
        prices_historical_in_current_season (liste) : 2-dim array avec les prix jusqu'ici (price_competitor et price)
        demand_historical_in_current_season (liste) : 1-dim array des demandes jusqu'ici (= proportion du stock vendu à ce jour (entre 0 et 1)?)
        competitor_has_capacity_current_period_in_current_season (bool) : Vrai si le competitor a encore du stock, faux sinon

    Returns:
        float: le nouveau prix ajusté
    """
    
    if (selling_period_in_current_season <= 1):
        return np.random.randint(30, 90), None


    # TODO : ajouter une façon de fournir des fichiers csv
    # reference = pd.read_csv("mean_curve.csv", header=None).squeeze()
    reference = [0.004,0.0071666666666666,0.015,0.0196666666666666,0.026,0.0336666666666666,0.0373333333333333,0.044,0.0508333333333333,0.0566666666666666,0.0615,0.0671666666666666,0.0763333333333333,0.084,0.0931666666666666,0.1011666666666666,0.1098333333333333,0.1153333333333333,0.1205,0.1268333333333333,0.1315,0.1384999999999999,0.144,0.1543333333333333,0.1604999999999999,0.1659999999999999,0.1729999999999999,0.1809999999999999,0.1886666666666666,0.1964999999999999,0.2036666666666666,0.208,0.2146666666666666,0.2206666666666666,0.2286666666666666,0.2339999999999999,0.2404999999999999,0.247,0.2526666666666666,0.2576666666666666,0.2638333333333333,0.2711666666666666,0.2766666666666666,0.2828333333333334,0.2901666666666667,0.296,0.3023333333333333,0.3065,0.3113333333333333,0.3211666666666666,0.3273333333333333,0.3351666666666666,0.3451666666666666,0.357,0.3666666666666667,0.376,0.3825,0.3891666666666667,0.399,0.4081666666666666,0.4186666666666666,0.4293333333333333,0.4396666666666666,0.4531666666666666,0.4688333333333334,0.4811666666666667,0.4965,0.5121666666666667,0.5248333333333334,0.5375,0.5516666666666666,0.5660000000000001,0.582,0.5988333333333333,0.6205000000000002,0.6368333333333334,0.6515000000000001,0.6711666666666666,0.6871666666666666,0.7021666666666668,0.7201666666666667,0.7355,0.7525,0.7696666666666666,0.7885000000000001,0.8060000000000002,0.8255,0.8418333333333334,0.8571666666666667,0.8733333333333334,0.886,0.8988333333333334,0.9128333333333334,0.923,0.934,0.945,0.9575,0.9661666666666666,0.9753333333333332,0.9813333333333334]

    # On récupère l'indice de la période précédente
    last_period_index = selling_period_in_current_season-1

    # On récupère le stock cible vendu selon la courbe référence 
    # target_stock_saled = reference.iloc[last_period_index]
    target_stock_saled = reference[last_period_index]

    # On récupère le stock réellement vendu pendant cette saison
    actual_stock_saled = np.sum(demand_historical_in_current_season) / 80 # <- 80 est le stock total
    print(f"ventes théorique à ce stade : {target_stock_saled}")
    print(f"ventes réelles proportion : {actual_stock_saled}")

    # Calcul de l’écart relatif
    delta = (actual_stock_saled - target_stock_saled) / target_stock_saled
    print(f'"delta : {delta}"')
    
    # Mise à jour du prix
    last_price = prices_historical_in_current_season[0][last_period_index-1] # 0 correspond à nos prix
    print(f"Last price : {last_price}")
    alpha = 0.2 # valeur de sensibilité de l'ajustement du prix
    new_price =  last_price * (1 + alpha * delta)
    print(f"new price : {new_price}")
    
    # On évite que le prix devienne négatif ou trop bas trop vite ou trop haut trop vite
    new_price = min(max(new_price, last_price * 0.8), last_price * 1.2) 
    new_price = max(min(new_price, 100), 5) # Faire en sorte que ce soit capé entre 5 et 100
    return new_price, None
