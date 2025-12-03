from typing import Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import sys
import pyomo.environ as pyo
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

# ==========================
#  Online PyTorch Regressor
# ==========================

class OnlineLinearRegressor(nn.Module):
    def __init__(self, lr=0.01, l2=0.01):
        super().__init__()
        self.lr = lr
        self.l2 = l2
        self.weights = None
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def _init_weights(self, n_features):
        # initialize or re-initialize weights if feature size changes
        if self.weights is None or self.weights.shape[0] != n_features:
            w = torch.zeros(n_features, dtype=torch.float32)
            self.weights = nn.Parameter(w)

    def predict(self, x_dict: dict):
        x = torch.tensor(list(x_dict.values()), dtype=torch.float32)
        self._init_weights(len(x))
        with torch.no_grad():
            return (x @ self.weights + self.bias).item()

    # compatibility with .predict_one used elsewhere
    def predict_one(self, x_dict: dict):
        return self.predict(x_dict)

    def learn(self, x_dict: dict, y: float):
        x = torch.tensor(list(x_dict.values()), dtype=torch.float32)
        y = torch.tensor([y], dtype=torch.float32)

        self._init_weights(len(x))

        pred = x @ self.weights + self.bias
        # simple squared error + L2 regularization
        loss = (pred - y) ** 2 + self.l2 * (self.weights @ self.weights)

        # compute gradients
        loss.backward()

        # manual SGD step: update .data to keep Parameter objects intact
        with torch.no_grad():
            self.weights.data -= self.lr * self.weights.grad.data
            self.bias.data -= self.lr * self.bias.grad.data

        # zero gradients safely
        if self.weights.grad is not None:
            self.weights.grad.zero_()
        if self.bias.grad is not None:
            self.bias.grad.zero_()

    # compatibility with .learn_one used elsewhere
    def learn_one(self, x_dict: dict, y: float):
        self.learn(x_dict, y)

# ============================================================
#  METRICS pour suivre les performances
# ============================================================

class OnlineMetric:
    def update(self, y, y_pred):
        raise NotImplementedError

    def get(self):
        raise NotImplementedError


class MAE(OnlineMetric):
    def __init__(self):
        self.s = 0
        self.n = 0

    def update(self, y, y_pred):
        self.s += abs(y - y_pred)
        self.n += 1

    def get(self):
        return self.s / self.n if self.n > 0 else 0


class MSE(OnlineMetric):
    def __init__(self):
        self.s = 0
        self.n = 0

    def update(self, y, y_pred):
        self.s += (y - y_pred) ** 2
        self.n += 1

    def get(self):
        return self.s / self.n if self.n > 0 else 0


class RMSE(OnlineMetric):
    def __init__(self):
        self.mse = MSE()

    def update(self, y, y_pred):
        self.mse.update(y, y_pred)

    def get(self):
        return self.mse.get() ** 0.5


class R2(OnlineMetric):
    def __init__(self):
        self.ss_res = 0
        self.ss_tot = 0
        self.n = 0
        self.y_mean = 0

    def update(self, y, y_pred):
        self.n += 1
        old_mean = self.y_mean
        self.y_mean += (y - self.y_mean) / self.n
        self.ss_tot += (y - old_mean) * (y - self.y_mean)
        self.ss_res += (y - y_pred) ** 2

    def get(self):
        if self.ss_tot == 0:
            return 0
        return 1 - self.ss_res / self.ss_tot

# ==========================
#  Main function p()
# ==========================

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
    
    # init model and metrics on first period of each season
    if (selling_period_in_current_season <= 1):
        model = OnlineLinearRegressor(lr=0.00001, l2=0.01)

        metrics_list = {
            "R²": R2(),
            "MAE": MAE(),
            "MSE": MSE(),
            "RMSE": RMSE()
        }

        metric_list_simple = {
            "R²": R2(),
            "MAE": MAE(),
            "MSE": MSE(),
            "RMSE": RMSE()
        }
        return np.random.randint(30, 90), {"model": model, "metrics": metrics_list, "copycat_metrics": metric_list_simple, "use_model" : False}

    # --- Load information dump ---
    model: OnlineLinearRegressor = information_dump["model"]
    metrics_list = information_dump["metrics"]
    copycat_metrics = information_dump["copycat_metrics"]
    use_model = information_dump["use_model"]

    # On récupère l'indice de la période précédente
    last_period_index = selling_period_in_current_season-1

    ### Prediction of the price of the competitor

    competitor_price_lag1 = prices_historical_in_current_season[1][last_period_index-1]
    competitor_price_lag2 = prices_historical_in_current_season[1][last_period_index-2] if last_period_index-2 >= 0 else 50.0 # 1 correspond au compétiteur (-2 cause index begin at 0 although period begin at 1)  
    price_lag2 = prices_historical_in_current_season[0][last_period_index-2] if last_period_index-2 >= 0 else 50.0 

    # training process
    if current_selling_season <= 25:
        x = {
            'selling_period': selling_period_in_current_season,
            'price_competitor_lag1': competitor_price_lag2,
            'price_self_lag1': price_lag2
            # 'price_competitor_lag2': history['price_competitor'][-2] if selling_period_in_current_season>2 else 0.0,
        }
        y = competitor_price_lag1

        # predict and learn
        y_pred = model.predict_one(x)
        y_pred_copycat = competitor_price_lag2

        # Update metrics
        for metric in metrics_list.values():
            if y_pred is not None:
                metric.update(y, y_pred)
        for metric in copycat_metrics.values():
            if y_pred_copycat is not None:
                metric.update(y, y_pred_copycat)

        model.learn_one(x, y)

    # determination of model parameters for the end of the competition
    if current_selling_season == 25:
        FEEDBACK_OBJECT = {
            "Final model metrics:" : {name: f"{metric.get():.4f}" for name, metric in metrics_list.items()},
            "Copycat model metrics:" : {name: f"{metric.get():.4f}" for name, metric in copycat_metrics.items()},
        }
        use_model = metrics_list["R²"].get() > copycat_metrics["R²"].get()

        FEEDBACK_OBJECT["Use model for final seasons:"] = use_model

        with open('duopoly_feedback.data', 'wb') as handle:
            pickle.dump(FEEDBACK_OBJECT, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
    # prediction of the competitor price 
    competitor_price_prediction = competitor_price_lag1 # for the 25 first seasons, we use the lag1 price as prediction
    if current_selling_season >= 25 :
        if use_model : 
            competitor_price_prediction = model.predict_one({
                'selling_period': selling_period_in_current_season,
                'price_competitor_lag1': competitor_price_lag1,
                'price_self_lag1': prices_historical_in_current_season[0][last_period_index-1]
            })



    # TODO : ajouter une façon de fournir des fichiers csv
    # reference = pd.read_csv("mean_curve.csv", header=None).squeeze()
    reference = [0.004,0.0071666666666666,0.015,0.0196666666666666,0.026,0.0336666666666666,0.0373333333333333,0.044,0.0508333333333333,0.0566666666666666,0.0615,0.0671666666666666,0.0763333333333333,0.084,0.0931666666666666,0.1011666666666666,0.1098333333333333,0.1153333333333333,0.1205,0.1268333333333333,0.1315,0.1384999999999999,0.144,0.1543333333333333,0.1604999999999999,0.1659999999999999,0.1729999999999999,0.1809999999999999,0.1886666666666666,0.1964999999999999,0.2036666666666666,0.208,0.2146666666666666,0.2206666666666666,0.2286666666666666,0.2339999999999999,0.2404999999999999,0.247,0.2526666666666666,0.2576666666666666,0.2638333333333333,0.2711666666666666,0.2766666666666666,0.2828333333333334,0.2901666666666667,0.296,0.3023333333333333,0.3065,0.3113333333333333,0.3211666666666666,0.3273333333333333,0.3351666666666666,0.3451666666666666,0.357,0.3666666666666667,0.376,0.3825,0.3891666666666667,0.399,0.4081666666666666,0.4186666666666666,0.4293333333333333,0.4396666666666666,0.4531666666666666,0.4688333333333334,0.4811666666666667,0.4965,0.5121666666666667,0.5248333333333334,0.5375,0.5516666666666666,0.5660000000000001,0.582,0.5988333333333333,0.6205000000000002,0.6368333333333334,0.6515000000000001,0.6711666666666666,0.6871666666666666,0.7021666666666668,0.7201666666666667,0.7355,0.7525,0.7696666666666666,0.7885000000000001,0.8060000000000002,0.8255,0.8418333333333334,0.8571666666666667,0.8733333333333334,0.886,0.8988333333333334,0.9128333333333334,0.923,0.934,0.945,0.9575,0.9661666666666666,0.9753333333333332,0.9813333333333334]

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
    return new_price, {"model": model, "metrics": metrics_list, "copycat_metrics": copycat_metrics, "use_model" : use_model}
