from typing import Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import uuid
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

# We load model parameters and target sales curve
model_params = pd.read_csv('/root/partcipant_folder/modeles_moyens_elasticite.csv')
with open('/root/partcipant_folder/target_sales_curve.pkl', 'rb') as f:
        courbe = pickle.load(f)
# For local testing 
# model_params = pd.read_csv('modeles_moyens_elasticite.csv')
# with open('target_sales_curve.pkl', 'rb') as f:
#         courbe = pickle.load(f)


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
                    'target_demand' : [],
                    'own_price': [],
                    'competitor_price': [],
                    'remaining_capacity': [],
                    'revenue': [],
                    'cumulative_revenue': [],
                    'demand_model_used': [],
                    'para_demand_model': [],
                })
            ),
            'current_simulation' : '',
            'cumulative_revenue_current_selling_season' : 0,
            'season_model_map': {},
            'model' : OnlineLinearRegressor(lr=0.00001, l2=0.01),
            'metrics_list' : {
                "R²": R2(),
                "MAE": MAE(),
                "MSE": MSE(),
                "RMSE": RMSE()
            },
            'metric_list_simple' : {
                "R²": R2(),
                "MAE": MAE(),
                "MSE": MSE(),
                "RMSE": RMSE()
            },
            "use_model" : False
        }
    
def est_demand(price, comp_price, para):
    demand = para[0]*price + para[1]*comp_price + para[2]
    demand = max(0,demand)
    return round(demand,3)

def get_best_price_target_demand(target_demand, 
                                 comp_price,
                                 para):
    """
    given target demand, compute the best price to generate 
    demand as close as possible to the target_demand
    
    return: tupel of best price and the corresponding demand
    """
    best_p = 0
    best_realistic_demand = 0
    a, b, c = para
    best_p = 1/a * (target_demand - b*comp_price - c)
    
    best_p = max(3, best_p)
    
    demand = est_demand(price=best_p, comp_price=comp_price, para=para)
    best_realistic_demand= min(demand, target_demand)      
    return best_p, round(best_realistic_demand)

def training_comp_price_estimator(
    model: OnlineLinearRegressor,
    competitor_price_lag2: float,
    metrics_list: dict,
    copycat_metrics: dict,
    x: dict,
    y: float,
):
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
    return model


# ==========================
#  Main function p()
# ==========================

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
    last_price = prices_historical_in_current_season[0,-1]    if day > 1 else 0
    last_comp_price = prices_historical_in_current_season[1,-1]   if day > 1 else 20

    # --- 1. Initialize Feedback (First time ever) ---
    if information_dump is None:
        information_dump = _initialize_data_feedback()
    FEEDBACK_OBJECT = {"information_dump": information_dump}    
        
    model = information_dump['model']
    metrics_list = information_dump['metrics_list']
    copycat_metrics = information_dump['metric_list_simple']
    use_model = information_dump["use_model"]

    # training process
    competitor_price_lag1 = prices_historical_in_current_season[1][day-2] if day-2 >= 0 else 50.0 # 1 correspond au compétiteur (-2 cause index begin at 0 although period begin at 1)
    competitor_price_lag2 = prices_historical_in_current_season[1][day-3] if day-3 >= 0 else 50.0 # 1 correspond au compétiteur (-3 cause index begin at 0 although period begin at 1)  
    price_lag2 = prices_historical_in_current_season[0][day-3] if day-3 >= 0 else 50.0 
    if current_selling_season <= 25:
        x = {
            'selling_period': selling_period_in_current_season,
            'price_competitor_lag1': competitor_price_lag2,
            'price_self_lag1': price_lag2
            # 'price_competitor_lag2': history['price_competitor'][-2] if selling_period_in_current_season>2 else 0.0,
        }
        y = competitor_price_lag1
        
        updated_model = training_comp_price_estimator(
            model,
            competitor_price_lag2,
            metrics_list,
            copycat_metrics,
            x,
            y,
        )
        information_dump['model'] = updated_model

    # determination of model parameters for the end of the competition
    if current_selling_season == 25:
        FEEDBACK_OBJECT["Final model metrics:"] = {name: f"{metric.get():.4f}" for name, metric in metrics_list.items()}
        FEEDBACK_OBJECT["Copycat model metrics:"] = {name: f"{metric.get():.4f}" for name, metric in copycat_metrics.items()}
        use_model = metrics_list["R²"].get() > copycat_metrics["R²"].get()

        FEEDBACK_OBJECT["Use model for final seasons:"] = use_model
        information_dump["use_model"] = use_model
        information_dump["FEEDBACK_OBJECT"] = FEEDBACK_OBJECT

    # prediction of the competitor price 
    competitor_price_prediction = competitor_price_lag1 # for the 25 first seasons, we use the lag1 price as prediction
    if current_selling_season >= 25 :
        if use_model : 
            competitor_price_prediction = model.predict_one({
                'selling_period': selling_period_in_current_season,
                'price_competitor_lag1': competitor_price_lag1,
                'price_self_lag1': prices_historical_in_current_season[0][day-2] if day-2 >= 0 else 50.0
            })

    # --- 2. Process YESTERDAY's Results (Feedback Loop) ---
    # We do this BEFORE checking if it's Day 1 of a new season, to capture Day 100 data.
    if day > 1:
        # Standard update for days 2-100
        last_demand = demand_historical_in_current_season[-1]
        last_price = prices_historical_in_current_season[0, -1]
        last_comp_price = prices_historical_in_current_season[1, -1]
        
        daily_rev = last_demand * last_price
        information_dump['cumulative_revenue_current_selling_season'] += daily_rev
        
        # Log history (Optional: can be expensive to concat every day)
        # For speed, maybe only log at end of season? 
        # Keeping your logic:
        # Note: prices_historical arguments are NOT available for day 1 call.
        pass

    # --- 3. Handle Day 1 (Season Reset) ---
    if day == 1:
        # If we just finished a season (e.g. season > 1), we archive the score
        if season > 1:
            prev_season = season - 1
            total_rev = information_dump['cumulative_revenue_current_selling_season']
            
            # Store this season's performance in a summary dict
            if 'season_scores' not in information_dump:
                information_dump['season_scores'] = {}
            information_dump['season_scores'][prev_season] = total_rev
            
            # Reset for new season
            information_dump['cumulative_revenue_current_selling_season'] = 0
        
        if season == 1:
            information_dump['current_simulation'] = str(uuid.uuid4())
            
        # Initial random price for Day 1
        price = np.random.uniform(30, 50)
        information_dump['price_today'] = price
        
        # Determine Model for this season
        if season <= 6:
            model_idx = (season - 1) % 3
            
        else:
            scores = {0: [], 1: [], 2: []}
            season_map = information_dump.get('season_model_map', {})
            season_revs = information_dump.get('season_scores', {})
            
            for s, rev in season_revs.items():
                if s in season_map:
                    m_idx = season_map[s]
                    scores[m_idx].append(rev)
            
            # Avg score per model
            avg_scores = {k: np.mean(v) if v else -1 for k, v in scores.items()}
            model_idx = max(avg_scores, key=avg_scores.get)

        # Save decision
        information_dump['season_model_map'][season] = model_idx
        information_dump['current_model_idx'] = model_idx
        
        return price, information_dump

    # --- 4. Day 2+ Logic ---
    
    # Retrieve current model parameters
    model_idx = information_dump['current_model_idx']
    para = model_params.iloc[model_idx][['coeff_price','coeff_competitor','intercept']].values
    

    # --- 5. Target Demand Calculation ---
    if day % 5 == 0 and day <= 90:
        # On doit alors mettre à jour notre besoin de demand
        capacity_obj = courbe[day+5]['cap_util']
        remaining_capacity = 1 - (80 - np.sum(demand))/80
        
        objective_demand_for_next_5_days = capacity_obj - remaining_capacity 
        target_daily_demand = (objective_demand_for_next_5_days / 5)*80
        information_dump['target_daily_demand'] = target_daily_demand

    else :
        target_daily_demand = information_dump.get('target_daily_demand', 1)
    

    # --- 6. Compute Price ---
    price, _ = get_best_price_target_demand(target_daily_demand, competitor_price_prediction, para)
    
    information_dump['price_today'] = price

    # --- Mise à jour de l'information ---
    last_demand = demand[-1] if demand is not None and len(demand) > 0 else 0
    revenue = demand[-1] * prices_historical_in_current_season[0, -1]
    rev_sofar = information_dump['cumulative_revenue_current_selling_season']

    info_row = {
        'simulation': information_dump['current_simulation'],
        'day': day,
        'season': season,
        'demand': last_demand,
        'own_price': last_price,
        'competitor_price': last_comp_price,
        'remaining_capacity': 80 - np.sum(demand),
        'revenue': revenue,
        'cumulative_revenue': rev_sofar + revenue,
        'demand_model_used': model_idx,
        'para_demand_model': para,
        'price_today': price,
        'target_demand': target_daily_demand,
    }

    information_dump['history'] = pd.concat(
        [information_dump['history'], pd.DataFrame([info_row])],
        ignore_index=True
    )
    information_dump['cumulative_revenue_current_selling_season'] = rev_sofar + revenue
    information_dump['last_base_price'] = information_dump['price_today']
    
    # --- Sauvegarde en fin de saison ---

    if day >= 100:
        with open('duopoly_feedback.data', 'wb') as handle:
            pickle.dump(information_dump, handle, protocol=pickle.HIGHEST_PROTOCOL)

    price = max(min(price, 100), 5) # Faire en sorte que ce soit capé entre 5 et 100

    return price, information_dump


