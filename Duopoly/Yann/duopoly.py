import numpy as np
import pandas as pd
import uuid
import pickle
from typing import Any, Optional, Tuple, Union

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


# --- Configuration ---
# EXACT list of columns used during training
COL_USABLE = [
    'competitor_has_capacity',
    'selling_period', 
    'price', 
    'rolling_avg_price_competitor_season', 
    'rolling_sum_demand_season',
    'lag_price_competitor_1',
    'price_competitor',        # Estimated price for today
    'price_ratio_competitor',
    'cumulative_revenue_season'
]

# --- Load Models ---
try:
    # For dynamic competition platform
    with open('/root/partcipant_folder/xgb_demand_model_all_period.pkl', 'rb') as f:
        model_xgboost = pickle.load(f)
    

    # For local testing
    # with open('xgb_demand_model.pkl', 'rb') as f:
    #     model_xgboost = pickle.load(f)
except:
    print("Warning: Model not found.")
    model_xgboost = None

def _initialize_data_feedback():
    """Initializes the persistent state dictionary."""
    return {
        'history_list': [], # Optimized: List of dicts instead of DataFrame
        'current_simulation': str(uuid.uuid4()),
        'cumulative_revenue_current_selling_season': 0.0,
        'season_scores': {},
        'last_season_processed': 0,
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

def est_demand(feature_dict, model):
    """Predicts demand using the exact columns from training."""
    df_features = pd.DataFrame([feature_dict])[COL_USABLE]
    pred = model.predict(df_features)[0]    
    return max(0.0, float(pred))

def get_best_price_target_demand(target_demand, static_features, estimated_competitor_price, model):
    """Finds the price that best meets the target demand."""
    best_price = 45.0
    min_diff = float('inf')
    
    # Grid search
    prices_to_test = np.arange(80.33, 6.2, -0.61) 
    
    for price in prices_to_test:
        # 1. Dynamic features (depend on OUR price)
        # We use the estimated competitor price (e.g., yesterday's price)
        price_ratio = price / estimated_competitor_price if estimated_competitor_price > 0 else 1.0
        
        # 2. Construct full feature set
        features = static_features.copy()
        features['price'] = price
        features['price_competitor'] = estimated_competitor_price # Feature required by your model
        features['price_ratio_competitor'] = price_ratio
        
        # 3. Predict
        estimated_d = est_demand(features, model)
        
        # 4. Strategy: Reach target demand
        if estimated_d >= target_demand:
            return float(price)
            
        diff = abs(estimated_d - target_demand)
        if diff < min_diff:
            min_diff = diff
            best_price = price
            
    return float(best_price)

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

def p(
    current_selling_season: int,
    selling_period_in_current_season: int,
    prices_historical_in_current_season: Union[np.ndarray, None],
    demand_historical_in_current_season: Union[np.ndarray, None],
    competitor_has_capacity_current_period_in_current_season: bool,
    information_dump: Optional[Any] = None,
) -> Tuple[float, Any]:
    
    # Context
    day = selling_period_in_current_season
    season = current_selling_season
    
    # --- 1. Initialization ---
    if information_dump is None:
        information_dump = _initialize_data_feedback()
    FEEDBACK_OBJECT = {}

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
    if current_selling_season == 25 and day == 100 :
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
    
    # --- 2. Season Reset Logic (Day 1) ---
    if day == 1:
        # Check if we need to archive the previous season
        last_processed = information_dump.get('last_season_processed', 0)
        
        if season > 1 and last_processed < season:
            prev_rev = information_dump['cumulative_revenue_current_selling_season']
            information_dump['season_scores'][season - 1] = prev_rev
            
        # Reset for new season
        information_dump['cumulative_revenue_current_selling_season'] = 0.0
        information_dump['history_list'] = [] 
        information_dump['last_season_processed'] = season
        
        # Default price for Day 1 (Model needs history)
        return 45.0, information_dump

    # --- 3. Update Revenue from YESTERDAY ---
    # Must be done BEFORE calculating today's cumulative revenue feature
    if prices_historical_in_current_season is not None:
        last_demand = demand_historical_in_current_season[-1]
        last_own_price = prices_historical_in_current_season[0, -1]
        
        daily_revenue = last_demand * last_own_price
        information_dump['cumulative_revenue_current_selling_season'] += float(daily_revenue)

    # --- 4. Calculate Features ---
    
    # Extract History
    hist_comp_prices = prices_historical_in_current_season[1, :]
    hist_demand = demand_historical_in_current_season
    
    # Calculate Variables
    # Note: Ensure types match what the model expects (float/int)
    lag_price_competitor_1 = float(hist_comp_prices[-1])
    rolling_sum_demand_season = float(np.sum(hist_demand))
    rolling_avg_price_competitor_season = float(np.mean(hist_comp_prices))
    cumulative_revenue_season = float(information_dump['cumulative_revenue_current_selling_season'])
    
    # Convert bool to int for the model (0 or 1)
    comp_has_cap_int = 1 if competitor_has_capacity_current_period_in_current_season else 0
    
    # Prepare Static Features (exclude price, price_ratio, and price_competitor for now)
    static_features = {
        'competitor_has_capacity': comp_has_cap_int,
        'selling_period': day,
        'rolling_avg_price_competitor_season': rolling_avg_price_competitor_season,
        'rolling_sum_demand_season': rolling_sum_demand_season,
        'lag_price_competitor_1': lag_price_competitor_1,
        'cumulative_revenue_season': cumulative_revenue_season
    }
    
    # --- 5. Determine Target Demand ---
    remaining_stock = 80 - rolling_sum_demand_season
    remaining_days = 100 - day + 1
    target_daily_demand = remaining_stock / remaining_days if remaining_days > 0 else 0
    
    # --- 6. Optimization ---
    # Naive Forecast: We assume competitor price today == yesterday's price
    est_comp_price_today = competitor_price_prediction
    
    final_price = get_best_price_target_demand(
        target_demand=target_daily_demand,
        static_features=static_features,
        estimated_competitor_price=est_comp_price_today,
        model=model_xgboost
    )

    # --- 7. Save & Return ---
    # Optional: Log history to list (fast)
    if day == 100 and season == 25:
        information_dump['history_list'].append({
            'competition' : information_dump['current_simulation'],
            'day': day,
            'price': final_price,
            'revenue': cumulative_revenue_season,
            'use_model': use_model,
            'metrics': {name: metric.get() for name, metric in metrics_list.items()},
        })
    
    else :
        information_dump['history_list'].append({
            'competition' : information_dump['current_simulation'],
            'day': day,
            'price': final_price,
            'revenue': cumulative_revenue_season
        })

    if day >= 100:
        with open('duopoly_feedback.data', 'wb') as handle:
            pickle.dump(information_dump, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if not competitor_has_capacity_current_period_in_current_season:
        final_price = final_price * 1.2
    return float(final_price), information_dump