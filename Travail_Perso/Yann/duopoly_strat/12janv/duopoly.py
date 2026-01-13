import numpy as np
import pandas as pd
import uuid
import pickle
from typing import Any, Optional, Tuple, Union

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
    with open('/root/partcipant_folder/xgb_demand_model.pkl', 'rb') as f:
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
        'last_season_processed': 0
    }

def est_demand(feature_dict, model):
    """Predicts demand using the exact columns from training."""
    if model is None: return 0.0
    
    # Create DataFrame with the specific columns order
    # Any missing feature in feature_dict will error, which is good for debugging
    try:
        df_features = pd.DataFrame([feature_dict])[COL_USABLE]
        pred = model.predict(df_features)[0]
    except Exception as e:
        # print(f"Prediction error: {e}")
        pred = 0.0
        
    return max(0.0, float(pred))

def get_best_price_target_demand(target_demand, static_features, estimated_competitor_price, model):
    """Finds the price that best meets the target demand."""
    best_price = 45.0
    min_diff = float('inf')
    
    # Grid search
    prices_to_test = np.arange(3, 100, 0.1) 
    
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
        if estimated_d <= target_demand:
            return float(price)
            
        diff = abs(estimated_d - target_demand)
        if diff < min_diff:
            min_diff = diff
            best_price = price
            
    return float(best_price)

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
    est_comp_price_today = lag_price_competitor_1
    
    final_price = get_best_price_target_demand(
        target_demand=target_daily_demand,
        static_features=static_features,
        estimated_competitor_price=est_comp_price_today,
        model=model_xgboost
    )

    # --- 7. Save & Return ---
    # Optional: Log history to list (fast)
    information_dump['history_list'].append({
        'day': day,
        'price': final_price,
        'revenue': cumulative_revenue_season
    })
    
    return float(final_price), information_dump