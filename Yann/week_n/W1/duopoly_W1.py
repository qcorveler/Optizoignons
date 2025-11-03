from typing import Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import uuid
import pickle

def p(
    current_selling_season: int,
    selling_period_in_current_season: int,
    prices_historical_in_current_season: Union[np.ndarray, None],
    demand_historical_in_current_season: Union[np.ndarray, None],
    competitor_has_capacity_current_period_in_current_season: bool,
    information_dump=Optional[Any],
) -> Tuple[float, Any]:
        # first set some shorter names for convenience
    day = selling_period_in_current_season
    season = current_selling_season
    demand = demand_historical_in_current_season
    prices = prices_historical_in_current_season
        
        # set some indicator variables
    first_day = True if day == 1 else False
    if day < 90 and competitor_has_capacity_current_period_in_current_season == True:
        price = 150
    else :
        price = 80
    return price, information_dump

