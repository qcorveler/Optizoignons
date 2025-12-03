from typing import Any, Optional, Tuple, Union
import numpy as np # type: ignore

def p(
    current_selling_season: int,
    selling_period_in_current_season: int,
    prices_historical_in_current_season: Union[np.ndarray, None],
    demand_historical_in_current_season: Union[np.ndarray, None],
    competitor_has_capacity_current_period_in_current_season: bool,
    information_dump=Optional[Any],
) -> Tuple[float, Any]:
    """Determine which price to set for the next period.

    Parameters
    ----------
    current_selling_season : int
        The current selling season (1, 2, 3, ...).
    selling_period_in_current_season : int
        The period in the current season (1, 2, ..., 1000).
    prices_historical_in_current_season : Union[np.ndarray, None]
        A two-dimensional array of historical prices. The rows index the competitors and the columns
        index the historical selling periods. Equal to `None` if`selling_period_in_current_season == 1`.
    demand_historical_in_current_season : Union[np.ndarray, None]
        A one-dimensional array of historical (own) demand. Equal to `None` if `selling_period_in_current_season == 1`.
    competitor_has_capacity_current_period_in_current_season : bool
        `False` if competitor is out of stock
    information_dump : Any, optional
        To keep a state (e.g., a trained model), by default None

    Examples
    --------

    >>> prices_historical_in_current_season.shape == (2, selling_period_in_current_season - 1)
    True

    >>> demand_historical_in_current_season.shape == (selling_period_in_current_season - 1, )
    True

    Returns
    -------
    Tuple[float, Any]
        The price and a the information dump (with, e.g., a state of the model).
    """
    # first set some shorter names for convenience
    day = selling_period_in_current_season
    season = current_selling_season
    demand = demand_historical_in_current_season
    prices = prices_historical_in_current_season
    print(' \t Algo: ---')
    print(" \t Algo: we are in selling_season %d and selling_period %d" % (season,day))
    # set some indicator variables
    first_day = True if day == 1 else False

    if first_day:
        # Randomize in the first period of the season
        price = np.random.uniform(30,70)
        print(" \t Algo: setting random price at init: %.3f" % price)
    elif selling_period_in_current_season == 101:
        # No price in selling period 101 (only used for getting information from previous period)
        price = None
    else:
        # change price with 33% chance or else keep old
        random_draw_4_price_change = np.random.rand()
        print(" \t Algo: random_draw_4_price_change: %.3f  => setting new price: %s" % 
                               (random_draw_4_price_change, random_draw_4_price_change < 0.33) )
        if random_draw_4_price_change < 0.33 :
            price = np.random.uniform(30,70)
            print(" \t Algo: new price is: %.2f" % price)
        else:
            price = prices[0, -1]
            print(" \t Algo: keeping old price")

    # our information dump is just the cumulative revenue within the given selling_season
    if first_day :
        information_dump =  {'revenue' : 0 }
    else:
        revenue_last_period = demand[-1]*prices[0, -1]
        information_dump['revenue'] = information_dump.get('revenue',0) + revenue_last_period
    
    print(" \t Algo: so far we have generated revenue: %.2f" % information_dump['revenue'])
    print(" \t Algo: my next price is: %.2f" % price)
    print(' \t Algo: ---')
    
    return price, information_dump




