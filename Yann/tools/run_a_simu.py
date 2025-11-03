import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None 
import pickle

import os
from importlib import reload

import matplotlib.pyplot as plt
import matplotlib 
plt.style.use('ggplot')
from matplotlib.ticker import MultipleLocator


def run_a_simu(csv_of_competition,
        s ,
        max_t = 100,
        is_courbe: str = None
        ):
    import duopoly as duopoly  
    reload(duopoly)
    df_comp_details = pd.read_csv('duopoly_competition_details.csv')
    df_comp_details['unique_selling_season_key'] = df_comp_details.apply(lambda r:
                                "%s_%s" % (r.competition_id,r.selling_season), axis=1)
    if is_courbe is not None:
        with open('target_sales_curve_quentin.pkl', 'rb') as f:
            courbe = pickle.load(f)
    if os.path.isfile('duopoly_feedback.data'):
        os.remove('duopoly_feedback.data')
        print('duopoly_feedback.data deleted')
    else:
        print('duopoly_feedback.data does not exists')
    factor = []
    s = 1 # first selling season
    max_t = 20 # selling period until we want to run the simulation
    information_dump = None
    my_prices = []

    # select form the competition details CSV which selling_season 
    # you want to use for the simulation
    id = df_comp_details['unique_selling_season_key'].unique()[0]
    df_select = df_comp_details[df_comp_details.unique_selling_season_key==id]

    # loop over the selling periods t
    for t in np.arange(1,(max_t+1),1):
        
        print("we are in selling_period = %d" % t)

        if t <=1 :
            prices_historical = None
            demand_historical = None
        else:
            # get price info from details CSV
            prices_historical = [ my_prices, 
                                df_select[df_select.selling_period < t]['price_competitor'] ]
            prices_historical  = np.array(prices_historical)
            # get demand info from details CSV
            demand_historical  = np.array(df_select[df_select.selling_period < t]['demand'])

        # get current info if competitor can sell
        competitor_has_capacity = df_select[df_select.selling_period == t]['competitor_has_capacity'].values[0]
        
        # build input object for price algo
        request_input = {
            "current_selling_season" : s, 
            "selling_period_in_current_season" : t,
            "prices_historical_in_current_season" : prices_historical, 
            "demand_historical_in_current_season" : demand_historical, 
            "competitor_has_capacity_current_period_in_current_season" : competitor_has_capacity, 
            "information_dump": information_dump
        }
        # request price from you algo and receive (price, information_dump)
        price , information_dump = duopoly.p(**request_input)
        factor.append(information_dump['factor'])
        
        print("###")
        print("We received from the algo the following results:")
        print("price: %.2f" % price)
        print("information_dump: %s" % str(information_dump) )
        # add you price to the own prices arroy
        my_prices.append(price)

    return information_dump, factor, df_select