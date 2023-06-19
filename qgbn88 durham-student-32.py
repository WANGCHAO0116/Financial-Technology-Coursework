import datetime as dt
import time
import logging
import pandas as pd
import numpy as np

from cointegration_analysis import estimate_long_run_short_run_relationships, engle_granger_two_step_cointegration_test

from optibook.synchronous_client import Exchange

from math import floor, ceil
from black_scholes import call_value, put_value, call_delta, put_delta
from libs import calculate_current_time_to_date

exchange = Exchange()
exchange.connect()

logging.getLogger('client').setLevel('ERROR')

def trade_would_breach_position_limit(instrument_id, volume, side, position_limit=300):
    positions = exchange.get_positions()
    position_instrument = positions[instrument_id]

    if side == 'bid':
        return position_instrument + volume > position_limit
    elif side == 'ask':
        return position_instrument - volume < -position_limit
    else:
        raise Exception(f'''Invalid side provided: {side}, expecting 'bid' or 'ask'.''')


def print_positions_and_pnl():
    positions = exchange.get_positions()
    pnl = exchange.get_pnl()

    print('Positions:')
    for instrument_id in positions:
        print(f'  {instrument_id:10s}: {positions[instrument_id]:4.0f}')

    print(f'\nPnL: {pnl:.2f}')


STOCK_IDS = ['ING', 'BAYER', 'SANTANDER']
volatility = {'ING': 4.0, 'BAYER': 4.0, 'SANTANDER': 3.2}

OPTIONS = [
    {'id': 'BAY-2022_03_18-050C', 'expiry_date': dt.datetime(2022, 3, 18, 12, 0, 0), 'strike': 50, 'callput': 'call'},
    {'id': 'BAY-2022_03_18-050P', 'expiry_date': dt.datetime(2022, 3, 18, 12, 0, 0), 'strike': 50, 'callput': 'put'},
    {'id': 'BAY-2022_03_18-075C', 'expiry_date': dt.datetime(2022, 3, 18, 12, 0, 0), 'strike': 75, 'callput': 'call'},
    {'id': 'BAY-2022_03_18-075P', 'expiry_date': dt.datetime(2022, 3, 18, 12, 0, 0), 'strike': 75, 'callput': 'put'},
    {'id': 'BAY-2022_03_18-100C', 'expiry_date': dt.datetime(2022, 3, 18, 12, 0, 0), 'strike': 100, 'callput': 'call'},
    {'id': 'BAY-2022_03_18-100P', 'expiry_date': dt.datetime(2022, 3, 18, 12, 0, 0), 'strike': 100, 'callput': 'put'},
    {'id': 'SAN-2022_03_18-040C', 'expiry_date': dt.datetime(2022, 3, 18, 12, 0, 0), 'strike': 40, 'callput': 'call'},
    {'id': 'SAN-2022_03_18-040P', 'expiry_date': dt.datetime(2022, 3, 18, 12, 0, 0), 'strike': 40, 'callput': 'put'},
    {'id': 'SAN-2022_03_18-050C', 'expiry_date': dt.datetime(2022, 3, 18, 12, 0, 0), 'strike': 50, 'callput': 'call'},
    {'id': 'SAN-2022_03_18-050P', 'expiry_date': dt.datetime(2022, 3, 18, 12, 0, 0), 'strike': 50, 'callput': 'put'},
    {'id': 'SAN-2022_03_18-060C', 'expiry_date': dt.datetime(2022, 3, 18, 12, 0, 0), 'strike': 60, 'callput': 'call'},
    {'id': 'SAN-2022_03_18-060P', 'expiry_date': dt.datetime(2022, 3, 18, 12, 0, 0), 'strike': 60, 'callput': 'put'},
    {'id': 'ING-2022_03_18-015C', 'expiry_date': dt.datetime(2022, 3, 18, 12, 0, 0), 'strike': 15, 'callput': 'call'},
    {'id': 'ING-2022_03_18-015P', 'expiry_date': dt.datetime(2022, 3, 18, 12, 0, 0), 'strike': 15, 'callput': 'put'},
    {'id': 'ING-2022_03_18-020C', 'expiry_date': dt.datetime(2022, 3, 18, 12, 0, 0), 'strike': 20, 'callput': 'call'},
    {'id': 'ING-2022_03_18-020P', 'expiry_date': dt.datetime(2022, 3, 18, 12, 0, 0), 'strike': 20, 'callput': 'put'},
    {'id': 'ING-2022_03_18-025C', 'expiry_date': dt.datetime(2022, 3, 18, 12, 0, 0), 'strike': 25, 'callput': 'call'},
    {'id': 'ING-2022_03_18-025P', 'expiry_date': dt.datetime(2022, 3, 18, 12, 0, 0), 'strike': 25, 'callput': 'put'},
]

underlying = {
    'ING-2022_03_18-015C':'ING',
    'ING-2022_03_18-015P':'ING',
    'ING-2022_03_18-020C':'ING',
    'ING-2022_03_18-020P':'ING',
    'ING-2022_03_18-025C':'ING',
    'ING-2022_03_18-025P':'ING',
    'BAY-2022_03_18-050C':'BAYER',
    'BAY-2022_03_18-050P':'BAYER',
    'BAY-2022_03_18-075C':'BAYER',
    'BAY-2022_03_18-075P':'BAYER',
    'BAY-2022_03_18-100C':'BAYER',
    'BAY-2022_03_18-100P':'BAYER',
    'SAN-2022_03_18-040C':'SANTANDER',
    'SAN-2022_03_18-040P':'SANTANDER',
    'SAN-2022_03_18-050C':'SANTANDER',
    'SAN-2022_03_18-050P':'SANTANDER',
    'SAN-2022_03_18-060C':'SANTANDER',
    'SAN-2022_03_18-060P':'SANTANDER',
}

cointegration_positions = {'BAYER': 0, 'ING': 0, 'SANTANDER': 0}

while True:
    print(f'')
    print(f'-----------------------------------------------------------------')
    print(f'TRADE LOOP ITERATION ENTERED AT {str(dt.datetime.now()):18s} UTC.')
    print(f'-----------------------------------------------------------------')
    
    #########################################
    #  Implement your main trade loop here  #
    #########################################
    
    print_positions_and_pnl()
    print(f'')
    
    # Determine stock value
    stocks_price = {}
    stocks_best_bid_price = {}
    stocks_best_ask_price = {}
    for stock_id in STOCK_IDS:
        stock_order_book = exchange.get_last_price_book(stock_id)
        if not (stock_order_book and stock_order_book.bids and stock_order_book.asks):
            print(f'Order book for {stock_id} does not have bids or offers. Skipping iteration.')
            continue
        best_bid_price = stock_order_book.bids[0].price
        best_ask_price = stock_order_book.asks[0].price
        
        stocks_price[stock_id] = (best_bid_price + best_ask_price) / 2
        stocks_best_bid_price[stock_id] = best_bid_price
        stocks_best_ask_price[stock_id] = best_ask_price
        print(f'Top level prices for {stock_id}: {best_bid_price:.2f} :: {best_ask_price:.2f}, The stock price for {stock_id}:{stocks_price[stock_id]:.2f}')
    if len(stocks_price) != len(STOCK_IDS):
        continue

    options_price = {}
    options_best_bid_price = {}
    options_best_ask_price = {}
    for option in OPTIONS:
        option_id = option['id']
        option_order_book = exchange.get_last_price_book(option_id)
        if not (option_order_book and option_order_book.bids and option_order_book.asks):
            print(f'Order book for {option_id} does not have bids or offers. Skipping iteration.')
            continue
        best_bid_price = option_order_book.bids[0].price
        best_ask_price = option_order_book.asks[0].price
    
        options_price[option_id] = (best_bid_price + best_ask_price) / 2
        options_best_bid_price[option_id] = best_bid_price
        options_best_ask_price[option_id] = best_ask_price
        print(f'Top level prices for {option_id}: {best_bid_price:.2f} :: {best_ask_price:.2f}, The option price for {option_id}:{options_price[option_id]:.2f}')
    if len(options_price) != len(OPTIONS):
        continue
        
    # Option-quoting strategy
    
    # For each option
    for option in OPTIONS:
        option_id = option['id']
        # Print which option we are updating
        print(f'''Updating option {option['id']} with expiry date {option['expiry_date']}, strike {option['strike']} '''
              f'''and type {option['callput']}.''')

        # Delete existing orders
        outstanding = exchange.get_outstanding_orders(option_id)
        for o in outstanding.values():
            result = exchange.delete_order(option_id, order_id=o.order_id)
            print(f"Deleted order id {o.order_id}: {result}")
            
        # Calculate option value
        if option['callput'] == 'call':
            option_value = call_value(S = stocks_price[underlying[option_id]], K = option['strike'], T = calculate_current_time_to_date(dt.datetime(2022, 3, 18, 12, 0, 0)), r = 0, sigma = volatility[underlying[option_id]])
        else:
            option_value = put_value(S = stocks_price[underlying[option_id]], K = option['strike'], T = calculate_current_time_to_date(dt.datetime(2022, 3, 18, 12, 0, 0)), r = 0, sigma = volatility[underlying[option_id]])
        print(f'The option value of {option_id} is {option_value}')
            
        # Calculate desired bid and ask prices
        desired_bid = floor(option_value / 0.1) * 0.1 - 0.1
        desired_ask = ceil(option_value / 0.1) * 0.1 + 0.1
        
        
        # Insert limit orders on those prices for a desired volume
        desired_volume = 30
        if not trade_would_breach_position_limit(instrument_id = option_id, volume = desired_volume, side = 'bid', position_limit = 150):
                print(f'''Inserting bid for {option_id}: {desired_volume:.0f} lot(s) at price {desired_bid:.2f}.''')
                exchange.insert_order(
                    instrument_id=option_id,
                    price=desired_bid,
                    volume= desired_volume,
                    side='bid',
                    order_type='limit')
        else:
            print(f'''Not inserting {desired_volume:.0f} lot bid for {option_id} to avoid position-limit breach.''')
        if not trade_would_breach_position_limit(instrument_id = option_id, volume = desired_volume, side = 'ask', position_limit = 150):
                print(f'''Inserting ask for {option_id}: {desired_volume:.0f} lot(s) at price {desired_ask:.2f}.''')
                exchange.insert_order(
                    instrument_id=option_id,
                    price=desired_ask,
                    volume= desired_volume,
                    side='ask',
                    order_type='limit')
        else:
            print(f'''Not inserting {desired_volume:.0f} lot ask for {option_id} to avoid position-limit breach.''')
            
        # Wait 1/10th of a second to avoid breaching the exchange frequency limit
        time.sleep(0.10)

    # Calculate current delta position across all instruments
    current_delta = {}
    for option in OPTIONS:
        option_id = option['id']
        option_position = exchange.get_positions()[option_id]
        if option['callput'] == 'call':
            delta = call_delta(S = stocks_price[underlying[option_id]], K = option['strike'], T = calculate_current_time_to_date(dt.datetime(2022, 3, 18, 12, 0, 0)), r = 0, sigma = volatility[underlying[option_id]]) * option_position
        else:
            delta = put_delta(S = stocks_price[underlying[option_id]], K = option['strike'], T = calculate_current_time_to_date(dt.datetime(2022, 3, 18, 12, 0, 0)), r = 0, sigma = volatility[underlying[option_id]]) * option_position
        current_delta[option_id] = delta
        
    for stock_id in STOCK_IDS:
        stock_position = exchange.get_positions()[stock_id] - cointegration_positions[stock_id]
        delta = stock_position * 1
        current_delta[stock_id] = delta
            
        
    # Calculate stocks to buy/sell to become close to delta-neutral
    hedging_volume = {}
    for stock_id in STOCK_IDS:
        hedging_delta = current_delta[stock_id]
        for option in OPTIONS:
            option_id = option['id']
            if underlying[option_id] == stock_id:
                hedging_delta += current_delta[option_id]
        hedging_volume[stock_id] = int(hedging_delta)
    print(f'Stocks required to buy/sell to become close to delta-neutral: ')
    print(f'{hedging_volume}')
          
    # Perform the hedging stock trade by inserting an IOC order on the stock against the current top-of-book
    for stock_id in STOCK_IDS:
        volume = 15
        best_bid_price = stocks_best_ask_price[stock_id]
        best_ask_price = stocks_best_bid_price[stock_id]
        while abs(hedging_volume[stock_id]) > 10:
            if hedging_volume[stock_id] < 0:
                if not trade_would_breach_position_limit(instrument_id = stock_id, volume = volume, side = 'bid', position_limit = 300):
                    print(f'''Inserting bid for {stock_id}: {volume:.0f} lot(s) at price {best_bid_price:.2f}.''')
                    exchange.insert_order(
                        instrument_id=stock_id,
                        price=best_bid_price,
                        volume= volume,
                        side='bid',
                        order_type='ioc')
                    best_bid_price += 0.002
                    if abs(best_bid_price - stocks_best_ask_price[stock_id]) > 0.1:
                        break
                else:
                    print(f'''Not inserting {volume:.0f} lot ask for {stock_id} to avoid position-limit breach.''')
                    break
            else:
                if not trade_would_breach_position_limit(instrument_id = stock_id, volume = volume, side = 'ask', position_limit = 300):
                    print(f'''Inserting ask for {stock_id}: {volume:.0f} lot(s) at price {best_ask_price:.2f}.''')
                    exchange.insert_order(
                        instrument_id=stock_id,
                        price=best_ask_price,
                        volume= volume,
                        side='ask',
                        order_type='ioc')
                    best_ask_price -= 0.002
                    if abs(best_ask_price - stocks_best_bid_price[stock_id]) > 0.1:
                        break
                else:
                    print(f'''Not inserting {volume:.0f} lot ask for {stock_id} to avoid position-limit breach.''')
                    break
            hedging_delta = exchange.get_positions()[stock_id] - cointegration_positions[stock_id]
            for option in OPTIONS:
                option_id = option['id']
                if underlying[option_id] == stock_id:
                    hedging_delta += current_delta[option_id]
            hedging_volume[stock_id] = int(hedging_delta)
            time.sleep(0.10)
            
    option_quoter_positions = {}
    for stock_id in STOCK_IDS:
        option_quoter_positions[stock_id] = exchange.get_positions()[stock_id] - cointegration_positions[stock_id]
    
    
    
    
    # Cointegration strategy
    
    Y = stocks_price['BAYER']
    X = stocks_price['SANTANDER']
    y = np.log(stocks_price['BAYER'])
    x = np.log(stocks_price['SANTANDER'])
    z = y - (- 0.57 + 1.25 * x)
    
    # Insert IOC ask orders on stock BAYER if z > 0.001
    if z > 0.001:
        y_price = stocks_best_bid_price['BAYER']
        y_volume = 20
        if not trade_would_breach_position_limit(instrument_id = 'BAYER', volume = y_volume, side = 'ask', position_limit=50):
            print(f'''Inserting ask for BAYER: {y_volume:.0f} lot(s) at price {y_price:.2f}.''')
            exchange.insert_order(
                instrument_id='BAYER',
                price=y_price,
                volume=y_volume,
                side='ask',
                order_type='ioc')
        else:
            print(f'''Not inserting {y_volume:.0f} lot ask for BAYER to avoid position-limit breach.''')
        
            
    
    # Insert IOC bid orders on stock BAYER if z < -0.006
    elif z < -0.006:
        y_price = stocks_best_ask_price['BAYER']
        y_volume = 20
        if not trade_would_breach_position_limit(instrument_id = 'BAYER', volume = y_volume, side = 'bid', position_limit=50):
            print(f'''Inserting bid for BAYER: {y_volume:.0f} lot(s) at price {y_price:.2f}.''')
            exchange.insert_order(
                instrument_id='BAYER',
                price=y_price,
                volume=y_volume,
                side='bid',
                order_type='ioc')
        else:
            print(f'''Not inserting {y_volume:.0f} lot bid for BAYER to avoid position-limit breach.''')

    # Calculate the current positions of stocks BAYER & SANTANDERS in cointegration strategy
    
    y_position = exchange.get_positions()['BAYER'] - option_quoter_positions['BAYER']
    x_position = exchange.get_positions()['SANTANDER'] - option_quoter_positions['SANTANDER']
    x_ask_price = stocks_best_bid_price['SANTANDER']
    x_bid_price = stocks_best_ask_price['SANTANDER']
    x_volume = 15
    
    # Perform the hedging stock trade by inserting IOC orders on stock SANTANDER
    
    while abs(y_position + X * x_position / (1.25 * Y)) > 10:
        if y_position < 0 and x_position >=0:
            if abs(X * x_position / (1.25 * Y)) > abs(y_position):
                if not trade_would_breach_position_limit(instrument_id = 'SANTANDER', volume = x_volume, side = 'ask'):
                    print(f'''Inserting ask for SANTANDER: {x_volume:.0f} lot(s) at price {x_ask_price:.2f}.''')
                    exchange.insert_order(
                        instrument_id='SANTANDER',
                        price=x_ask_price,
                        volume=x_volume,
                        side='ask',
                        order_type='ioc')
                    x_ask_price -= 0.005
                    if abs(x_ask_price - stocks_best_bid_price['SANTANDER']) > 0.1:
                        break
                else:
                    print(f'''Not inserting {x_volume:.0f} lot ask for SANTANDER to avoid position-limit breach.''')
                    break
            else:
                if not trade_would_breach_position_limit(instrument_id = 'SANTANDER', volume = x_volume, side = 'bid'):
                    print(f'''Inserting bid for SANTANDER: {x_volume:.0f} lot(s) at price {x_bid_price:.2f}.''')
                    exchange.insert_order(
                        instrument_id='SANTANDER',
                        price=x_bid_price,
                        volume=x_volume,
                        side='bid',
                        order_type='ioc')
                    x_bid_price += 0.005
                    if abs(x_bid_price - stocks_best_ask_price['SANTANDER']) > 0.1:
                        break
                else:
                    print(f'''Not inserting {x_volume:.0f} lot bid for SANTANDER to avoid position-limit breach.''')
                    break
        elif y_position < 0 and x_position < 0:
            if not trade_would_breach_position_limit(instrument_id = 'SANTANDER', volume = x_volume, side = 'bid'):
                print(f'''Inserting bid for SANTANDER: {x_volume:.0f} lot(s) at price {x_bid_price:.2f}.''')
                exchange.insert_order(
                    instrument_id='SANTANDER',
                    price=x_bid_price,
                    volume=x_volume,
                    side='bid',
                    order_type='ioc')
                x_bid_price += 0.005
                if abs(x_bid_price - stocks_best_ask_price['SANTANDER']) > 0.1:
                        break
            else:
                print(f'''Not inserting {x_volume:.0f} lot bid for SANTANDER to avoid position-limit breach.''')
                break
            
        elif y_position >= 0 and x_position < 0:
            if abs(X * x_position / (1.25 * Y)) > abs(y_position):
                if not trade_would_breach_position_limit(instrument_id = 'SANTANDER', volume = x_volume, side = 'bid'):
                    print(f'''Inserting bid for SANTANDER: {x_volume:.0f} lot(s) at price {x_bid_price:.2f}.''')
                    exchange.insert_order(
                        instrument_id='SANTANDER',
                        price=x_bid_price,
                        volume=x_volume,
                        side='bid',
                        order_type='ioc')
                    x_bid_price += 0.005
                    if abs(x_bid_price - stocks_best_ask_price['SANTANDER']) > 0.1:
                        break
                else:
                    print(f'''Not inserting {x_volume:.0f} lot bid for SANTANDER to avoid position-limit breach.''')
                    break
            else:
                if not trade_would_breach_position_limit(instrument_id = 'SANTANDER', volume = x_volume, side = 'ask'):
                    print(f'''Inserting ask for SANTANDER: {x_volume:.0f} lot(s) at price {x_ask_price:.2f}.''')
                    exchange.insert_order(
                        instrument_id='SANTANDER',
                        price=x_ask_price,
                        volume=x_volume,
                        side='ask',
                        order_type='ioc')
                    x_ask_price -= 0.005
                    if abs(x_ask_price - stocks_best_bid_price['SANTANDER']) > 0.1:
                        break
                else:
                    print(f'''Not inserting {x_volume:.0f} lot ask for SANTANDER to avoid position-limit breach.''')
                    break
        elif y_position >= 0 and x_position >= 0:
            if not trade_would_breach_position_limit(instrument_id = 'SANTANDER', volume = x_volume, side = 'ask'):
                print(f'''Inserting ask for SANTANDER: {x_volume:.0f} lot(s) at price {x_ask_price:.2f}.''')
                exchange.insert_order(
                    instrument_id='SANTANDER',
                    price=x_ask_price,
                    volume=x_volume,
                    side='ask',
                    order_type='ioc')
                x_ask_price -= 0.005
                if abs(x_ask_price - stocks_best_bid_price['SANTANDER']) > 0.1:
                        break
            else:
                print(f'''Not inserting {x_volume:.0f} lot ask for SANTANDER to avoid position-limit breach.''')
                break
                    
        y_position = exchange.get_positions()['BAYER'] - option_quoter_positions['BAYER']
        x_position = exchange.get_positions()['SANTANDER'] - option_quoter_positions['SANTANDER']
        time.sleep(0.10)
        
    for stock_id in STOCK_IDS:
        if x_position * y_position > 0:
            cointegration_positions[stock_id] = 0
        else:
            cointegration_positions[stock_id] = exchange.get_positions()[stock_id] - option_quoter_positions[stock_id]
        
    print(f'option_quoter_positions: {option_quoter_positions}')
    print(f'cointegration_positions: {cointegration_positions}')

    # Sleep until next iteration
    print(f'\nSleeping for 3 seconds.')
    time.sleep(3)
