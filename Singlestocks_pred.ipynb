import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from pydtmc import MarkovChain
#import scikit-learn as sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error


pdf = PdfPages("Allsamplestocks.pdf")

# Define the stock and time period
ticker = ["ASIANPAINT.NS","HDFCBANK.NS","HDFCLIFE.NS","ICICIBANK.NS","JIOFIN.NS","MRF.NS","RCOM.NS","RELIANCE.NS","SAREGAMA.NS","SBIN.NS",
          "TATAMOTORS.NS","ANGELONE.NS","ATGL.NS","GREAVESCOT.NS","KIOCL.NS","RAJRATAN.NS","RICOAUTO.NS","TATASTEEL.NS","THANGAMAYL.NS",
         "CANBK.NS","CDSL.NS","DLF.NS","HAL.NS","IDFCFIRSTB.NS","INDHOTEL.NS","ITC.NS","KPITTECH.NS","TATAPOWER.NS","VBL.NS","VEDL.NS",
         "ADANIPOWER.NS","BLS.NS","CROMPTON.NS","GEPIL.NS","GPIL.NS","MRPL.NS","NETWORK18.NS","OLECTRA.NS","SUZLON.NS","ADANIENSOL.NS",
          "ADANIGREEN.NS","DABUR.NS","DCBBANK.NS","DMART.NS","DRREDDY.NS","IOC.NS","KOTAKBANK.NS","NTPC.NS","POWERGRID.NS","SUNPHARMA.NS"
         ]  
# Replace with your preferred stock symbol
start_date = "2015-01-01"
end_date = "2025-06-10"

data = pd.DataFrame()

data_combined = []

# Download stock data
for i in range(len(ticker)):
    data= yf.download(ticker[i], start=start_date, end=end_date)
    data_combined.append(data)
data_combined

# Simulate future states
def simulate_markov_chain(start_state, n_steps):
    future_states = [start_state]
    for _ in range(n_steps):
        next_state = np.random.default_rng().choice(
            transition_matrix.columns,
            p=transition_matrix.loc[future_states[-1]].values, replace = True
        )
        future_states.append(next_state)
    return future_states


initial_price = [None]*len(ticker)
final_price = [None]*len(ticker)
returns = [None]*len(ticker)
annualized_return = [None]*len(ticker)
dailyreturns_matrix = []

for i in range(len(ticker)):
    initial_price[i] = data_combined[i]['Close'][ticker[i]].iloc[0]
    final_price[i] = data_combined[i]['Close'][ticker[i]].iloc[-1]
    returns[i] = ((final_price[i] - initial_price[i]) / initial_price[i]) * 100
    annualized_return[i] = ((final_price[i]/initial_price[i])**(252/len(data_combined[i]['Close'][ticker[i]]))-1)*100
    # Calculate daily returns
    daily_returns = data_combined[i]['Close'].pct_change().dropna()

    #prices = data['Close']
    prices = data_combined[i]['Close'][ticker[i]]

    # Discretize price changes into states (e.g., Up, Down, No Change)
    returns = prices.pct_change().dropna()
    states = returns.apply(lambda r: 'up' if r > 0.0001 else ('down' if r < -0.0001 else 'steady'))

    #print("states")
    #print(states)

    states_for_pydtmc = ['up','down','steady']
    # Create transition matrix
    transition_matrix = pd.crosstab(states.shift(), states, normalize='index')
    
    #print("Transition Matrix:")
    #print(ticker[i])
    #print(transition_matrix)

     # Example simulation
    predicted_path = simulate_markov_chain(start_state=states.iloc[-1], n_steps=252)
    #print("\nPredicted State Path for Next 252 Days:")
    #print(ticker[i])
    #print(predicted_path)

    # Step 5: Backtest - compare simulated states to actual states
    backtest_start = -350 # last 350 days
    actual_states = states.iloc[backtest_start:]
    simulated_states = simulate_markov_chain(states.iloc[backtest_start - 1], len(actual_states))
    
    # Step 6: Evaluate accuracy
    accuracy = np.mean([a == s for a, s in zip(actual_states, simulated_states[1:])])
    print(ticker[i])
    #print("actual states")
    #print(actual_states)
    #print("simulated states")
    #print(simulated_states)
    print(f"Backtest Accuracy over {len(actual_states)} days: {accuracy:.2%}")

    # Define transition matrix and states
    p = transition_matrix
    #states = ['Down', 'Up']

    # Create the Markov chain
    mc = MarkovChain(p, states_for_pydtmc)

    # Print basic info
    #print("pydtmc markov chains")
    #print(mc)

    # Get steady-state probabilities
    print("Steady states from pydtmc:", mc.steady_states)

    # Simulate a sequence
    sequence = mc.simulate(steps=100,seed = 50)
    print("Simulated sequence from pydtmc:", sequence)

    encoder = LabelEncoder()
    actual_encoded = encoder.fit_transform(actual_states)
    predicted_encoded = encoder.transform(simulated_states[1:])
    #print("Actual encoded")
    #print(len(actual_encoded),actual_encoded)
    #print("predicted encoded")
    #print(len(predicted_encoded),predicted_encoded)
    rmse = np.sqrt(mean_squared_error(actual_encoded, predicted_encoded))
    mae = mean_absolute_error(actual_encoded, predicted_encoded)
    mape = mape = mean_absolute_percentage_error(actual_encoded, predicted_encoded)  # Avoid division by zero
    sd = np.std(actual_encoded - predicted_encoded)

    print(f"RMSE: {rmse}, MAE: {mae},, SD: {sd}")
    print(f"RMSE/SD: {rmse/sd}, MAE: {mae}, MAPE: {mape}")

    
    plt.figure(figsize=(10, 6))
    plt.plot(data_combined[i]['Close'][ticker[i]], label=f"{ticker[i]} Price")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid()
    #plt.show()
    plt.title(f"Buy-and-Hold Return for {ticker[i]} from {start_date} to {end_date} is: {returns.iloc[i]:.2f}%"              
    f"Annualised Buy-and-Hold Return for {ticker[i]} from {start_date} to {end_date} is:{annualized_return[i]:.2f}%", wrap = True)
    pdf.savefig()  # Save the figure to the PDF
    plt.close()
    dailyreturns_matrix.append(daily_returns)
pdf.close()

#print("daily returns matrix")
type(dailyreturns_matrix)

# Extract the first column from each DataFrame and combine into a new DataFrame

positive_dfs = [df.map(lambda x: x if x > 0 else None) for df in dailyreturns_matrix]
negative_dfs = [df.map(lambda x: x if x < 0 else None) for df in dailyreturns_matrix]

combined_down = pd.concat(list(map(lambda df: df.iloc[:, 0], negative_dfs)), axis=1)
combined_up = pd.concat(list(map(lambda df: df.iloc[:, 0], positive_dfs)), axis=1)

# Optionally name the columns
combined_down.columns = ticker
combined_up.columns = ticker

# Now calculate the correlation matrix
correlation_matrix_down = combined_down.corr()
correlation_matrix_up = combined_up.corr()

print("corr mat down")
print(correlation_matrix_down)

print("corr mat up")
print(correlation_matrix_up)

cutoff = 0.3

# Let's say 'corr_matrix' is your correlation matrix
# Step 1: Unstack to long format
corr_pairs_up = correlation_matrix_up.unstack()
corr_pairs_down = correlation_matrix_down.unstack()

# Step 2: Drop self-correlations (i.e., correlation of a variable with itself)
corr_pairs_up = corr_pairs_up[corr_pairs_up.index.get_level_values(0) != corr_pairs_up.index.get_level_values(1)]
corr_pairs_down = corr_pairs_down[corr_pairs_down.index.get_level_values(0) != corr_pairs_down.index.get_level_values(1)]

# Step 3: Drop duplicate pairs (since corr(A,B) == corr(B,A))
corr_pairs_up = corr_pairs_up.groupby(lambda x: frozenset(x)).mean()
corr_pairs_down = corr_pairs_down.groupby(lambda x: frozenset(x)).mean()

# Step 4: Sort and pick top 5
top_5_up = corr_pairs_up.sort_values(ascending=False).head(5)
top_5_down = corr_pairs_down.sort_values(ascending=False).head(5)

print("top 5 up move correlations")
print(top_5_up)
print("top 5 down move correlations")
print(top_5_down)

large_up = corr_pairs_up.sort_values(ascending=False)[corr_pairs_up > cutoff]
large_down = corr_pairs_down.sort_values(ascending=False)[corr_pairs_down > cutoff]

print("large up move correlations")
print(large_up)
print("large down move correlations")
print(large_down)

negative_up = corr_pairs_up.sort_values(ascending=False)[corr_pairs_up < -0.01]
negative_down = corr_pairs_down.sort_values(ascending=False)[corr_pairs_down < -0.01]

print("negative up move correlations")
print(negative_up)
print("negative down move correlations")
print(negative_down)


