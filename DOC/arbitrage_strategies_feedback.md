Strategy 1: Temporal Arbitrage - Contract vs. Spot Price Spread 

Market Mechanism 

The Jiangsu market uses a "Contract for Difference" settlement model, where:

Medium-to-long-term contracts are settled at the contract price.

Contract deviation electricity volume is settled at the spot market price.

There is a ±3% deviation tolerance before penalties are incurred.

Arbitrage Opportunity 

Price Spread Utilization: When the spot price significantly diverges from the contract price, profit can be generated through controlled deviations.

Implementation Strategy 


Over-consumption Arbitrage (Spot Price < Contract Price)---This is feasible, but this strategy needs to target specific customer types, consider deviation assessments, and have clear rules for monthly, day-ahead, and intraday trading to not exceed the deviation limits.


Generation Side: When the deviation between actual generation and the day-ahead awarded volume exceeds ±3%, the excess portion is settled at the real-time market price ±10% (a punitive tariff).


User Side (including electricity retailers): When the deviation between actual consumption and the day-ahead declared volume exceeds ±5%, the excess portion is settled at 1.5 times the real-time price.


New Energy Stations: A stricter ±2% deviation limit is enforced, with the excess portion assessed at 2 times the real-time price.


Scenario: The day-ahead spot price is ¥350/MWh, and your contract price is ¥450/MWh.


Action: Increase customer sales volume by 2.9% (staying within the tolerance limit).


Settlement:

Contracted volume: Pay ¥450/MWh.

Over-consumed volume: Pay ¥350/MWh (spot price).

Gain a profit of ¥100/MWh on the additional volume.

B. Under-consumption Arbitrage (Spot Price > Contract Price)




Scenario: The real-time price surges to ¥600/MWh, while the contract price is ¥400/MWh.


Action: Reduce sales volume by 2.9%, triggering a demand response.


Settlement:

Sell the contracted volume at a retail markup.

Avoid paying the ¥600/MWh spot price.

Avoided a cost of ¥200/MWh + demand response fees.

Strategy 2: Zonal Price Spread Arbitrage - Geographic Optimization 

The Jiangsu electricity spot market uses a "uniform clearing point price" model, not the Locational Marginal Pricing (LMP) model of the US PJM market. Therefore, there are no strict price zones within the province

. This model is not currently supported. The only possibilities are long-term PPAs or inter-provincial trading, both of which are rare and not a primary consideration for now.

Market Structure 

During periods of transmission congestion, the Jiangsu market is divided into different price zones:

Southern Jiangsu: Industrial area, higher demand, higher prices.

Northern Jiangsu: Rural area, lower demand, lower prices.

Provincial Average Price: A weighted average of the various zones.

Arbitrage Mechanism 

Geographic Price Spread Trading: Purchase electricity in a low-price zone to serve customers in a high-price zone.

Price Analysis 

Average Zonal Price Spread: ¥20-50/MWh between Southern and Northern Jiangsu.

Peak Spread Events: Up to ¥100-150/MWh during congestion.

Frequency: Meaningful price spreads appear on 40-60% of trading days.

Implementation Strategy 


Focus on Southern Jiangsu: 

Target: Manufacturing plants, commercial complexes.

Advantage: Can charge higher rates due to high local prices.

Volume: Accounts for 70% of the customer base.


Procurement in Northern Jiangsu: 

Strategy: Actively bid in the day-ahead market in the Northern Jiangsu region.

Advantage: Less competition, surplus generation.

Cost Savings: 15-25% lower than the provincial average.

Strategy 3: Renewable Energy Forecasting Arbitrage 

This is feasible, but we need to define our competitive advantage. How can our model's accuracy be better than others? The current understanding is that some market players use logic based on multiple models with high data refresh rates.

Data Category	Data Source Name
Forecast Data	[NOAA] GFS Forecast
[ECMWF] AIFS Large Model Forecast ᴬᴵ
[NOAA] GEFS P25 Ensemble Forecast
[NOAA] GEFS P50 Ensemble Forecast (Long-term)
[ECMWF] IFS P25 Forecast
[ECMWF] ENS Open Ensemble Forecast
[NOAA] Graphcast Large Model Forecast ᴬᴵ
Observation Data	[ECMWF] ERA5 Observation
[ECMWF] ERA5-land Land Observation
Commercial Data	[ECMWF] ENS-Extended 45D Commercial Forecast
[ECMWF] ENS Commercial Forecast
[ECMWF] HRES 10D Commercial Forecast

Export to Sheets
If we aggregate multiple models (public or periodically updated), is this advantage significant? Are there other superior strategies that can be iterated upon? 

Based on Forecast Discrepancies: There are significant differences between official renewable energy forecasts and actual output.

Wind Power Forecast Error: Average ±20-30%.

Solar (PV) Forecast Error: Average ±15-25%.

Price Impact: A 100MW forecast error can cause a price fluctuation of ¥50-100/MWh.

Arbitrage Logic 

Superior Forecasting Advantage: Build a better renewable energy forecasting model to anticipate price movements.

Trading Strategy 

A. Day-Ahead Market Positioning




Scenario 1: Your model predicts wind power will be 20% higher than the official forecast.


Action: Reduce day-ahead purchase volume (as prices are expected to drop in the real-time market).


Execution: Purchase only 95% of expected demand in the day-ahead market.


Settlement: Cover the remaining 5% at a lower real-time price.


Scenario 2: Your model predicts a shortage in solar power generation.


Action: Increase day-ahead purchase volume (as prices are expected to spike in the real-time market).


Execution: Purchase 105% of expected demand in the day-ahead market.


Settlement: Sell the excess 5% at a higher real-time price.

Supplementary Information on Trading Strategies: 

1. Sources of Core Profit: 

Price Arbitrage: 

Capture price differences between the day-ahead and real-time markets, primarily based on the accuracy of forecasting new energy output and load.

Policy Arbitrage: 

Different provinces have different trading rules. Some, like Gansu, allow multi-directional trading at a single time point, enabling quantitative trading based on price differentials.

Some provinces limit the scale of spot trading, allowing for arbitrage by over-purchasing or selling based on deviation rules.

Model Arbitrage: 

Using ARIMA-LSTM hybrid models to predict regional electricity price fluctuations?? I don't understand this, is it feasible? 

Trading Structure and Strategy Services: 

Forming alliances with thermal and new energy power generators to submit joint bids.

Combining Power Purchase Agreements (PPAs) with industrial loads to offer special packages that reduce costs (heard of this, but haven't seen case studies).

Public Sentiment Analysis: Building a sentiment model to update forecasts for abnormal fluctuations in generation and load, including factors like power plant maintenance, unusual weather, and public events. I've seen bits and pieces of this; can it be standardized into a model to add sentiment weight to the algorithm? 

Using financial quantitative trading models to optimize bidding?? Does this exist abroad? How is it used? There is very limited information on this in the domestic market.

Risk Management for Trading Models: 

To increase the win rate, execute trades only when multiple models suggest the same conclusion.