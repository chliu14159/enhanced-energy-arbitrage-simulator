
# Jiangsu Province Electricity Spot Market Operating Rules (V1.0)

This document provides a detailed summary of the "Jiangsu Province Electricity Spot Market Operating Rules (V1.0版)". It covers the market structure, participants, trading mechanisms, settlement procedures, and other related regulations for the Jiangsu electricity spot market.

---

## Chapter 1: General Provisions

The rules are established to standardize the operation and management of the Jiangsu electricity spot market, protect the rights of market members, ensure the market's safe and efficient operation, and promote the consumption of new energy. The establishment of these rules is based on national electricity system reform policies and relevant laws.

The core principles guiding the spot market are:
* **Market-Oriented**: Using market price signals to guide production and consumption for optimal resource allocation.
* **Open and Transparent**: Ensuring market rules, price mechanisms, and operational data are clear and accessible.
* **Active and Prudent**: Progressing steadily with market construction based on Jiangsu's specific conditions.
* **Adapted to Local Conditions**: Building a system that reflects Jiangsu's unique power production and consumption characteristics.
* **Safe and Controllable**: Prioritizing the safety, stability, and reliability of the power system.
* **Clean and Low-Carbon**: Creating mechanisms that favor the large-scale integration and consumption of new energy.

These rules apply to the operation, management, and settlement of the Jiangsu electricity spot energy market and its ancillary services market.

---

## Chapter 2: Terminology

Key terms used in the Jiangsu spot market include:

* **Electricity Wholesale Market**: A market where generation companies trade large volumes of electricity with "Type 1 users" (direct participants) and electricity sales companies.
* **Energy Market**: The component of the wholesale market where electrical energy is the traded commodity.
* **Spot Trading**: Centralized electricity trading activities for the next day (day-ahead) or within the operating day (real-time). The traded products include energy, frequency regulation, and reserve services.
* **Day-Ahead Market (DAM)**: A market where participants buy and sell energy and ancillary services for the next operating day at financially binding prices.
* **Real-Time Market (RTM)**: A market for trading energy and ancillary services for the upcoming 5 or 15-minute intervals of the operating day.
* **Operating Day (D Day)**: The calendar day when the results of the day-ahead market transactions are physically executed.
* **Zonal Price**: When transmission congestion occurs, the grid is divided into price zones based on the main congested sections.
* **Security-Constrained Unit Commitment (SCUC)**: An optimization process that determines the on/off schedule for generation units over multiple time periods, aiming to maximize social welfare or minimize supply costs while respecting system security constraints.
* **Security-Constrained Economic Dispatch (SCED)**: An optimization process that determines the power output levels for online generation units, aiming to minimize supply costs while respecting system security constraints.

---

## Chapter 3: Market Members

### Definition and Roles

Market members are categorized as follows:
* **Market Entities**: Include generation companies, Type 1 users (direct wholesale participants), electricity sales companies, and independent ancillary service providers.
* **Grid Enterprise**: Refers to the State Grid Jiangsu Electric Power Co., Ltd..
* **Market Operating Institutions**: Include the Jiangsu Power Dispatch Control Center (responsible for transaction organization and clearing) and the Jiangsu Electric Power Trading Center Co., Ltd. (responsible for entity registration and settlement basis).

### Rights and Obligations

All market members have specific rights and obligations. Common obligations include adhering to market rules, fulfilling contracts, providing and protecting information, and submitting to unified dispatch and scheduling.

* **Generation Companies**: Must participate in trading, obey dispatch instructions, and provide ancillary services.
* **Electricity Users (Type 1)**: Must participate in trading, pay fees on time, and comply with demand-side management and orderly consumption requirements during system emergencies.
* **Electricity Sales Companies**: Represent retail users (Type 2 users) in the market, manage contracts, and are responsible for information confirmation for their clients.
* **Grid Enterprise**: Guarantees grid safety, provides fair transmission services, provides various customer services (metering, billing), and acts as the purchasing agent for users not directly participating in the market.

### Market Registration and Exit

* **Registration**: Entities must be legally incorporated, financially independent, and have a good credit history to register. They must complete a specific spot market registration with the Power Trading Center.
* **Voluntary Exit**: A market entity (like a sales company or Type 1 user) wishing to exit must apply 30 working days in advance, clear all outstanding fees, and settle all unfulfilled transactions.
* **Forced Exit**: Entities that severely violate rules, abuse market power, or engage in fraudulent behavior can be forcibly removed from the market and may be placed on a "blacklist". A sales company forced to exit must transfer its contracts; if unsuccessful, the contracts are handled by the trading institution, with the exiting company bearing any losses.

---

## Chapter 4: Day-Ahead Energy Market (DAM)

The DAM is organized as a centralized, full-volume bidding market cleared through a unified optimization process.

### Organization and Timing

* **Schedule**: The market for the next operating day (D-day) is conducted on the day before (D-1 day).
* **Trading Intervals**: The day is divided into 96 trading intervals, each 15 minutes long.
* **Bidding**: On D-1 day, generation units, Type 1 users, and sales companies submit their price and quantity bids for D-day. Grid companies acting as agents for other users participate as "price takers".

### Bidding and Boundary Conditions

* **Generation Bids**: Generators submit multi-part bids including:
    * **Energy Costs**: A price curve with up to 10 segments (in ¥/MWh), which must be non-decreasing with output.
    * **Startup Costs**: Separate costs for cold, warm, and hot starts (in ¥/time).
    * **Shutdown Costs**: A cost for stopping the unit (in ¥/time).
* **User Bids**: Type 1 users and sales companies submit their demand curves (price and quantity) for each price zone.
* **Deadlines**: All market participants must complete their DAM bids by **10:30 AM on D-1 day**.
* **Boundary Conditions**: The market clearing process considers numerous constraints, which are published by 9:30 AM on D-1 day. These include:
    * System and nodal load forecasts.
    * Inter-provincial power exchange schedules.
    * New energy generation forecasts (which are given priority in dispatch).
    * Unit maintenance schedules and operational limits.
    * Transmission grid constraints and line limits.
    * "Must-run" and "must-stop" unit designations for grid security.

### Market Clearing and Results

* **Clearing Algorithm**: The market is cleared using a **Security-Constrained Unit Commitment (SCUC)** and **Security-Constrained Economic Dispatch (SCED)** model. The objective is to maximize social welfare (the difference between the value to consumers and the cost to producers).
* **Outputs**: The clearing process determines the binding results for D-day, including:
    * The on/off status of each generation unit for each interval.
    * The hourly power output curve for each online unit.
    * The hourly zonal clearing price for energy.
* **Timeline**:
    * **14:30 (D-1 Day)**: The DAM is cleared.
    * **17:30 (D-1 Day)**: Final, security-checked DAM results are published. This includes cleared prices, unit schedules, load forecasts, and grid constraint information.
* **Dispatch Plan**: The cleared generation schedules generally become the final dispatch plan for the next day. If a generator cannot meet its cleared schedule, it must notify the dispatch center within 30 minutes of the results being published and will bear the costs of procuring a replacement.

---

## Chapter 5: Real-Time Energy Market (RTM)

The RTM adjusts the day-ahead schedule to handle real-time deviations in load, generation, and grid conditions. It is also a centralized, full-volume bidding market.

### Organization and Timing

* **Clearing Cycle**: The RTM runs on the operating day (D-day) and clears every 5 or 15 minutes on a rolling basis.
* **Bidding**: Participants can submit updated bids for the RTM. If no new bid is submitted, the bid from the DAM is used by default.
* **Clearing Algorithm**: The RTM uses a **Security-Constrained Economic Dispatch (SCED)** algorithm to minimize the cost of electricity procurement based on the latest system conditions.
* **Boundary Conditions**: The RTM uses ultra-short-term forecasts for system load and new energy output, and accounts for any real-time changes in unit availability or grid constraints.

### Results and Dispatch

* **Results**: The RTM continuously produces dispatch instructions (output levels) for generators and real-time zonal energy prices.
* **Publication**: Real-time prices are published on a rolling basis for the next dispatch interval. A full summary of the previous day's (D-day's) real-time market results, including actual loads, generation outputs, and final prices, is published on D+1 day.
* **Emergency Adjustments**: In case of major grid events or emergencies, the dispatch center can override market results to ensure system security, with all such actions being logged and reported.

---

## Chapter 6: Risk Control

The rules establish a framework for managing market risks.

* **Risk Categories**: Risks are classified into:
    * **Price Risk**: Extreme or volatile market prices due to fuel price shocks or supply/demand imbalances.
    * **Technical System Risk**: Failures in the software or communication systems supporting the market.
    * **Cybersecurity Risk**: Malicious attacks on market systems.
    * **Performance Risk**: Failure of participants to meet their financial obligations (e.g., non-payment).
    * **Force Majeure Risk**: Unforeseeable major events that disrupt the market.
* **Risk Management Process**: This involves risk identification, analysis, early warning, and disposal.
* **Price Limits**: To mitigate price risk, the market sets upper and lower limits on both bid prices and clearing prices for energy. These limits are designed to balance market efficiency with system reliability and user protection.

---

## Chapter 7: Ancillary Services Market

The spot market framework includes markets for essential grid support services.

* **Market Varieties**: Includes markets for spinning reserve, frequency regulation, and black start services.
* **Frequency Regulation**: This market is organized via centralized competitive bidding. Participants (generators, energy storage, etc.) are selected based on a combination of their performance and their price bid. Compensation is divided into a capacity payment (for being available) and a mileage payment (for actual service provided).
* **Linkage with Energy Market**: Initially, the ancillary services markets run sequentially but in coordination with the energy market. The long-term goal is to achieve joint clearing of energy and ancillary services. The spot energy market itself replaces the previous peak-shaving ancillary service market.

---

## Chapter 8: Market Linkage Mechanisms

The spot market must seamlessly connect with other electricity markets.

* **Linkage with Medium and Long-Term (MLT) Markets**:
    * Participants are encouraged to sign a high proportion of their needs through MLT contracts to hedge against spot price volatility.
    * MLT contracts must specify a delivery price zone and a power curve (or a method to create one). These contracts act as a financial hedge; physical dispatch is determined by the spot market.
    * The difference between a participant's MLT contract position and their actual physical generation/consumption is settled at the spot market price.
* **Linkage with Inter-Provincial Markets**:
    * Power schedules determined by inter-provincial transactions (both MLT and spot) and regional ancillary service markets (e.g., East China Grid) serve as fixed boundary conditions for the Jiangsu provincial spot market.

---

## Chapter 9: Metering and Settlement

This chapter details the financial heart of the market.

### Metering

* **Requirements**: All spot market participants must have metering systems capable of recording and transmitting data for each 15-minute settlement interval.
* **Data Handling**: The grid company is responsible for installing and managing meters. If data is missing, it is estimated using approved data completion algorithms. Metering points are typically at the property rights boundary.

### Settlement

* **Settlement Cycle**: The market uses a two-part settlement system. While clearing happens every 15 minutes, formal settlement is conducted monthly.
* **Settlement Principle**: This is a "Contract for Differences" model.
    1.  Medium and long-term contracts are settled financially based on the contract price.
    2.  The deviation between the MLT contract volume and the **day-ahead cleared volume** is settled at the **day-ahead zonal price**.
    3.  The deviation between the day-ahead cleared volume and the **actual metered volume** is settled at the **real-time zonal price**.
* **Wholesale Charges**: A participant's final bill is composed of multiple components:
    * **Energy Charges**: From the MLT and spot market settlements.
    * **Capacity Compensation Charges**: A mechanism to compensate generators for their availability, helping to cover fixed costs. This is funded by users based on their spot market energy usage.
    * **Cost Compensation Charges**: Payments to generators for specific actions taken for grid security, such as starts/stops ordered by dispatch, must-run operation, or low-load operation. These costs are socialized among users.
    * **Ancillary Service Charges**: Costs from the ancillary service markets.
    * **Imbalance Funds**: A mechanism to reconcile funds, such as those arising from congestion between price zones. These are allocated across market participants.
* **Settlement Workflow**:
    * **D+2 Day**: The grid company provides final, verified meter data for D-day.
    * **D+3 Day**: The market operator calculates and publishes the preliminary daily spot settlement results.
    * **Monthly**: At the start of the next month, the market operator consolidates all daily results and MLT contract settlements to issue the final, official monthly settlement statement for each participant.

---

## Chapter 10: Market Suspension and Exemption

* **Market Suspension**: The spot market can be suspended under extreme conditions, such as:
    * Severe power supply shortages or major grid security risks (e.g., due to natural disasters).
    * Major power grid or generation facility failures.
    * Critical failure of the technical support systems.
    * During a suspension, the dispatch center will manage the grid based on security protocols rather than market outcomes.
* **Exemption from Liability (Force Majeure)**: Market participants are not held economically liable for failures caused by unforeseeable and unavoidable events (force majeure). The market operator is also exempt from liability when taking necessary actions to intervene or suspend the market to protect system security.

---

## Chapter 11: Information Disclosure

Transparency is a core principle, managed through a structured information disclosure system.

* **Information Types**:
    * **Public Information**: Available to everyone (e.g., market rules, registered participants, general market outcomes).
    * **Open Information**: Available to all market members (e.g., system load forecasts, aggregated generation data, transmission constraints, must-run unit lists).
    * **Private Information**: Available only to the specific market entity (e.g., their own bids, detailed settlement statements, individual dispatch instructions).
    * **By-Request Information**: Specific data (e.g., detailed grid models) available to participants after a formal application and confidentiality agreement.
* **Responsibilities**: The market operating institutions are responsible for managing the information disclosure platform. All members are responsible for the accuracy of the information they provide and are forbidden from disclosing private information improperly.
* **Information Sealing**: Critical data from each operating day (bids, clearing models, constraint data, manual interventions) must be securely archived for at least 5 years to allow for later review or dispute resolution.

---

## Chapter 12: Market Power Monitoring

Mechanisms are in place to detect and mitigate the abuse of market power.

* **Definition of Abuse**: Includes:
    * **Withholding**: Physical (deliberately making a unit unavailable) or economic (bidding a unit at a price far above its cost to raise the market price).
    * **Collusion**: Coordinating bids with other participants to maximize joint profits.
    * **Market Manipulation**: Spreading false information or submitting false data to disrupt the market.
* **Detection**: The market operator monitors for suspicious behavior, such as unusual bidding patterns, unexplained unit outages, or bids that deviate significantly from verified costs.
* **Mitigation**: If a generator with market power is detected, its bid may be replaced with a regulated price based on its verified marginal cost before the market is cleared. This prevents them from setting an artificially high price. Such incidents are reported to government regulators for potential penalties.

---

## Chapter 13: Market Dispute Resolution

A formal process exists for handling disputes between market members.

* **Types of Disputes**: Can relate to registration, rights and obligations, or transaction/metering/settlement issues.
* **Resolution Process**: The prescribed methods, in order of preference, are:
    1.  Negotiation between the parties.
    2.  Mediation or adjudication.
    3.  Formal arbitration.
    4.  Judicial litigation.

---

## Chapter 14: Market Supervision

Government bodies, such as the Provincial Development and Reform Commission, are responsible for the overall supervision of the spot market. They regulate fair competition, open access to the grid, and ensure market operators perform their duties impartially. Any individual or entity is prohibited from improperly interfering with market operations or manipulating the market.

---

## Chapter 15: Credit Management

A robust credit management system is in place to manage performance risk.

* **Credit Evaluation**: The trading institution periodically evaluates the credit rating of all market participants. Ratings are made public and fall on a nine-level scale (AAA down to C). A "C" rated entity is forcibly removed from the market.
* **Performance Guarantees (Letters of Guarantee)**: Participants must provide a financial guarantee (typically a bank-issued letter of guarantee) to the trading institution. The size of the required guarantee is based on the participant's trading volume and credit risk.
* **Credit Monitoring**: The trading institution monitors each entity's "credit utilization" (the ratio of their financial risk exposure to their credit limit). If utilization exceeds certain thresholds (e.g., 70% or 90%), warnings are issued. If it exceeds 100%, the entity's trading rights are suspended until they provide additional collateral.

---

## Appendices: Mathematical Models and Formulas

### Day-Ahead Market Mathematical Model

The day-ahead market clearing is based on a Security-Constrained Unit Commitment (SCUC) model.

* **Objective Function**: To minimize the total system cost, which includes generation costs, startup/shutdown costs, and penalty factors for violating transmission constraints.
    $$min \sum_{t=1}^{T} \sum_{i=1}^{N} [C_{i,t}(P_{i,t}) + C_{i,t}^{U} + C_{i,t}^{V}] + M \sum_{t=1}^{T} [\sum_{l=1}^{L}(SL_{l,t}^{+} + SL_{l,t}^{-}) + \sum_{s=1}^{S}(SL_{s,t}^{+} + SL_{s,t}^{-})]$$
* **Key Constraints**:
    * System Power Balance: Total generation must equal total load.
    * System Reserve: Total available capacity must meet positive and negative reserve requirements.
    * Unit Operating Limits: Each unit's output must be between its minimum and maximum levels.
    * Unit Ramp Rates: Changes in output cannot exceed the unit's ramp rate limits.
    * Minimum Up/Down Times: Units must remain on or off for a minimum duration after starting or stopping.
    * Network Constraints: Power flow on transmission lines and interfaces must not exceed their thermal or stability limits.

### Settlement Fee Formulas

* **Generator Revenue**: A generator's spot market revenue (or cost) for a given interval is calculated based on the deviation between their contracted position and their actual dispatched output, settled at spot prices.
    * **Day-Ahead Deviation Charge**: $R_{DA\_deviation} = (Q_{DA\_cleared} - Q_{MLT}) \times P_{DA\_zonal}$.
    * **Real-Time Deviation Charge**: $R_{RT\_deviation} = (Q_{Actual} - Q_{DA\_cleared}) \times P_{RT\_zonal}$.
* **Capacity Compensation Fee**: This fee is collected from users based on their spot market consumption and paid to generators based on their available capacity.
    $$R_{capacity\_fee, j} = (Q_{DA\_deviation, j} + Q_{RT\_deviation, j}) \times P_{capacity\_price}$$
* **Cost Compensation**: Includes startup costs, shutdown costs, and compensation for must-run units whose market revenue does not cover their operating costs. These are socialized across users.