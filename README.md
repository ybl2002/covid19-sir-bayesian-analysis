COVID-19 SIR Model Bayesian Analysis
A comprehensive implementation of Bayesian inference for the SIR (Susceptible-Infected-Recovered) epidemiological model applied to COVID-19 data, featuring both MCMC (Markov Chain Monte Carlo) and SMC (Sequential Monte Carlo) methods.
Features

Dual Bayesian Approaches: Implementation of both fixed-parameter MCMC and time-varying SMC methods
Custom MCMC Sampler: Self-implemented Metropolis-Hastings algorithm without external MCMC libraries
Sequential Monte Carlo: Particle filter implementation for tracking time-varying transmission rates
Comprehensive Diagnostics: Autocorrelation analysis, effective sample size calculation, and convergence diagnostics
Professional Visualizations: Publication-ready figures including posterior distributions, time series, and method comparisons
Complete Analysis Pipeline: From data preprocessing to final report generation

Requirements
pythonpandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scipy >= 1.7.0
tqdm >= 4.62.0
Installation

Clone the repository:

bashgit clone https://github.com/yourusername/covid19-sir-bayesian-analysis.git
cd covid19-sir-bayesian-analysis

Install required packages:

bashpip install -r requirements.txt
Usage

Data Preparation:

Place your preprocessed SIR data in the specified directory
The code expects a CSV file named sir_data_China.csv with columns: date, population, S, I, R, confirmed, daily_new_cleaned


Run the Analysis:

pythonpython covid19_sir_bayesian_analysis.py

Outputs: All results will be saved to the output directory including:

Parameter estimates
MCMC samples
Model fit results
Visualizations
Analysis reports



Code Structure
The analysis is organized into 15 main sections:

Configuration and Data Preparation: Load data and set up parameters
SIR Model Definition: Define differential equations and solver
Bayesian Model Class: Implementation of prior, likelihood, and posterior
MCMC Sampling: Custom Metropolis-Hastings implementation
MCMC Diagnostics: Autocorrelation analysis and convergence checks
Posterior Analysis: Statistical summary of posterior distributions
Model Fit Evaluation: Calculate fit metrics (RMSE, R²)
Sequential Monte Carlo: Time-varying parameter estimation
Posterior Comparison: MCMC vs SMC comparison visualization
SMC Time Series: Epidemic dynamics visualization
Beta Evolution: Posterior distribution evolution over time
Method Comparison: Time series comparison of fixed vs time-varying parameters
Results Saving: Export all numerical results
Comprehensive Visualization: MCMC results summary
Report Generation: Automated analysis report