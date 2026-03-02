This repo is created for WQU Capstone project
Adaptive Regime-Switching Mean-Reversion on the NIFTY-50 Index


Following commands need to be run to setup the project


git clone https://github.com/data-decode/wqu-capstone.git

cd wqu-capstone

pip install -r requirements.txt

pip install -e .

jupyter notebook



                                                            Abstract
In this project, we design a regime-aware mean-reversion (MR) trading framework for the NIFTY-50 to enhance robustness, tail-risk control, and performance stability. It mixes market-dynamics diagnostics, fractal persistence metrics, and Hidden Markov Model–based volatility regimes, with regime-sensitive position sizing and layered risk controls. We analysed the NIFTY 50 data with different frequencies to generate signals that perform well in anti-persistent, low-trend environments, usually fail in trending, high-volatility, or in so-called event-driven regimes. The project, which we are developing, explores how fractal persistence measures, latent volatility regime models, and market-context conditioning can be integrated into a trading framework in a way that is adaptive. This framework will have several parts as follows: regime-sensitive risk allocation, as well as layered tail-risk controls. Strategy variants are evaluated net of transaction costs using walk-forward validation and controlled ablations to isolate causal contributions. 
