import numpy as np


def GeneratePathsHestonEuler(S0: float, v0: float, risk_free_rate: float, maturity: float, rho: float, kappa: float,
                             theta: float, sigma: float, nb_steps=252, nb_simuls=100000, seed=1):
    """
    Inputs:
     - S0, v0         : initial asset spot and variance (float)
     - risk_free_rate : yearly asset continuous drift (perc)
     - maturity       : yearly duration of simulation (float)
     - rho            : correlation between asset returns and variance (float)
     - kappa          : rate of mean reversion in variance process (float)
     - theta          : long-term mean of variance process (float)
     - sigma          : vol of vol / volatility of variance process (float)
     - nb_steps       : number of time steps (int)
     - nb_simuls      : number of simulations (int)
     - seed           : random seed (int)
    Outputs:
     - asset prices over time (2D array)
     - variance over time (2D array)
    """
    np.random.seed(seed)
    dt = maturity / nb_steps
    mu = np.array([0, 0])
    cov = np.array([[1, rho],
                    [rho, 1]])
    S = np.full(shape=(nb_steps + 1, nb_simuls), fill_value=float(S0))
    v = np.full(shape=(nb_steps + 1, nb_simuls), fill_value=float(v0))
    Z = np.random.multivariate_normal(mu, cov, (nb_steps, nb_simuls))
    for i in range(1, nb_steps + 1):
        S[i] = S[i - 1] * np.exp((risk_free_rate - 0.5 * v[i - 1]) * dt + np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 0])
        v[i] = np.maximum(v[i - 1] + kappa * (theta - v[i - 1]) * dt + sigma * np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 1], 0)
    return S.T, v.T


def Payoff(strike: float, barrier: float, S: np.array):
    """
    Inputs:
     - strike         : american D&O Call strike (float)
     - barrier        : american D&O Call barrier (float)
     - S              : asset prices over time (1D array)
    Outputs:
     - american D&O Call payoff (float)
    """
    if min(S) <= barrier:
        payoff = 0
    else:
        payoff = max(0, S[-1] - strike)
    return payoff    


def MC_Pricing(strike: float, barrier: float, S0: float, v0: float, risk_free_rate: float, maturity: float, rho: float,
               kappa: float, theta: float, sigma: float, nb_steps=252, nb_simuls=100000, seed=1):
    """
    Inputs:
     - strike         : american D&O Call strike (float)
     - barrier        : american D&O Call barrier (float)
     - S0, v0         : initial asset spot and variance (float)
     - risk_free_rate : yearly asset continuous drift (perc)
     - maturity       : yearly duration of simulation (float)
     - rho            : correlation between asset returns and variance (float)
     - kappa          : rate of mean reversion in variance process (float)
     - theta          : long-term mean of variance process (float)
     - sigma          : vol of vol / volatility of variance process (float)
     - nb_steps       : number of time steps (int)
     - nb_simuls      : number of simulations (int)
     - seed           : random seed (int)
    Outputs:
     - american D&O Call price (float)
    """
    S_matrix, V_matrix = GeneratePathsHestonEuler(S0=S0, v0=v0, risk_free_rate=risk_free_rate, maturity=maturity, rho=rho,
                                           kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                                           nb_simuls=nb_simuls, seed=seed)
    payoffs = []
    for S in S_matrix:
        payoffs.append(Payoff(strike=strike, barrier=barrier, S=S))
    return np.exp(-risk_free_rate * maturity) * np.mean(payoffs)


def DeltaFD(strike: float, barrier: float, S0: float, v0: float, risk_free_rate: float, maturity: float, rho: float,
            kappa: float, theta: float, sigma: float, nb_steps=252, nb_simuls=100000, seed=1):
    """
    Inputs:
     - strike         : american D&O Call strike (float)
     - barrier        : american D&O Call barrier (float)
     - S0, v0         : initial asset spot and variance (float)
     - risk_free_rate : yearly asset continuous drift (perc)
     - maturity       : yearly duration of simulation (float)
     - rho            : correlation between asset returns and variance (float)
     - kappa          : rate of mean reversion in variance process (float)
     - theta          : long-term mean of variance process (float)
     - sigma          : vol of vol / volatility of variance process (float)
     - nb_steps       : number of time steps (int)
     - nb_simuls      : number of simulations (int)
     - seed           : random seed (int)
     - dS0            : S0 differential
    Outputs:
     - american D&O Call delta (float)
    """
    dS0 = S0 / 200
    price_up = MC_Pricing(strike=strike, barrier=barrier, S0=S0+dS0, v0=v0, risk_free_rate=risk_free_rate,
                          maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                          nb_simuls=nb_simuls, seed=seed)
    price_down = MC_Pricing(strike=strike, barrier=barrier, S0=S0-dS0, v0=v0, risk_free_rate=risk_free_rate,
                          maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                          nb_simuls=nb_simuls, seed=seed)
    return (price_up - price_down) / (2 * dS0)


def GammaFD(strike: float, barrier: float, S0: float, v0: float, risk_free_rate: float, maturity: float, rho: float,
            kappa: float, theta: float, sigma: float, nb_steps=252, nb_simuls=100000, seed=1):
    """
    Inputs:
     - strike         : american D&O Call strike (float)
     - barrier        : american D&O Call barrier (float)
     - S0, v0         : initial asset spot and variance (float)
     - risk_free_rate : yearly asset continuous drift (perc)
     - maturity       : yearly duration of simulation (float)
     - rho            : correlation between asset returns and variance (float)
     - kappa          : rate of mean reversion in variance process (float)
     - theta          : long-term mean of variance process (float)
     - sigma          : vol of vol / volatility of variance process (float)
     - nb_steps       : number of time steps (int)
     - nb_simuls      : number of simulations (int)
     - seed           : random seed (int)
     - dS0            : S0 differential
    Outputs:
     - american D&O Call gamma (float)
    """
    dS0 = S0 / 200
    price_up = MC_Pricing(strike=strike, barrier=barrier, S0=S0+dS0, v0=v0, risk_free_rate=risk_free_rate,
                          maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                          nb_simuls=nb_simuls, seed=seed)
    price = MC_Pricing(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate,
                       maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                       nb_simuls=nb_simuls, seed=seed)
    price_down = MC_Pricing(strike=strike, barrier=barrier, S0=S0-dS0, v0=v0, risk_free_rate=risk_free_rate,
                          maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                          nb_simuls=nb_simuls, seed=seed)
    return (price_up - 2 * price + price_down) / (pow(dS0, 2))


def RhoFD(strike: float, barrier: float, S0: float, v0: float, risk_free_rate: float, maturity: float, rho: float,
          kappa: float, theta: float, sigma: float, nb_steps=252, nb_simuls=100000, seed=1):
    """
    Inputs:
     - strike         : american D&O Call strike (float)
     - barrier        : american D&O Call barrier (float)
     - S0, v0         : initial asset spot and variance (float)
     - risk_free_rate : yearly asset continuous drift (perc)
     - maturity       : yearly duration of simulation (float)
     - rho            : correlation between asset returns and variance (float)
     - kappa          : rate of mean reversion in variance process (float)
     - theta          : long-term mean of variance process (float)
     - sigma          : vol of vol / volatility of variance process (float)
     - nb_steps       : number of time steps (int)
     - nb_simuls      : number of simulations (int)
     - seed           : random seed (int)
     - dr             : risk_free_rate differential
    Outputs:
     - american D&O Call rho (float)
    """
    dr = 0.005
    price_up = MC_Pricing(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate+dr,
                          maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                          nb_simuls=nb_simuls, seed=seed)
    price_down = MC_Pricing(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate-dr,
                          maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                          nb_simuls=nb_simuls, seed=seed)
    return (price_up - price_down) / (2 * dr) / 100


def StandardError(nb_simuls : int ,payoffvec):
    """
    Inputs:
     - nb_simuls      : number of simulations (int)
     - payoffvec      : payoff vector (table of float)
    Outputs:
     - Interval Confidence
    """
    return [np.mean(payoffvec)+1.96*np.std(payoffvec)/np.sqrt(nb_simuls),np.mean(payoffvec)-1.96*np.std(payoffvec)/np.sqrt(nb_simuls)]

def LSM_dataset(strike: float, barrier: float, v0: float, risk_free_rate: float, maturity: float, rho: float,
               kappa: float, theta: float, sigma: float, nb_steps=252, nb_simuls=100000, seed=1):
    """
    Inputs:
     - strike         : american D&O Call strike (float)
     - barrier        : american D&O Call barrier (float)
     - v0             : initial asset variance (float)
     - risk_free_rate : yearly asset continuous drift (perc)
     - maturity       : yearly duration of simulation (float)
     - rho            : correlation between asset returns and variance (float)
     - kappa          : rate of mean reversion in variance process (float)
     - theta          : long-term mean of variance process (float)
     - sigma          : vol of vol / volatility of variance process (float)
     - nb_steps       : number of time steps (int)
     - nb_simuls      : number of simulations (int)
     - seed           : random seed (int)
    Outputs:
     - training samples (1D array)
     - labels (1D array)
     - pathwise differentials (1D array)
    """
    seed_list = np.arange(seed, nb_simuls + seed)
    X_list = np.linspace(10, 200, nb_simuls)
    Y_list = []
    dYdX_list = []
    for S0, seed in zip(X_list, seed_list):
        S_matrix, V_matrix = GeneratePathsHestonEuler(S0=S0, v0=v0, risk_free_rate=risk_free_rate, maturity=maturity,
                                                      rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                                                      nb_simuls=1, seed=seed)
        Y_list.append(Payoff(strike=strike, barrier=barrier, S=S_matrix[0]))
    return X_list, Y_list, dYdX_list


def normalize_data(X: list, Y: list, dYdX: list):
    """
    Inputs:
     - training samples (1D array)
     - labels (1D array)
     - pathwise differentials (1D array)
    Outputs:
     - training samples mean (float)
     - training samples stdev (float)
     - normalized training samples (1D array)
     - labels mean (float)
     - labels stdev (float)
     - labels samples (1D array)
     - normalized pathwise differentials (1D array)
     - differential weights of the cost function (float)
    """
    mean_X = np.mean(X)
    std_X = np.std(X)
    norm_X = (X - mean_X) / std_X
    mean_Y = np.mean(Y)
    std_Y = np.std(Y)
    norm_Y = (Y - mean_Y) / std_Y
    mean_dYdX = np.mean(dYdX)
    std_dYdX = np.std(dYdX)
    norm_dYdX = (dYdX - mean_dYdX) / std_dYdX
    lambda_j = 1 / np.sqrt((1/len(dYdX)) * sum(np.power(norm_dYdX, 2)))
    return mean_X, std_X, norm_X, mean_Y, std_Y, norm_Y, norm_dYdX, lambda_j
