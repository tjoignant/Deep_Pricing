import numpy as np

def GeneratePathsHestonEuler(S0: float, v0: float, risk_free_rate: float, maturity: float, rho: float, kappa: float,
                             theta: float, sigma: float, nb_steps=252, nb_simuls=100000, seed=1):
    """
    Inputs:
     - S0, v0         : initial asset spot and variance (float)
     - risk_free_rate : yearly asset continuous drift (float)
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
    return S.T


def Payoff(strike: float, barrier: float, S: list):
    """
    Inputs:
     - strike         : american D&O Call strike (float)
     - barrier        : american D&O Call barrier (float)
     - S              : asset prices over time (1D array)
    Outputs:
     - american D&O Call payoff (float)
    """
    if min(S) <= barrier:
        return 0
    else:
        return max(0, S[-1] - strike)


def MC_Pricing(strike: float, barrier: float, S0: float, v0: float, risk_free_rate: float, maturity: float, rho: float,
               kappa: float, theta: float, sigma: float, nb_steps=252, nb_simuls=100000, seed=1):
    """
    Inputs:
     - strike         : american D&O Call strike (float)
     - barrier        : american D&O Call barrier (float)
     - S0, v0         : initial asset spot and variance (float)
     - risk_free_rate : yearly asset continuous drift (float)
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
    S_matrix = GeneratePathsHestonEuler(S0=S0, v0=v0, risk_free_rate=risk_free_rate, maturity=maturity, rho=rho,
                                        kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps, nb_simuls=nb_simuls,
                                        seed=seed)
    payoffs = []
    for S in S_matrix:
        payoffs.append(Payoff(strike=strike, barrier=barrier, S=S))
    return np.exp(-risk_free_rate * maturity) * np.mean(payoffs)


def DeltaFD(strike: float, barrier: float, S0: float, v0: float, risk_free_rate: float, maturity: float, rho: float,
            kappa: float, theta: float, sigma: float, nb_steps=252, nb_simuls=100000, seed=1, dS0=pow(10, -4)):
    """
    Inputs:
     - strike         : american D&O Call strike (float)
     - barrier        : american D&O Call barrier (float)
     - S0, v0         : initial asset spot and variance (float)
     - risk_free_rate : yearly asset continuous drift (float)
     - maturity       : yearly duration of simulation (float)
     - rho            : correlation between asset returns and variance (float)
     - kappa          : rate of mean reversion in variance process (float)
     - theta          : long-term mean of variance process (float)
     - sigma          : vol of vol / volatility of variance process (float)
     - nb_steps       : number of time steps (int)
     - nb_simuls      : number of simulations (int)
     - seed           : random seed (int)
     - dS0            : S0 differential (float)
    Outputs:
     - american D&O Call delta (float)
    """
    price_up = MC_Pricing(strike=strike, barrier=barrier, S0=S0+dS0, v0=v0, risk_free_rate=risk_free_rate,
                          maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                          nb_simuls=nb_simuls, seed=seed)
    price_down = MC_Pricing(strike=strike, barrier=barrier, S0=S0-dS0, v0=v0, risk_free_rate=risk_free_rate,
                          maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                          nb_simuls=nb_simuls, seed=seed)
    return (price_up - price_down) / (2 * dS0)


def GammaFD(strike: float, barrier: float, S0: float, v0: float, risk_free_rate: float, maturity: float, rho: float,
            kappa: float, theta: float, sigma: float, nb_steps=252, nb_simuls=100000, seed=1, dS0=pow(10, -4)):
    """
    Inputs:
     - strike         : american D&O Call strike (float)
     - barrier        : american D&O Call barrier (float)
     - S0, v0         : initial asset spot and variance (float)
     - risk_free_rate : yearly asset continuous drift (float)
     - maturity       : yearly duration of simulation (float)
     - rho            : correlation between asset returns and variance (float)
     - kappa          : rate of mean reversion in variance process (float)
     - theta          : long-term mean of variance process (float)
     - sigma          : vol of vol / volatility of variance process (float)
     - nb_steps       : number of time steps (int)
     - nb_simuls      : number of simulations (int)
     - seed           : random seed (int)
     - dS0            : S0 differential (float)
    Outputs:
     - american D&O Call gamma (float)
    """
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
          kappa: float, theta: float, sigma: float, nb_steps=252, nb_simuls=100000, seed=1, dr=pow(10, -4)):
    """
    Inputs:
     - strike         : american D&O Call strike (float)
     - barrier        : american D&O Call barrier (float)
     - S0, v0         : initial asset spot and variance (float)
     - risk_free_rate : yearly asset continuous drift (float)
     - maturity       : yearly duration of simulation (float)
     - rho            : correlation between asset returns and variance (float)
     - kappa          : rate of mean reversion in variance process (float)
     - theta          : long-term mean of variance process (float)
     - sigma          : vol of vol / volatility of variance process (float)
     - nb_steps       : number of time steps (int)
     - nb_simuls      : number of simulations (int)
     - seed           : random seed (int)
     - dr             : risk_free_rate differential (float)
    Outputs:
     - american D&O Call rho (float)
    """
    price_up = MC_Pricing(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate+dr,
                          maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                          nb_simuls=nb_simuls, seed=seed)
    price_down = MC_Pricing(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate-dr,
                          maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                          nb_simuls=nb_simuls, seed=seed)
    return (price_up - price_down) / (2 * dr) / 100


def VegaFD(strike: float, barrier: float, S0: float, v0: float, risk_free_rate: float, maturity: float, rho: float,
           kappa: float, theta: float, sigma: float, nb_steps=252, nb_simuls=100000, seed=1, dv=pow(10, -4)):
    """
    Inputs:
     - strike         : american D&O Call strike (float)
     - barrier        : american D&O Call barrier (float)
     - S0, v0         : initial asset spot and variance (float)
     - risk_free_rate : yearly asset continuous drift (float)
     - maturity       : yearly duration of simulation (float)
     - rho            : correlation between asset returns and variance (float)
     - kappa          : rate of mean reversion in variance process (float)
     - theta          : long-term mean of variance process (float)
     - sigma          : vol of vol / volatility of variance process (float)
     - nb_steps       : number of time steps (int)
     - nb_simuls      : number of simulations (int)
     - seed           : random seed (int)
     - dv             : v0 & theta differential (float)
    Outputs:
     - american D&O Call rho (float)
    """
    price_v0_up = MC_Pricing(strike=strike, barrier=barrier, S0=S0, v0=v0+dv, risk_free_rate=risk_free_rate,
                          maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                          nb_simuls=nb_simuls, seed=seed)
    price_v0_down = MC_Pricing(strike=strike, barrier=barrier, S0=S0, v0=v0-dv, risk_free_rate=risk_free_rate,
                          maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                          nb_simuls=nb_simuls, seed=seed)
    price_theta_up = MC_Pricing(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate,
                          maturity=maturity, rho=rho, kappa=kappa, theta=theta+dv, sigma=sigma, nb_steps=nb_steps,
                          nb_simuls=nb_simuls, seed=seed)
    price_theta_down = MC_Pricing(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate,
                          maturity=maturity, rho=rho, kappa=kappa, theta=theta-dv, sigma=sigma, nb_steps=nb_steps,
                          nb_simuls=nb_simuls, seed=seed)
    vega_v0 = (price_v0_up - price_v0_down) / (2 * dv) * 2 * np.sqrt(v0)
    vega_theta = (price_theta_up - price_theta_down) / (2 * dv) * 2 * np.sqrt(theta)
    return (vega_v0 + vega_theta) / 100


def StandardError(Y: list, nb_simuls: int):
    """
    Inputs:
     - Y              : payoffs (1D array)
     - nb_simuls      : number of simulations (int)
    Outputs:
     - Standard error (float)
    """
    return np.sqrt(np.var(Y) / nb_simuls)
