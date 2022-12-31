import torch
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
     - variance over time (2D array)
    """
    np.random.seed(seed)
    dt = maturity / nb_steps
    mu = torch.tensor([0, 0])
    cov = torch.tensor([[1, rho],
                        [rho, 1]])
    S_it = torch.ones(nb_simuls) * S0
    v_it = torch.ones(nb_simuls) * v0
    Z = torch.from_numpy(np.random.multivariate_normal(mu, cov, (nb_steps, nb_simuls)))
    S = S_it
    for rdn in Z:
        S_it = S_it * torch.exp((risk_free_rate - 0.5 * v_it) * dt + torch.sqrt(v_it * dt) * rdn[:, 0])
        v_it = torch.maximum(v_it + kappa * (theta - v_it) * dt + sigma * torch.sqrt(v_it * dt) * rdn[:, 1], torch.tensor(0))
        S = torch.cat((S, S_it), 0)
    S = S.view(nb_steps + 1, nb_simuls)
    return S.T


def Payoff(strike: float, barrier: float, S: np.array):
    """
    Inputs:
     - strike         : american D&O Call strike (float)
     - barrier        : american D&O Call barrier (float)
     - S              : asset prices over time (1D tensor)
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
    payoffs_sum = 0
    for S in S_matrix:
        payoffs_sum = payoffs_sum + Payoff(strike=strike, barrier=barrier, S=S)
    return torch.exp(-risk_free_rate * maturity) * (payoffs_sum / len(S_matrix))


def DeltaFD(strike: float, barrier: float, S0: float, v0: float, risk_free_rate: float, maturity: float, rho: float,
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


def DeltaAAD(strike: float, barrier: float, S0: float, v0: float, risk_free_rate: float, maturity: float, rho: float,
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
     - dS0            : S0 differential
    Outputs:
     - american D&O Call delta (float)
    """
    S0.requires_grad = True
    price = MC_Pricing(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate,
                       maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                       nb_simuls=nb_simuls, seed=seed)
    if price.grad_fn:
        price.backward()
        return S0.grad.clone()
    else:
        return 0


def GammaAAD(strike: float, barrier: float, S0: float, v0: float, risk_free_rate: float, maturity: float, rho: float,
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
     - dS0            : S0 differential
    Outputs:
     - american D&O Call gamma (float)
    """
    S0.requires_grad = True
    delta = DeltaFD(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate,
                          maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                          nb_simuls=nb_simuls, seed=seed)
    if delta.grad_fn:
        delta.backward()
        return S0.grad.clone()
    else:
        return 0


def RhoAAD(strike: float, barrier: float, S0: float, v0: float, risk_free_rate: float, maturity: float, rho: float,
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
     - dS0            : S0 differential
    Outputs:
     - american D&O Call rho (float)
    """
    risk_free_rate.requires_grad = True
    price = MC_Pricing(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate,
                       maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                       nb_simuls=nb_simuls, seed=seed)
    if price.grad_fn:
        price.backward()
        return risk_free_rate.grad.clone() / 100
    else:
        return 0


def VegaAAD(strike: float, barrier: float, S0: float, v0: float, risk_free_rate: float, maturity: float, rho: float,
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
     - dS0            : S0 differential
    Outputs:
     - american D&O Call vega (float)
    """
    v0.requires_grad = True
    theta.requires_grad = True
    price = MC_Pricing(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate,
                       maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                       nb_simuls=nb_simuls, seed=seed)
    if price.grad_fn:
        price.backward()
        return (v0.grad.clone() * 2 * torch.sqrt(v0) + theta.grad.clone() * 2 * torch.sqrt(theta)) / 100
    else:
        return 0


def HestonLSM(strike: float, barrier: float, v0: float, risk_free_rate: float, maturity: float, rho: float,
               kappa: float, theta: float, sigma: float, nb_steps=252, nb_simuls=100000, seed=1):
    """
    Inputs:
     - strike         : american D&O Call strike (float)
     - barrier        : american D&O Call barrier (float)
     - v0             : initial asset variance (float)
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
     - training samples (1D array)
     - labels (1D array)
     - pathwise differentials (1D array)
    """
    seed_list = np.arange(seed, nb_simuls + seed)
    X_list = torch.linspace(10, 200, nb_simuls)
    Y_list = []
    dYdX_list = []
    for S0, seed in zip(X_list, seed_list):
        S_matrix = GeneratePathsHestonEuler(S0=S0, v0=v0, risk_free_rate=risk_free_rate, maturity=maturity, rho=rho,
                                            kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps, nb_simuls=1,
                                            seed=seed)
        Y_list.append(Payoff(strike=strike, barrier=barrier, S=S_matrix[0]))
        dYdX_list.append(DeltaAAD(strike=strike, barrier=barrier, S0=S0, v0=v0, risk_free_rate=risk_free_rate,
                                  maturity=maturity, rho=rho, kappa=kappa, theta=theta, sigma=sigma, nb_steps=nb_steps,
                                  nb_simuls=1, seed=seed))
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
    X_mean = torch.mean(X)
    X_std = torch.std(X)
    X_norm = (X - X_mean) / X_std
    Y_mean = torch.mean(Y)
    Y_std = torch.std(Y)
    Y_norm = (Y - Y_mean) / Y_std
    dYdX_mean = torch.mean(dYdX)
    dYdX_std = torch.std(dYdX)
    dYdX_norm = (dYdX - dYdX_mean) / dYdX_std
    lambda_j = 1 / torch.sqrt((1/len(dYdX)) * torch.sum(dYdX_norm * dYdX_norm))
    return X_mean, X_std, X_norm, Y_mean, Y_std, Y_norm, dYdX_norm, lambda_j
