import numpy as np
from scipy.stats import norm


class ExoticOptionPricer:
    """
    Monte Carlo pricer for exotic options.

    GBM paths and their driving standard normals (Z) are simulated once on
    construction.  Paths can be exactly reconstructed for any (S0, sigma)
    via ``paths_from_params``, which lets greek finite-differences reuse the
    same randomness and avoids extra MC noise.
    """

    def __init__(self, S0, r, sigma, T, Npaths, Nsteps):
        self.S0     = S0
        self.r      = r
        self.sigma  = sigma
        self.T      = T
        self.Npaths = Npaths
        self.Nsteps = Nsteps
        self.dt     = T / Nsteps
        self.t, self.S, self.Z = self._simulate_gbm_paths()

    def _simulate_gbm_paths(self):
        t = np.linspace(0, self.T, self.Nsteps + 1)
        Z = np.random.standard_normal((self.Npaths, self.Nsteps))
        S = np.zeros((self.Npaths, self.Nsteps + 1))
        S[:, 0] = self.S0
        for i in range(1, self.Nsteps + 1):
            S[:, i] = S[:, i - 1] * np.exp(
                (self.r - 0.5 * self.sigma ** 2) * self.dt
                + self.sigma * np.sqrt(self.dt) * Z[:, i - 1]
            )
        return t, S, Z

    def paths_from_params(self, S0, sigma):
        """Reconstruct GBM paths from stored normals for arbitrary (S0, sigma)."""
        S = np.zeros((self.Npaths, self.Nsteps + 1))
        S[:, 0] = S0
        for i in range(1, self.Nsteps + 1):
            S[:, i] = S[:, i - 1] * np.exp(
                (self.r - 0.5 * sigma ** 2) * self.dt
                + sigma * np.sqrt(self.dt) * self.Z[:, i - 1]
            )
        return S

    # ── Analytical ────────────────────────────────────────────────────────

    def black_scholes_option(self, K, option_type='call'):
        S0, T, r, sigma = self.S0, self.T, self.r, self.sigma
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

    def price_digital_option_with_spread(self, K, spread):
        call_high = self.black_scholes_option(K + spread, option_type='call')
        call_low  = self.black_scholes_option(K,          option_type='call')
        return (call_low - call_high) / (2 * spread)

    # ── Monte Carlo (use self.S) ──────────────────────────────────────────

    def price_up_and_out(self, K, barrier, option_type='call'):
        knocked_out = np.max(self.S, axis=1) >= barrier
        S_T = self.S[:, -1]
        if option_type == 'call':
            payoff = np.where(~knocked_out, np.maximum(S_T - K, 0.0), 0.0)
        else:
            payoff = np.where(~knocked_out, np.maximum(K - S_T, 0.0), 0.0)
        return np.exp(-self.r * self.T) * np.mean(payoff)

    def price_lookback_option(self, lookback_type='floating', option_type='call'):
        S_min = np.min(self.S, axis=1)
        S_max = np.max(self.S, axis=1)
        S_T   = self.S[:, -1]
        if lookback_type == 'floating':
            payoff = (np.maximum(S_T - S_min, 0) if option_type == 'call'
                      else np.maximum(S_max - S_T, 0))
        else:  # fixed
            payoff = (np.maximum(S_max - self.S0, 0) if option_type == 'call'
                      else np.maximum(self.S0 - S_min, 0))
        return np.exp(-self.r * self.T) * np.mean(payoff)

    def price_asian_option(self, K, option_type='call'):
        avg = np.mean(self.S[:, 1:], axis=1)
        payoff = (np.maximum(avg - K, 0) if option_type == 'call'
                  else np.maximum(K - avg, 0))
        return np.exp(-self.r * self.T) * np.mean(payoff)


# ── Portfolio building blocks ─────────────────────────────────────────────────

class OptionPosition:
    """
    A single option leg: type, signed quantity (positive = long, negative = short),
    and type-specific parameters.

    Supported types and required parameters
    ───────────────────────────────────────
    european_call / european_put         K
    digital_call                         K, spread
    up_and_out_call / up_and_out_put     K, barrier
    lookback_floating_call/put           (none)
    lookback_fixed_call/put              (none)
    asian_call / asian_put               K
    """

    REQUIRED_PARAMS = {
        'european_call':          ('K',),
        'european_put':           ('K',),
        'digital_call':           ('K', 'spread'),
        'up_and_out_call':        ('K', 'barrier'),
        'up_and_out_put':         ('K', 'barrier'),
        'lookback_floating_call': (),
        'lookback_floating_put':  (),
        'lookback_fixed_call':    (),
        'lookback_fixed_put':     (),
        'asian_call':             ('K',),
        'asian_put':              ('K',),
    }

    _DISPATCH = {
        'european_call':          lambda pr, p: pr.black_scholes_option(p['K'], option_type='call'),
        'european_put':           lambda pr, p: pr.black_scholes_option(p['K'], option_type='put'),
        'digital_call':           lambda pr, p: pr.price_digital_option_with_spread(p['K'], p['spread']),
        'up_and_out_call':        lambda pr, p: pr.price_up_and_out(p['K'], p['barrier'], option_type='call'),
        'up_and_out_put':         lambda pr, p: pr.price_up_and_out(p['K'], p['barrier'], option_type='put'),
        'lookback_floating_call': lambda pr, p: pr.price_lookback_option('floating', 'call'),
        'lookback_floating_put':  lambda pr, p: pr.price_lookback_option('floating', 'put'),
        'lookback_fixed_call':    lambda pr, p: pr.price_lookback_option('fixed', 'call'),
        'lookback_fixed_put':     lambda pr, p: pr.price_lookback_option('fixed', 'put'),
        'asian_call':             lambda pr, p: pr.price_asian_option(p['K'], option_type='call'),
        'asian_put':              lambda pr, p: pr.price_asian_option(p['K'], option_type='put'),
    }

    def __init__(self, option_type, quantity, **params):
        if option_type not in self.REQUIRED_PARAMS:
            raise ValueError(
                f"Unknown option type '{option_type}'.\n"
                f"Valid types: {sorted(self.REQUIRED_PARAMS)}"
            )
        missing = [p for p in self.REQUIRED_PARAMS[option_type] if p not in params]
        if missing:
            raise ValueError(f"'{option_type}' is missing parameters: {missing}")
        self.option_type = option_type
        self.quantity    = quantity
        self.params      = params

    def price(self, pricer):
        return self._DISPATCH[self.option_type](pricer, self.params)

    def __repr__(self):
        sign       = '+' if self.quantity >= 0 else ''
        param_str  = ', '.join(f"{k}={v}" for k, v in self.params.items())
        param_part = f"({param_str})" if param_str else ''
        return f"{sign}{self.quantity} × {self.option_type}{param_part}"


class OptionPortfolio:
    """
    A portfolio of option positions backed by a shared ``ExoticOptionPricer``.

    Greeks are computed by finite differences, reconstructing paths from the
    stored driving normals for each bump — so sigma bumps produce zero extra
    Monte Carlo noise.

    Typical usage
    ─────────────
    pricer = ExoticOptionPricer(S0=100, r=0.05, sigma=0.2,
                                T=1.0, Npaths=100_000, Nsteps=252)
    pf = OptionPortfolio(pricer)
    (pf
     .add('european_call',    quantity= 2, K=100)
     .add('european_put',     quantity= 1, K=100)
     .add('asian_call',       quantity= 3, K=105)
     .add('asian_put',        quantity=-2, K=95)
     .add('up_and_out_call',  quantity= 1, K=100, barrier=120))
    pf.summary()
    pf.print_greeks()
    """

    def __init__(self, pricer: ExoticOptionPricer):
        self.pricer    = pricer
        self.positions = []

    def add(self, option_type, quantity, **params):
        """Add a position.  Returns self so calls can be chained."""
        self.positions.append(OptionPosition(option_type, quantity, **params))
        return self

    def price(self):
        """Total marked-to-model value of the portfolio."""
        return sum(pos.quantity * pos.price(self.pricer) for pos in self.positions)

    def summary(self):
        """Print a per-leg breakdown and total portfolio value."""
        col = (46, 6, 12, 12)
        header = f"{'Position':<{col[0]}} {'Qty':>{col[1]}} {'Unit Price':>{col[2]}} {'Value':>{col[3]}}"
        sep    = '─' * sum(col)
        print(f"\n{header}\n{sep}")
        total = 0.0
        for pos in self.positions:
            unit  = pos.price(self.pricer)
            value = pos.quantity * unit
            total += value
            print(f"{str(pos):<{col[0]}} {pos.quantity:>{col[1]}} "
                  f"{unit:>{col[2]}.4f} {value:>{col[3]}.4f}")
        print(sep)
        print(f"{'Portfolio Total':>{col[0] + col[1] + col[2] + 2}} {total:>{col[3]}.4f}")

    # ── Greeks ────────────────────────────────────────────────────────────

    def _price_with(self, S0, sigma):
        """Price portfolio with (S0, sigma), keeping stored normals fixed."""
        pr = self.pricer
        orig_S, orig_S0, orig_sigma = pr.S, pr.S0, pr.sigma
        pr.S     = pr.paths_from_params(S0, sigma)
        pr.S0    = S0
        pr.sigma = sigma
        val      = self.price()
        pr.S, pr.S0, pr.sigma = orig_S, orig_S0, orig_sigma
        return val

    def greeks(self, eps_S_frac=1e-4, eps_v=1e-4):
        """
        Compute delta, gamma, vega, vanna, volga via central finite differences.

        Parameters
        ----------
        eps_S_frac : relative S0 bump  (default 0.01% of S0)
        eps_v      : absolute sigma bump (default 0.0001)

        Returns
        -------
        dict with keys: delta, gamma, vega, vanna, volga
        """
        S0    = self.pricer.S0
        v0    = self.pricer.sigma
        eS    = S0 * eps_S_frac
        ev    = eps_v

        p0  = self._price_with(S0,      v0     )
        pSu = self._price_with(S0 + eS, v0     )
        pSd = self._price_with(S0 - eS, v0     )
        pVu = self._price_with(S0,      v0 + ev)
        pVd = self._price_with(S0,      v0 - ev)
        ppp = self._price_with(S0 + eS, v0 + ev)
        ppm = self._price_with(S0 - eS, v0 + ev)
        pmp = self._price_with(S0 + eS, v0 - ev)
        pmm = self._price_with(S0 - eS, v0 - ev)

        return {
            'delta': (pSu - pSd)              / (2 * eS),
            'gamma': (pSu - 2*p0 + pSd)       / eS**2,
            'vega':  (pVu - pVd)              / (2 * ev),
            'vanna': (ppp - ppm - pmp + pmm)  / (4 * eS * ev),
            'volga': (pVu - 2*p0 + pVd)       / ev**2,
        }

    def print_greeks(self):
        g = self.greeks()
        print("\nPortfolio Greeks:")
        for name in ('delta', 'gamma', 'vega', 'vanna', 'volga'):
            print(f"  {name.capitalize():<8}: {g[name]:>14.6f}")


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    pricer = ExoticOptionPricer(
        S0=22713, r=0.1, sigma=0.25, T=10.0/365, Npaths=100_000, Nsteps=252
    )

    # ── Construct a portfolio ─────────────────────────────────────────────
    pf = OptionPortfolio(pricer)

    (pf
     .add('european_call',          quantity= 2, K=23000)           # long 2 ATM calls
     .add('european_put',           quantity= 0, K=100)           # long 1 ATM put
     .add('digital_call',           quantity= 0, K=100, spread=1) # long 5 digitals
     .add('up_and_out_call',        quantity= 0, K=100, barrier=120)
     .add('lookback_floating_call', quantity= 0)
     .add('asian_call',             quantity= 0, K=105)
     .add('asian_put',              quantity= 0, K=95))            # short 2 asian puts

    pf.summary()
    pf.print_greeks()
