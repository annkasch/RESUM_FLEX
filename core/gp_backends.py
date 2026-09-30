"""Lazy factory for direct count-GP backends, independent of event surrogates."""


def build_count_gp_backend(config):
    if config.kind == "binomial_laplace":
        from core.binomial_gp import BinomialGP

        return BinomialGP(kernel=config.kernel)
    raise ValueError(f"Unsupported count-GP backend: {config.kind}")
