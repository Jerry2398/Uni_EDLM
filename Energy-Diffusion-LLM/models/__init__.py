from . import dit
from . import ema
from . import autoregressive

# Make dimamba import optional (requires causal_conv1d which may have compatibility issues)
try:
    from . import dimamba
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import dimamba: {e}. DiMamba backbone will not be available.")
    dimamba = None
