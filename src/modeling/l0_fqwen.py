# L0 stretched-concrete sampling is architecture-independent.
# Re-export from l0_fllama so modeling_fqwen.py can `from l0_fqwen import ...`
# without code duplication.
from l0_fllama import (
    cdf_stretched_concrete,
    sample_z_from_u,
    deterministic_z_from_log_alpha,
    sample_z_from_log_alpha,
    sample_z_from_log_alpha_old,
    LIMIT_LEFT,
    LIMIT_RIGHT,
    EPS,
    TEMPERATURE,
    FACTOR,
)
