"""
Policy implementations for bimanual manipulation.

Available policies:
- PrivilegedPolicy: Handcrafted policy using privileged simulation information
- ACT: Action Chunking Transformer (in policy.act submodule)
"""

# Import MuJoCo-dependent policies conditionally
try:
    from .privileged_policy import PrivilegedPolicy
    __all__ = ['PrivilegedPolicy']
except ImportError:
    __all__ = []

# ACT policy module (doesn't require MuJoCo for training)
from . import act
__all__ += ['act']

# For backward compatibility, also expose at top level
from .act import ACTPolicy, build_act_policy
__all__ += ['ACTPolicy', 'build_act_policy']
