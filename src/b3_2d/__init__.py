"""b3_2d: 2D meshing for b3m using cgfoil."""

from .core.mesh import process_vtp_multi_section
from .statesman.b3_2d_step import B32dStep

__version__ = "0.1.0"
__all__ = ["process_vtp_multi_section", "B32dStep"]
