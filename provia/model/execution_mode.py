"""Joint execution-mode and procedure-state filtering.

The latent execution mode distinguishes ordinary progress, error, and
recovery.  Error branches retain the accepted procedure state; normal and
recovery branches accept the observed event.  All branches remain in one
normalized posterior, without score thresholding or a post-hoc state machine.
"""

from __future__ import annotations





EXECUTION_MODES = ("normal", "error", "recovery")
NORMAL, ERROR, RECOVERY = range(len(EXECUTION_MODES))