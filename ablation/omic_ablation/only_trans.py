"""Train CoMET using transcriptomic inputs only.

Retained inputs: RNA latent embeddings, DavidLiu MHC-II, and IMvigor210
Lund transcriptomic subtypes.
"""

try:
    from .common import run
except ImportError:
    from common import run


if __name__ == "__main__":
    run("transcriptomic")
