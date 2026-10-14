"""Train CoMET using genomic inputs only."""

try:
    from .common import run
except ImportError:
    from common import run


if __name__ == "__main__":
    run("genomic")
