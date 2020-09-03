"""Contains custom argparse actions & formatters."""

import argparse


class CustomHelpFormatter(argparse.RawDescriptionHelpFormatter,
                          argparse.ArgumentDefaultsHelpFormatter):
    pass
