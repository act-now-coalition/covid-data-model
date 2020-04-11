#!/usr/bin/env python
"""
Entry point for covid-data-model CLI.

"""
import logging
import click
from cli import run_data
from cli import run_model
from cli import run_dod_dataset
from cli import api


@click.group()
def entry_point():
    """Entry point for covid-data-model CLI."""
    pass


entry_point.add_command(run_data.main)
entry_point.add_command(run_model.main)
entry_point.add_command(run_dod_dataset.deploy_dod_projections)
entry_point.add_command(api.main)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    entry_point()
