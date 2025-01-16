"""
A CLI following https://packaging.python.org/en/latest/guides/creating-command-line-tools/
"""

import typer

from .pdal_pipeline import create_dsm


app = typer.Typer()
app.command()(create_dsm)


if __name__ == "__main__":
    app()
