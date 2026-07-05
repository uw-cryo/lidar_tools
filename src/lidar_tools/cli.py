"""
A CLI following https://packaging.python.org/en/latest/guides/creating-command-line-tools/
"""

import cyclopts

from .pdal_pipeline import rasterize
from .survey import survey


app = cyclopts.App()
app.command()(rasterize)
app.command()(survey)


if __name__ == "__main__":
    app()
