"""
A CLI following https://packaging.python.org/en/latest/guides/creating-command-line-tools/
"""

import cyclopts

from .driver import rasterize_projects
from .merge import merge
from .pdal_pipeline import rasterize
from .preview import preview
from .survey import fetch_reports, survey


app = cyclopts.App()
app.command()(rasterize)
app.command()(rasterize_projects)
app.command()(survey)
app.command()(preview)
app.command()(merge)
app.command()(fetch_reports)


if __name__ == "__main__":
    app()
