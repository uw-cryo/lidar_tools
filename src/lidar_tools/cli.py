"""
A CLI following https://packaging.python.org/en/latest/guides/creating-command-line-tools/
"""

import cyclopts

from .catalog import fetch_reports, search
from .driver import rasterize
from .merge import merge
from .preview import preview
from .report_metrics import report_metrics
from .staging import prepare

# Help groups follow the run order of the multi-project workflow (search ->
# prepare -> rasterize -> merge -> inspect), not the alphabetical default,
# which listed the first step last.
_discovery = cyclopts.Group("Discovery (what covers this AOI?)", sort_key=0)
_processing = cyclopts.Group("Processing (point clouds -> rasters)", sort_key=1)
_inspection = cyclopts.Group("Inspection (QA figures and vendor reports)", sort_key=2)

app = cyclopts.App()
app.command(group=_discovery, sort_key=0)(search)
app.command(group=_discovery, sort_key=1)(prepare)
app.command(group=_processing, sort_key=0)(rasterize)
app.command(group=_processing, sort_key=1)(merge)
app.command(group=_inspection, sort_key=0)(preview)
app.command(group=_inspection, sort_key=1)(fetch_reports)
app.command(group=_inspection, sort_key=2)(report_metrics)


if __name__ == "__main__":
    app()
