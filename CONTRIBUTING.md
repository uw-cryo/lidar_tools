# Contributing Guide

Contributions are welcome!

We recommend use [pixi](https://pixi.sh) for environment management. The "dev" environment installs additional packages such as `pytest` for developing new functions.

```bash
git clone https://github.com/uw-cryo/lidar_tools.git
cd lidar_tools
git checkout -b newfeature
pixi shell -e dev # type `exit` to deactivate
pixi run test
```

Before opening a pull request, run the regression gate and read the review
process in [docs/pr_review.md](docs/pr_review.md):

```bash
pixi run -e dev test
pixi run -e dev check-regressions
```
