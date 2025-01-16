# Contributing Guide

Contributions are welcome!

We recommend use [pixi](https://pixi.sh) for environment management. The "dev" environment installs additional packages such as `pytest` for developing new functions.

```bash
git clone https://github.com/uw-cryo/grid_pc.git
cd coincident
git checkout -b newfeature
pixi shell -e dev # type `exit` to deactivate
pixi run test
```
