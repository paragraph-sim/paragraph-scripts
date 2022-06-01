# paragraph-scripts
This repo contains examples of configuration and workflow arrangement script that could be used with [ParaGraph](https://github.com/paragraph-sim/paragraph-core).

## install_paragraph.sh
The script installs [paragraph-core](https://github.com/paragraph-sim/paragraph-core), [paragraph-creator](https://github.com/paragraph-sim/paragraph-creator), [hlo-bridge](https://github.com/paragraph-sim/hlo-bridge), and [hlo-examples](https://github.com/paragraph-sim/hlo-examples) to the folder one level above the script.
For example, if you have this repository under `~/dev/paragraph-sim/paragraph-scripts`, it will install all the tools under `~/dev/paragraph-sim`, so it will contain:
```
~/dev/paragraph-sim/hlo-bridge
~/dev/paragraph-sim/hlo-examples
~/dev/paragraph-sim/paragraph-core
~/dev/paragraph-sim/paragraph-creator
~/dev/paragraph-sim/paragraph-scripts
```
## paragraph_paper_result_generation.py
This script can be used to reproduce experiments from ParaGraph paper sections 4.1 and 4.2.
See [reproducibility readme](https://github.com/paragraph-sim/paragraph-scripts/blob/main/icpp_reproducibility.md).
To see helpline with information about flags and how to use it, type
```
./paragraph_paper_result_generation.py --help
``` 
## supersim_configs
This folder contains examples of supersim configs to use with ParaGraph. In particular, these configs can be used for [ParaGraph paper results reproducibility](https://github.com/paragraph-sim/paragraph-scripts/blob/main/icpp_reproducibility.md).
## translation_configs
This folder contains several examples of translation configs for various ring and torus based topologies using different `all-reduce` algorithms.
