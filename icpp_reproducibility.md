# ICPP reproducibility
This readme describes steps necessary to reproduce results from ICPP ParaGraph paper. For convenience we released a [paragraph_paper_result_generation.py](https://github.com/paragraph-sim/paragraph-scripts/blob/main/paragraph_paper_result_generation.py) script that can be used to streamline the task.
To see helpline with information about flags and how to use it, type
``
./paragraph_paper_result_generation.py --help
`` 
## Section 4.1 and 4.2
The exact configuration of the experiments is set in the `Program variables menu` part of the script.
* `apps_list` sets neural network applications to model from the [hlo-examples](https://github.com/paragraph-sim/hlo-examples) repo
* `proc_type_list` sets processor configuration (TPUv3 used for all experiments)
  * `ar_size_list` sets `all-reduce` size for experiments with traffic generator
  * `num_proc_list` sets the size of the system
  * `configs_list` updates SuperSim config and sets translation config for exact system architecture to model
  * `allreduce_algos_list` sets an `all-reduce` algorithm (translation)

## Section 4.3
  To reproduce experiments in section 4.3 we need to install repo [mkpg]().
  Given that other ParaGraph tools are installed under `~/dev/paragraph-sim`, you can type
  ``
  git clone git@github.com:nicmcd/mkpg.git ~/dev/paragraph-sim/mkpg
  bazel build -c opt mkpg_stencil
  bazel build -c opt mkss_stencil
  ``
  To run experiments, we can use following commands:
  ``
  mkdir graphs/pg_27p_stencil
  ./bazel-bin/mkpg_stencil -v 1 -n pg_27p_stencil -- 4 4 4 2 0 1e-6 800 80 8 8 1e-8 graphs/pg_27p_stencil/graph.textproto
  ../paragraph-core/bazel-bin/paragraph/translation/graph_translator --input_graph graphs/pg_27p_stencil/graph.textproto --output_dir graphs/pg_27p_stencil --translation_config translation_configs/translation_bidir_ring.json
  ../supersim/bazel-bin/supersim supersim_configs/pg_stencil.json
  ./bazel-bin/mkss_stencil -v 1 -- 4 4 4 800 80 8 8 graphs/halo_exchange_64.csv
  ../supersim/bazel-bin/supersim supersim_configs/ss_stencil.json
  ``
## Section 4.4
  Results in this section cannot be reproduced as we used proprietary simulator developed at NVidia and configured it using settings that cannot be publicly released.`
