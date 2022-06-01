#!/usr/bin/env python3

import argparse
import concurrent.futures
import json
import numpy as np
import os
from pathlib import Path
import shutil
import subprocess

#  config_filename = "ss_pg_tpu_singlerouter.json"
#############################
# Program argumentS
ap = argparse.ArgumentParser()
ap.add_argument('--graph_dir', type=str, default='graphs',
                help='The output directory for generated graphs')
ap.add_argument('--paragraph_core_dir', type=str,
                help='Path to paragraph-core copy')
ap.add_argument('--paragraph_creator_dir', type=str,
                help='Path to paragraph-creator copy')
ap.add_argument('--hlo_examples_dir', type=str,
                help='Path to hlo-examples copy')
ap.add_argument('--hlo_bridge_dir', type=str,
                help='Path to hlo-bridge copy')
ap.add_argument('--translation_config_dir', type=str,
                help='Folder with graph translation json configs')
ap.add_argument('--simulator', choices=['simple', 'supersim'], type=str,
                help='Simulator to use for experiments')
ap.add_argument('--simulation_config_dir', nargs='?', type=str,
                help='Path to configuration file for simulator')
ap.add_argument('--simulator_location', type=str,
                help='Path to simulator binary')
ap.add_argument('--supersim_config', type=str,
                help='Configuration file name for SuperSim simulator')
ap.add_argument('--result_dir', type=str, default='graphs',
                help='The output directory for generated graphs')
ap.add_argument('--num_workers', type=int, default=14,
                help='Number of workers that run experiments')
ap.add_argument('--experiment_type', type=str, default='ring',
                choices=['ring', 'grid-ring', 'torus', 'mesh', 'singlerouter'],
                help='Type of experiments to run')
ap.add_argument('--use-trace', dest='use_trace', action='store_true')
ap.add_argument('--no-use-trace', dest='use_trace', action='store_false')
ap.set_defaults(feature=False)
args = ap.parse_args()
print("Start experiments with parameters:\n", args)
graph_dir = args.graph_dir
paragraph_core = args.paragraph_core
paragraph_creator = args.paragraph_creator
hlo_examples = args.hlo_examples
hlo_bridge = args.hlo_bridge
translation_config_dir = args.translation_config_dir
sim_type = args.simulator
max_workers = args.num_workers
#  sim_config = args.simulator_config
sim = args.simulator_location
config_filename = args.supersim_congig
exp_type = args.experiment_type
result_dir = args.result_dir
use_trace = args.use_trace
#############################
# Static tables
flops_table = {
    'TPUv2': 46/2,
    'TPUv3': 123/2,
    'v100': 125,
    'a100': 312,
}

mem_bw_table = {
    'TPUv2': 700 / 2,
    'TPUv3': 900 / 2,
    'v100': 900,
    'a100': 2039,
}

net_link_bw_table = {
    'TPUv2': 496,
    'TPUv3': 656,
    'v100': 200,
    'a100': 200,
}

# Both TPUv2 and TPUv3 actually have 4 links for 2D Torus
# NVidia a100 and v100 have multiple link to connect with NVSwitch in a
# single-router topology
net_num_links = {
    'TPUv2': 1,
    'TPUv3': 1,
    'v100': 6,
    'a100': 12,
}

app_trace_rel_path = {
    'bert': 'tensorflow/mlperf_training_v0.7'
    '/hlo_after_optimizations_bert_32_tpuv3_stats.csv',
    'dlrm': 'tensorflow/mlperf_training_v0.7'
    '/hlo_after_optimizations_dlrm_32_tpuv3_stats.csv',
    'maskrcnn': 'tensorflow/mlperf_training_v0.7'
    '/hlo_after_optimizations_maskrcnn_32_tpuv3_stats.csv',
    'nmt': 'tensorflow/mlperf_training_v0.7'
    '/hlo_after_optimizations_nmt_32_tpuv3_stats.csv',
    'resnet50': 'tensorflow/mlperf_training_v0.7'
    '/hlo_after_optimizations_resnet_32_tpuv3_stats.csv',
    'ssd': 'tensorflow/mlperf_training_v0.7'
    '/hlo_after_optimizations_ssd_32_tpuv3_stats.csv',
    'transformer': 'tensorflow/mlperf_training_v0.7'
    '/hlo_after_optimizations_transformer_32_tpuv3_stats.csv',
}

app_rel_path = {
    'bert': 'tensorflow/mlperf_training_v0.7'
    '/hlo_after_optimizations_bert_32_tpuv3.txt',
    'dlrm': 'tensorflow/mlperf_training_v0.7'
    '/hlo_after_optimizations_dlrm_32_tpuv3.txt',
    'maskrcnn': 'tensorflow/mlperf_training_v0.7'
    '/hlo_after_optimizations_maskrcnn_32_tpuv3.txt',
    'nmt': 'tensorflow/mlperf_training_v0.7'
    '/hlo_after_optimizations_nmt_32_tpuv3.txt',
    'resnet50': 'tensorflow/mlperf_training_v0.7'
    '/hlo_after_optimizations_resnet_32_tpuv3.txt',
    'ssd': 'tensorflow/mlperf_training_v0.7'
    '/hlo_after_optimizations_ssd_32_tpuv3.txt',
    'transformer': 'tensorflow/mlperf_training_v0.7'
    '/hlo_after_optimizations_transformer_32_tpuv3.txt',
}


#############################
# Helper functions
def get_app_graph_name(app, num_proc, proc_type, use_trace):
    name = app + '_' + str(num_proc) + '_' + str(proc_type)
    if use_trace:
        name += '_traced'
    return name


def get_app_graph_cmd(hlo_bridge, hlo_examples, terget_dir, app,
                      num_proc, proc_type, use_trace):
    name = get_app_graph_name(app, num_proc, proc_type, use_trace)
    cmd = [
        os.path.join(hlo_bridge, 'bazel-bin/bridge/hlo_bridge'),
        '--hlo_module',
        os.path.join(hlo_examples, app_rel_path[app]),
        '--output_graph',
        os.path.join(graph_dir, name, 'graph.textproto'),
        '--num_replicas',
        '64',
        '--estimate_gibps',
        str(mem_bw_table[proc_type]),
        '--estimate_tflops',
        str(flops_table[proc_type])
    ]
    if use_trace:
        cmd.extend([
            '--profiled_data',
            '--enforce_postorder',
            '--trim_graph',
            '--time_from_trace',
            '--noloop_counters_from_trace',
            '--profiled_data_file',
            os.path.join(hlo_examples, app_trace_rel_path[app])
        ])
    else:
        cmd.extend([
            '--profiled_data',
            '--enforce_postorder',
            '--trim_graph',
            '--instructions_from_trace',
            '--noloop_counters_from_trace',
            '--notime_from_trace',
            '--profiled_data_file',
            os.path.join(hlo_examples, app_trace_rel_path[app])
        ])
    return cmd


def generate_app_graphs(hlo_bridge, hlo_examples, graph_dir,
                        app, num_proc, proc_type, use_trace):
    composite_graph = get_app_graph_name(app, num_proc, proc_type, use_trace)
    composite_graph_dir = os.path.join(graph_dir, composite_graph)
    cmd = get_app_graph_cmd(hlo_bridge, hlo_examples, graph_dir,
                            app, num_proc, proc_type, use_trace)
    print("Creating composite  graph", composite_graph)
    make_composite_graph(composite_graph_dir, cmd)
    if num_proc != 64:
        print("Extending graph", composite_graph, "to", num_proc, "processors")
        extend_composite_graph(composite_graph_dir, num_proc)


def get_ar_name(compute_delay, ar_size, num_proc, overlap):
    name = "allreduce"
    overlap_suffix = ''
    if overlap:
        overlap_suffix = '_overlapped'
    name += "_{0}p_{1}s_{2:.0e}B{3}".format(
        num_proc,
        compute_delay,
        ar_size,
        overlap_suffix
    )
    return name


def get_ar_cmd(paragraph_creator, graph_dir,
               compute_delay, ar_size, num_proc, overlap):
    name = get_ar_name(compute_delay, ar_size, num_proc, overlap)
    cmd = [
        os.path.join(paragraph_creator, 'bazel-bin/allreduce'),
        '--output_graph',
        os.path.join(graph_dir, name, 'graph.textproto'),
        '--num_proc',
        str(num_proc),
        '--reduction_size',
        str(ar_size),
        '--format_size',
        '2',
        '--time_delay',
        str(compute_delay)
    ]
    if overlap:
        cmd.append('--is_overlapped')
    return cmd


def generate_ar_graphs(paragraph_creator, graph_dir,
                       compute_delay, ar_size, num_proc, overlap):
    composite_graph = get_ar_name(compute_delay, ar_size, num_proc, overlap)
    composite_graph_dir = os.path.join(graph_dir, composite_graph)
    cmd = get_ar_cmd(paragraph_creator, graph_dir,
                     compute_delay, ar_size, num_proc, overlap)
    print("Creating composite  graph", composite_graph)
    make_composite_graph(composite_graph_dir, cmd)


def composite_graph_exists(composite_graph_dir):
    folder_exists = os.path.exists(composite_graph_dir)
    graph_exists = os.path.exists(os.path.join(composite_graph_dir,
                                               'graph.textproto'))
    return folder_exists and graph_exists


def make_composite_graph(composite_graph_dir, cmd):
    if not composite_graph_exists(composite_graph_dir):
        shutil.rmtree(composite_graph_dir, ignore_errors=True)
        Path(composite_graph_dir).mkdir(parents=True, exist_ok=True)
        result = subprocess.run(' '.join(cmd), shell=True, check=False,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        log_path = os.path.join(composite_graph_dir, 'creation.log')
        with open(log_path, 'wb') as f:
            f.write(result.stdout)


def extend_composite_graph(composite_graph_dir, num_proc):
    orig_path = os.path.join(composite_graph_dir, 'graph.textproto')
    tmp_path = os.path.join(composite_graph_dir, 'graph_.textproto')
    shutil.move(orig_path, tmp_path)
    cmd = [
        os.path.join(
            paragraph_core,
            'bazel-bin/paragraph/graph/graph_data_parallel_extender'),
        '--input_graph',
        tmp_path,
        '--output_graph',
        orig_path,
        '--num_replicas',
        str(num_proc)
    ]
    result = subprocess.run(' '.join(cmd), shell=True, check=False,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path = os.path.join(composite_graph_dir, 'extension.log')
    with open(log_path, 'wb') as f:
        f.write(result.stdout)
    os.remove(tmp_path)


def get_translation_name(allreduce_algo, protocol_algo, topo_config):
    return allreduce_algo + '_' + protocol_algo + '_' + topo_config.split('b')[0]


def get_translated_dir(composite_graph_dir,
                       allreduce_algo,
                       protocol_algo,
                       topo_config):
    return os.path.join(composite_graph_dir,
                        get_translation_name(
                            allreduce_algo, protocol_algo, topo_config))


def translated_graphs_exist(composite_graph_dir, translation_name, num_proc):
    folder_exists = os.path.exists(os.path.join(composite_graph_dir,
                                                translation_name))
    fully_translated = True
    for n in range(num_proc):
        if not os.path.exists(os.path.join(composite_graph_dir,
                                           translation_name,
                                           "graph.{}.textproto".format(n))):
            fully_translated = False
    return folder_exists and fully_translated


def prepare_translation_conf_json(translation_config_dir,
                                  translation_name,
                                  allreduce_algo,
                                  protocol_algo,
                                  topo_config):
    translation_json_path = os.path.join(
        translation_config_dir, 'translation_' + translation_name + '.json')
    topo, dims, _, num_proc, conc, ports = topo_config_parse(topo_config)
    if not os.path.exists(translation_json_path):
        if allreduce_algo.startswith('mesh'):
            algo = 'mesh-' + str(len(dims)) + 'd'
            _, local_exchange_type = allreduce_algo.split('-')
            int_local_exchange = False
            if local_exchange_type == 'ile':
                int_local_exchange = True
            json_config = {
                "collective": {
                    "all-reduce": {
                        "algorithm": algo,
                        "concentration": conc,
                        "dimension_widths": dims,
                        "integrated_local_exchange": int_local_exchange,
                    }
                },
                "protocol": {
                    "sendrecv": {
                        "algorithm": protocol_algo
                    },
                    "send": {
                        "algorithm": protocol_algo
                    },
                    "recv": {
                        "algorithm": protocol_algo
                    },
                }
            }
        elif allreduce_algo.startswith('torus'):
            algo = 'torus-' + str(len(dims)) + 'd'
            _, local_exchange_type = allreduce_algo.split('-')
            int_local_exchange = False
            if local_exchange_type == 'ile':
                int_local_exchange = True
            json_config = {
                "collective": {
                    "all-reduce": {
                        "algorithm": algo,
                        "concentration": conc,
                        "dimension_widths": dims,
                        "integrated_local_exchange": int_local_exchange,
                    }
                },
                "protocol": {
                    "sendrecv": {
                        "algorithm": protocol_algo
                    },
                }
            }
        elif allreduce_algo == 'ring-2d-grid':
            json_config = {
                "collective": {
                    "all-reduce": {
                        "algorithm": allreduce_algo,
                        "concentration": conc,
                        "dimension_widths": dims,
                    }
                },
                "protocol": {
                    "sendrecv": {
                        "algorithm": protocol_algo
                    },
                }
            }
        else:
            json_config = {
                "collective": {
                    "all-reduce": {
                        "algorithm": allreduce_algo,
                    }
                },
                "protocol": {
                    "sendrecv": {
                        "algorithm": protocol_algo
                    },
                }
            }
        with open(translation_json_path, 'w') as f:
            json.dump(json_config, f, indent=4)
    return translation_json_path


def make_translated_graphs(paragraph_core, composite_graph_dir,
                           translation_config_dir, num_proc,
                           allreduce_algo, protocol_algo, topo_config):
    translation_name = get_translation_name(allreduce_algo,
                                            protocol_algo,
                                            topo_config)
    if not translated_graphs_exist(composite_graph_dir,
                                   translation_name, num_proc):
        print("Previous translation not found or incomplete, start translating")
        translation_path = os.path.join(composite_graph_dir, translation_name)
        shutil.rmtree(translation_path, ignore_errors=True)
        Path(translation_path).mkdir(parents=True, exist_ok=True)
        json_config = prepare_translation_conf_json(translation_config_dir,
                                                    translation_name,
                                                    allreduce_algo,
                                                    protocol_algo,
                                                    topo_config)
        cmd = [
            os.path.join(paragraph_core,
                         'bazel-bin/paragraph/translation/graph_translator'),
            '--input_graph',
            os.path.join(composite_graph_dir, 'graph.textproto'),
            '--output_dir',
            translation_path,
            '--translation_config',
            json_config
        ]
        result = subprocess.run(' '.join(cmd), shell=True, check=False,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        log_path = os.path.join(translation_path, 'translation.log')
        with open(log_path, 'wb') as f:
            f.write(result.stdout)


def translate_app_graphs(paragraph_core, translation_config_dir, graph_dir,
                         app, proc_type,
                         allreduce_algo, protocol_algo, topo_config, use_trace):
    topo, dims, _, num_proc, conc, ports = topo_config_parse(topo_config)
    composite_graph = get_app_graph_name(app, num_proc, proc_type, use_trace)
    composite_graph_dir = os.path.join(graph_dir, composite_graph)
    print("Translating graph", composite_graph)
    make_translated_graphs(paragraph_core, composite_graph_dir,
                           translation_config_dir, num_proc,
                           allreduce_algo, protocol_algo, topo_config)


def translate_ar_graphs(paragraph_core, translation_config_dir, graph_dir,
                        compute_delay, ar_size, overlap,
                        allreduce_algo, protocol_algo, topo_config):
    topo, dims, _, num_proc, conc, ports = topo_config_parse(topo_config)
    composite_graph = get_ar_name(compute_delay, ar_size, num_proc, overlap)
    composite_graph_dir = os.path.join(graph_dir, composite_graph)
    print("Translating graph", composite_graph)
    make_translated_graphs(paragraph_core, composite_graph_dir,
                           translation_config_dir, num_proc,
                           allreduce_algo, protocol_algo, topo_config)


def topo_config_parse(topo_config):
    topo, nodes = topo_config.split('-')
    proc, loc = nodes.split('c')
    dims = [int(i) for i in proc.split('x')]
    dim_weights = '[' + ','.join(['1'] * len(proc.split('x'))) + ']'
    conc = int(loc.split('b')[0])
    num_proc = np.prod([int(i) for i in proc.split('x')]) * conc
    if len(loc.split('b')) > 1:
        ports = int(loc.split('b')[1])
    else:
        ports = 1
    if topo == 'ring':
        topo = 'torus'
    if topo == 'singlerouter':
        topo = 'single_router'
    return topo, dims, dim_weights, num_proc, conc, ports

#############################
# Program variables menu
# There should be a more elegant way to explore them, e.g. using json config for
# the script, but commenting out these parameters manually suffice.
apps_list = [
    #  'allreduce',
    #  'resnet50',
    #  'transformer',
    #  'dlrm',
    'bert',
    #  'maskrcnn',
    #  'nmt',
    #  'ssd',
]
proc_types_list = [
    #  'TPUv2',
    'TPUv3',
    #  'v100',
    #  'a100',
]
compute_delay_list = [
    1e-6,
]
ar_size_list = [
    1e4,
    2e4,
    5e4,
    1e5,
    2e5,
    5e5,
    1e6,
    2e6,
    5e6,
    1e7,
    #  2e7,
    #  5e7,
    #  1e8,
    #  2e8,
    #  5e8,
    #  1e9,
]
num_proc_list = [
    32,
    #  64,
    #  256,
]
overlap = [
    #  True,
    False,
]
configs_list = [
    #  'torus-4x4c2b1',  # HW study,
    #  'torus-4x4c2b2',  # HW study, NN study
    #  'torus-4x4c2b4',  # HW study, Algo study, NN study, Scalability Study
    #  'torus-4x4c2b6',
    #  'torus-4x4c2b8',
    #  'torus-8x4c1b1',  # HW study
    #  'torus-8x4c1b2',  # HW study
    #  'torus-8x4c1b4',  # HW study
    #  'torus-8x4c1b6',
    #  'singlerouter-32c32b2',
    #  'singlerouter-32c32b1',
    #  'ring-32c1b1',
    #  'ring-32c1b2',
    #  'ring-16c2b1',
    #  'ring-16c2b2',
    #  'ring-16c2b4',
    #  'mesh-4x4c2b1',
    #  'mesh-4x4c2b2',
    #  'mesh-4x4c2b4',
    #  'mesh-8x4c1b1',
    #  'mesh-8x4c1b2',
    #  'torus-2x2c2b2',  # NN study
    #  'torus-2x2c2b4',  # NN study, Scalability  study
    #  'torus-8x4c2b2',  # NN study
    #  'torus-8x4c2b4',  # NN study
    #  'torus-8x8c2b1',
    #  'torus-8x8c2b2',  # NN study
    'torus-8x8c2b4',  # NN study, Scalability study
    #  'torus-8x8c2b6',
    #  'torus-16x16c2b1',
    #  'torus-16x16c2b2',
    #  'torus-16x16c2b4',
    #  'torus-16x16c2b6',
    #  'torus-32x32c2b2',
]
allreduce_algos_list = [
    #  'unidir-ring',
    #  'bidir-ring',
    #  'mesh-ile',
    #  'torus-ile',
    'mesh-sle',
    #  'torus-sle',
    #  'ring-2d-grid',
]
protocol_algos_list = [
    'push',
]

#############################
# Iterate over each variable to make graph creation tasks
creation_task_list = []
for a in apps_list:
    for c in configs_list:
        if a == 'allreduce':
            for d in compute_delay_list:
                for s in ar_size_list:
                    for o in overlap:
                        topo, dims, _, n, conc, ports = topo_config_parse(c)
                        creation_task_list.append([generate_ar_graphs,
                                                   paragraph_creator, graph_dir,
                                                   d, s, n, o])
        else:
            for typ in proc_types_list:
                topo, dims, _, n, conc, ports = topo_config_parse(c)
                creation_task_list.append([generate_app_graphs,
                                           hlo_bridge, hlo_examples,
                                           graph_dir, a, n, typ, use_trace])

#############################
# Run graph creation tasks concurrently
print("Start creating graphs...")
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_task = {
        executor.submit(*t): t for t in creation_task_list
    }
    for future in concurrent.futures.as_completed(future_to_task):
        print("Completed:", future_to_task[future])
print("Finished creating graphs.")

#############################
# Iterate over each variable to make graph translation tasks
translation_task_list = []
for a in apps_list:
    for c in configs_list:
        for ar in allreduce_algos_list:
            for p in protocol_algos_list:
                if a == 'allreduce':
                    for d in compute_delay_list:
                        for s in ar_size_list:
                            for o in overlap:
                                translation_task_list.append([
                                    translate_ar_graphs,
                                    paragraph_core, translation_config_dir,
                                    graph_dir, d, s, o, ar, p, c])
                else:
                    for typ in proc_types_list:
                        translation_task_list.append([
                            translate_app_graphs,
                            paragraph_core, translation_config_dir,
                            graph_dir, a, typ, ar, p, c, use_trace])

#############################
# Run graph translation tasks concurrently
print("Start translating graphs...")
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    future_to_task = {
        executor.submit(*t): t for t in translation_task_list
    }
    for future in concurrent.futures.as_completed(future_to_task):
        print("Completed:", future_to_task[future])
print("Finished translating graphs.")

#############################
# Simulation parameters and helper functions

def get_simplesim_command(sim, translated_graph_dir, log_dir,  proc_type):
    log_path = os.path.join(log_dir, 'paragraph.log')
    cmd = ' '.join(['{0}',
                    ' --input_graph {1}',
                    ' --log_file {2}',
                    ' --estimate_tflops {3}',
                    ' --estimate_mem_gibps {4}',
                    ' --estimate_net_gbit {5}']).format(
            sim,
            os.path.join(translated_graph_dir, 'graph.0.textproto'),
            log_path,
            flops_table[proc_type],
            mem_bw_table[proc_type],
            net_link_bw_table[proc_type] * net_num_links[proc_type])
    return cmd


def get_ss_command(sim, sim_config, topo_config,
                   translated_graph_dir, log_dir, proc_type):
    log_path = os.path.join(log_dir, 'paragraph_log.%d.csv')
    graph_path = os.path.join(translated_graph_dir, 'graph.%d.textproto')
    info_csv = os.path.join(log_dir, 'info.csv')
    messages_mpf = os.path.join(log_dir, 'messages.mpf')
    rates_csv = os.path.join(log_dir, 'rates.csv')
    channels_csv = os.path.join(log_dir, 'channels.csv')
    cmd = ('{0}'
           ' {1}'
           ' /workload/applications/0/graph_terminal/graph_file=string={2}'
           ' /workload/applications/0/graph_terminal/log_file=string={3}'
           ' /simulator/info_log/file=string={4} '
           ' /workload/message_log/file=string={5} '
           ' /workload/applications/0/rate_log/file=string={6} '
           ' /network/channel_log/file=string={7}'
           .format(sim, sim_config, graph_path, log_path,
                   info_csv, messages_mpf, rates_csv, channels_csv))
    topo, dims, dim_weights, num_proc, conc, ports = topo_config_parse(
        topo_config)
    dims = '[' + ','.join([str(i) for i in dims]) + ']'
    cmd += (' /network/topology=string={0}'
            ' /network/dimension_widths=uint={1}'
            ' /network/dimension_weights=uint={2}'
            ' /network/concentration=uint={3}'
            ' /network/interface_ports=uint={4}'
            .format(topo, dims, dim_weights, conc * ports, ports))
    return cmd


def get_log_dir(sim_type, result_dir, composite_graph,
                allreduce_algo, protocol_algo, topo_config):
    translation_name = get_translation_name(allreduce_algo,
                                            protocol_algo,
                                            topo_config)
    topo_config_parts = topo_config.split('b')
    if len(topo_config_parts) > 1:
        translation_name += 'b' + topo_config_parts[1]
    log_dir = os.path.join(result_dir,
                           sim_type,
                           composite_graph,
                           translation_name)
    return log_dir


def run_sim(sim, sim_config, sim_type, topo_config,
            graph_dir, result_dir, composite_graph,
            proc_typ, allreduce_algo, protocol_algo):
    composite_graph_dir = os.path.join(graph_dir, composite_graph)
    translated_dir = get_translated_dir(composite_graph_dir,
                                        allreduce_algo,
                                        protocol_algo,
                                        topo_config)
    log_dir = get_log_dir(sim_type, result_dir, composite_graph,
                          allreduce_algo, protocol_algo, topo_config)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    cmd = ''
    if sim_type == 'simple':
        cmd = get_simplesim_command(sim, translated_dir, log_dir, proc_typ)
    elif sim_type == 'supersim':
        cmd = get_ss_command(sim, sim_config, topo_config,
                             translated_dir, log_dir, proc_typ)
    else:
        raise ValueError('Wrong simulator type:', sim_type)
    print(cmd)
    result = subprocess.run(cmd, shell=True, check=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    log_path = os.path.join(log_dir, 'simulation.log')
    with open(log_path, 'wb') as f:
        f.write(result.stdout)


#############################
# Iterate over each variable to make graph translation tasks
sim_task_list = []
for a in apps_list:
    for c in configs_list:
        for ar in allreduce_algos_list:
            for p in protocol_algos_list:
                if a == 'allreduce':
                    for d in compute_delay_list:
                        for s in ar_size_list:
                            for o in overlap:
                                topo, dims, _, n, _2, _3 = topo_config_parse(c)
                                composite_graph = get_ar_name(d, s, n, o)
                                sim_config = os.path.join(
                                    args.simulation_config,
                                    config_filename)
                                sim_task_list.append([
                                    run_sim,
                                    sim, sim_config, sim_type, c,
                                    graph_dir, result_dir, composite_graph,
                                    'TPUv3', ar, p])
                else:
                    for typ in proc_types_list:
                        topo, dims, _, n, _2, _3 = topo_config_parse(c)
                        composite_graph = get_app_graph_name(
                            a, n, typ, use_trace)
                        sim_config = os.path.join(
                            args.simulation_config,
                            config_filename)
                        sim_task_list.append([
                            run_sim,
                            sim, sim_config, sim_type, c,
                            graph_dir, result_dir, composite_graph,
                            typ, ar, p])

#############################
# Run graph translation tasks concurrently
print("Start simulation jobs...")
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    future_to_task = {
        executor.submit(*t): t for t in sim_task_list
    }
    for future in concurrent.futures.as_completed(future_to_task):
        print("Completed:", future_to_task[future])
print("Finished simulations")
