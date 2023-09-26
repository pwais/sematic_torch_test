# Copyright 2023 Maintainers of sematic_torch_test

"""main.py defines the Sematic-integrated job code that will run on the K8s
cluster (as well as locally in our dev container).  This program is designed
to run in the Dockerized environment started with:
  ```bash
  ./devtool.py --docker-build --shell
  ```

The substance of the job is in `simple_net.train_test_model()`-- We train a
small 2-layer network on a 2-D XOR problem and report the accuracy.


## Part 1: Test Local Execution

First, try ensuring you can run / train locally with just vanilla pytorch:
  ```bash
  python main.py --run-no-sematic --cpu
  ```
You should see tqdm progress bar output and in the end get 100% accuracy on the
test set after 1,000 epochs of training.


Now check that you get the same using your local GPU:
  ```bash
  python main.py --run-no-sematic
  ```

Second, try ensuring Sematic local execution works:
  ```bash
  sematic run main.py -- --run-smoketest
  ```
You should see "Runner result: yes-i-works" output from the script as well as
runner output logged in the Sematic WebUI.

Now try running the Pytorch pipeline with Sematic (without and with GPU):
  ```bash
  sematic run main.py -- --cpu
  sematic run main.py
  ```
You should see output logged in the Sematic WebUI.  When GPU training is
enabled, you'll see "device info" in the output includes info about your
GPU.


## Part 2: Test Cloud (K8S) Execution

First, ensure that Sematic+K8S are set up properly.  The command below will
also make Sematic `--build` the worker docker image, which will be slow 
the first time but fast (cached) every subsequent invocation. (Note that 
local code changes will invalidate a small part of the cache [small Docker 
layer] and so `sematic run --build` will always be rather fast after the
first time).
  ```bash
  sematic run --build main.py -- --cloud --run-smoketest
  ```
In the Sematic WebUI, you should see both "yes-i-works" in output as well as
some info about the Kubernetes Pods started during the run.  You may also 
manually track progress in the cluster with e.g. `kubectl get pods -A` or
other monitoring dashboards.

Now try running the Pytorch pipeline with Sematic on the K8S cluster.  First
ensure vanilla CPU training works (this will run on *any* K8S node, even
if the node has no GPUs):
  ```bash
  sematic run --build main.py -- --cloud --cpu
  ```
You should see similar output as for `sematic run main.py -- --cpu` except
that in the Sematic WebUI you'll see a `training_hostname` like
'sematic-worker-xxxxx' and there will be some info about K8S pod scheduling.

Now try with a GPU, and tell K8S to allocate a GPU to the job (see
implementation details i.e. Sematic's `KubernetesResourceRequirements`):
  ```bash
  sematic run --build main.py -- --run-cloud-one-gpu
  ```
You should see similar output as for `sematic run main.py` including
"device info" in the output includes info about the cluster GPU used.

You can also try running the same job but have K8S allocate more than one
GPU to the job (simulates e.g. a single-machine multi-GPU data-parallel job):
  ```bash
  sematic run --build main.py -- --run-cloud-n-gpus=2
  ```

Lastly, you can also try running the job on a specific host / K8S node in the
cluster (simulates using `node_selector` to control scheduling; you might
want to use your own custom node labels beyond just 'hostname', e.g. some
nodes might only be for inference):
  ```bash
  sematic run --build main.py -- --run-cloud-on-host=hostname
  ```


## Part 3: Load-test the Cluster!

One of the primary benefits of having a K8S cluster is that you can schedule
a large number of job and let the cluster deal with scheduling and 
orchestrating them.  Furthermore, Sematic can help you visualize complex
workflow DAGs (i.e. a bunch of jobs associated with grid search over one
or more parameters).  Now that we know Sematic and your K8S cluster works,
let's schedule a bunch of jobs and load up the cluster:
  ```bash
  sematic run --build main.py -- --cloud --run-grid-test=20
  ```
Now go to the Sematic WebUI and to the Pipeline page for 
'sematic_pipeline.sematic_gpu_grid_test'.  You should see a Execution Graph
viz of the DAG of your 20 jobs and eventually Output for the aggregated
results.  Additionally you can click on any individual job and you'll see
Sematic tracks the `TrainTestParams` and `TrainTestResults` for all jobs.

Note that for debugging you can also run the above without `--cloud` and all
your code will run locally (and sequentially) and stop at any `breakpoint()`
inserted.

"""

try:
  import sematic
  import torch
except ImportError as e:
  raise ValueError(f"""
      Run this code from the dev container, 
      do ./devtool.py --shell first.  Error: {e}
    """)

from sematic import LocalRunner
from sematic import CloudRunner
from loguru import logger as log

from simple_net import TrainTestParams
from simple_net import train_test_model
from sematic_pipeline import sematic_gpu_grid_test
from sematic_pipeline import sematic_train_once
from sematic_pipeline import sematic_train_one_gpu_node
from sematic_pipeline import sematic_make_train_on_node
from sematic_pipeline import smoketest_pipeline


def run_no_sematic(args):
  train_params = TrainTestParams(use_cuda=not args.cpu)
  log.info(f"Torch-only mode!  Training with params {train_params} ...")
  results = train_test_model(train_params)
  log.info(f"Final results: {results}")


def create_arg_parser():
  import argparse

  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  gconf = parser.add_argument_group('Global Configuration')
  gconf.add_argument('--cloud',
                     default=False,
                     action='store_true',
                     help='Run on the cloud / K8S cluster')
  gconf.add_argument('--cpu',
                     default=False,
                     action='store_true',
                     help='Force Torch CPU execution')
  gconf.add_argument('--custom-tag',
                     default='',
                     type=str,
                     help='Add this custom tag to the Sematic run')

  gaction = parser.add_argument_group('Global Actions')
  gaction.add_argument(
    '--run-no-sematic',
    default=False,
    action='store_true',
    help='Run one local train-test job without any Sematic usage')
  gaction.add_argument(
    '--run-smoketest',
    default=False,
    action='store_true',
    help='Run a simple smoketest Sematic pipeline; useful for debugging')
  gaction.add_argument(
    '--run-cloud-one-gpu',
    default=False,
    action='store_true',
    help='Run in --cloud mode requesting one gpu')
  gaction.add_argument(
    '--run-cloud-n-gpus',
    default=-1,
    type=int,
    help='Run in --cloud mode requesting N gpus [default disabled]')
  gaction.add_argument(
    '--run-cloud-on-host',
    default='',
    type=str,
    help='Run in --cloud mode running on the given hostname [default disabled]')
  gaction.add_argument(
    '--run-grid-test',
    default=-1,
    type=int,
    help='Run a grid test with this many treatments.  To exercise cluster job '
         'queing, use a number of treatments greater than the total number of '
         'cluster GPUs [default disabled]')

  return parser


def main(args=None):
  if not args:
    parser = create_arg_parser()
    args = parser.parse_args()

  if args.run_no_sematic:
    run_no_sematic(args)
    return

  tags = ["sematic_torch_test"]
  if args.custom_tag:
    tags.append(args.custom_tag)

  if args.run_smoketest:
    future = smoketest_pipeline()
  elif args.run_cloud_one_gpu:
    log.info(f"Training on cloud with one GPU")
    args.cloud = True
    train_params = TrainTestParams(use_cuda=True)
    future = sematic_train_one_gpu_node(train_params)
    tags.append("run_one_cloud_gpu")
  elif args.run_cloud_n_gpus > 0:
    log.info(f"Training on cloud with {args.run_cloud_n_gpus} GPUs")
    args.cloud = True
    train_params = TrainTestParams(use_cuda=True)
    future = sematic_train_one_gpu_node(train_params)
    k8s_props = future.props.resource_requirements.kubernetes
    k8s_props.requests['nvidia.com/gpu'] = str(args.run_cloud_n_gpus)
    tags.append("run_cloud_n_gpus")
  elif args.run_cloud_on_host:
    log.info(f"Training on cloud on host {args.run_cloud_on_host}")
    args.cloud = True
    train_params = TrainTestParams(use_cuda=not args.cpu)
    future = sematic_make_train_on_node(args.run_cloud_on_host, train_params)
    tags.append("run_cloud_on_host")
  elif args.run_grid_test > 0:
    future = sematic_gpu_grid_test(args.run_grid_test)
    tags.append("grid_test")
  else:
    train_params = TrainTestParams(use_cuda=not args.cpu)
    future = sematic_train_once(train_params)

  log.info("Setting up Sematic execution ...")
  runner = CloudRunner() if args.cloud else LocalRunner()

  
  if args.cloud:
    tags.append("cloud_runner")

  future.set(
    name="sematic_torch_test Example",
    tags=tags,
  )
  log.info("... running ...")
  result = runner.run(future)
  log.info(f"Runner result: {result}")


if __name__ == '__main__':
  main()
