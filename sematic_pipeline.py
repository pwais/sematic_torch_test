# Copyright 2023 Maintainers of sematic_torch_test

# TODO(sematic) fix putting pipeline funcs in main module:
# AttributeError: Unable to find this run's function at __main__.pipeline, did it change location? module '__main__' has no attribute 'pipeline'

from dataclasses import dataclass, field
from typing import List

from loguru import logger as log

from sematic import func
from sematic import KubernetesResourceRequirements
# from sematic import KubernetesToleration
# from sematic import KubernetesTolerationEffect
from sematic import ResourceRequirements

from simple_net import TrainTestParams
from simple_net import TrainTestResults
from simple_net import train_test_model


## Smoke Test Pipeline

@func(standalone=True)
def smoketest_pipeline() -> str:
  return 'yes-i-works'


## GPU Pipelines

@func(standalone=True)
def sematic_train_once(params: TrainTestParams) -> TrainTestResults:
  log.info("Runnning train_test_model() on Sematic")
  return train_test_model(params)


ONE_GPU = ResourceRequirements(
    kubernetes=KubernetesResourceRequirements(
        requests={
          "nvidia.com/gpu": "1",
        },
    )
)

@func(standalone=True, resource_requirements=ONE_GPU)
def sematic_train_one_gpu_node(params: TrainTestParams) -> TrainTestResults:
  log.info("Runnning sematic_train_one_gpu_node() on Sematic")
  return train_test_model(params)


## Host-pinned Pipeline

HOST_REQUIREMENT = ResourceRequirements(
    kubernetes=KubernetesResourceRequirements(
        node_selector={"kubernetes.io/hostname": ""},
    )
)
@func(standalone=True, resource_requirements=HOST_REQUIREMENT)
def train_on_host(params: TrainTestParams) -> TrainTestResults:
  log.info("Runnning train_on_host() on Sematic")
  return train_test_model(params)

def sematic_make_train_on_node(hostname: str, params: TrainTestParams) -> TrainTestResults:

  future = train_on_host(params)
  k8s_props = future.props.resource_requirements.kubernetes
  k8s_props.node_selector['kubernetes.io/hostname'] = hostname

  return future


## Grid Test Pipeline

@dataclass
class GridTestResults:
  mean_accuracy : float = 0.
  all_accuracies : List[float] = field(default_factory=list)
  num_jobs : int = 0


@func
def aggregate_results(results: List[TrainTestResults]) ->GridTestResults:
  n = len(results)
  accs = [r.accuracy for r in results]
  return GridTestResults(
    mean_accuracy = (1. / n) * sum(accs),
    all_accuracies= accs,
    num_jobs=n,
  )

@func
def sematic_gpu_grid_test(num_jobs: int) -> GridTestResults:
  log.info(f"Creating {num_jobs} jobs to run ...")
  futures = []
  for i in range(num_jobs):
    params = TrainTestParams(use_cuda=True)
    params.learning_rate *= (2 ** i)
    future = sematic_train_one_gpu_node(params)
    future.set(
      name=f"sematic_gpu_grid_test_job_{i}",
    )
    futures.append(future)
  
  log.info(f"... connecting to `aggregate_results()` future ... ")
  final_future = aggregate_results(futures)
  final_future.set(name="Aggregate Final Results")
  
  return final_future
