#!/usr/bin/env python
# vim: tabstop=2 shiftwidth=2 expandtab

# Copyright 2023 Maintainers of sematic_torch_test

"""
cli.py - A pure-python script to:
  (1) help build and run `sematic_torch_test`
  (2) document (through code) canonical usage of this project
       and provide inline comments on important details

We use a dockerized environment for both ease of reproducibility as well as
to help ensure the local development environment closely matches the 
image / runtime used in K8S execution.

See this project's root README.md for usage.

"""

import sys
assert sys.version_info[0] != 2, "Python 2.7 not supported no mores :("

import datetime
import os
import subprocess
from pathlib import Path

## Preamble

### Logging
import logging

LOG_FORMAT = "%(asctime)s\t%(name)-4s %(process)d : %(message)s"
log = logging.getLogger("op")
log.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
log.addHandler(console_handler)

### Utils


def run_cmd(cmd):
  cmd = cmd.replace('\n', '').strip()
  log.info("Running %s ..." % cmd)
  subprocess.check_call(cmd, shell=True)
  log.info("... done with %s " % cmd)


## Go!


def create_arg_parser():
  import argparse

  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  gconf = parser.add_argument_group('Global Configuration')
  gconf.add_argument(
    '--root',
    default=Path(__file__).parent.resolve(),
    help='Use source at this root directory [default %(default)s]')
  gconf.add_argument(
    '--dev-container-name',
    default='my-sematic-torch-test-dev',
    help='Use Dockerized shell with this name [default %(default)s]')
  gconf.add_argument(
    '--registry-name',
    default='my-master:30100',
    help='Use this hostname for the Docker registry accessible to k8s '
    '[default %(default)s]')
  gconf.add_argument(
    '--image-name',
    default='sematic_torch_test:v1',
    help='Build and use this Docker image for local and cloud execution '
    '[default %(default)s]')

  gaction = parser.add_argument_group('Docker Actions')
  gaction.add_argument('--docker-build',
                       default=False,
                       action='store_true',
                       help='Build image for local use')
  gaction.add_argument('--docker-push',
                       default=False,
                       action='store_true',
                       help='Push base image to registry')
  gaction.add_argument(
    '--shell',
    default=False,
    action='store_true',
    help='Drop into a dockerized shell for building, testing, running, etc')
  gaction.add_argument('--shell-rm',
                       default=False,
                       action='store_true',
                       help='Remove the dockerized dev shell')

  gdaction = parser.add_argument_group('Developer Actions')
  gdaction.add_argument(
    '--auto-format',
    default=False,
    action='store_true',
    help='Auto-format all code (run this in the container).')
  gdaction.add_argument(
    '--local-test',
    default=False,
    action='store_true',
    help='Exercise main.py functionality on the local machine.')

  return parser


def main(args=None):
  if not args:
    parser = create_arg_parser()
    args = parser.parse_args()

  if args.docker_build:
    dockerfile_path = args.root / 'Dockerfile'
    run_cmd(f"""
      DOCKER_BUILDKIT=1 \
        docker build \
          -f {dockerfile_path} \
          -t {args.image_name} \
            {args.root}
    """)

  if args.docker_push:
    run_cmd(f"""
      docker tag {args.image_name} {args.registry_name}/{args.image_name} &&
      docker push {args.registry_name}/{args.image_name}
    """)

  if args.shell:
    log.info("Starting dev container ...")

    # Sematic wants to know the Sematic API Server address, among other things,
    # from ~/.sematic/settings.yaml.  If the user does not have that file
    # then use the example one from this project.
    sematic_settings_path = Path.home() / '.sematic' / 'settings.yaml'
    if not sematic_settings_path.exists():
      sematic_settings_path = args.root / 'sematic_settings.yaml'
      log.info("")
      log.info(
        f"Using the in-repo sematic settings at {sematic_settings_path}. To "
        f"use your own Sematic settings, create ~/.sematic/settings.yaml "
        f"yourself, e.g. copy {sematic_settings_path} to your home directory "
        f"at ~/.sematic/settings.yaml.")
      log.info("")
    sematic_settings_vol_mount = (
      f'-v {sematic_settings_path}:/root/.sematic/settings.yaml:ro')

    # Sematic requires Docker access in order to build and push the worker
    # image, so we'll use docker-in-docker
    did_volmount = '-v /var/run/docker.sock:/var/run/docker.sock'
    if sys.platform == 'win32':
      # I don't have a Windows, but allegedly this works
      # https://stackoverflow.com/a/41005007
      did_volmount = '-v //var/run/docker.sock:/var/run/docker.sock'

    # NB: We need to use --add-host below because the DNS in our test
    # environment is very basic and Sematic in the container fails to
    # resolve our WebUI's hostname otherwise.  You might not need --add-host
    # in your own deployment.

    run_cmd(f"""
      docker run \
            --gpus all \
            --name {args.dev_container_name} \
            --net host \
            -it -d \
            --add-host "my-master:192.168.0.100" \
            --privileged \
            {did_volmount} \
            {sematic_settings_vol_mount} \
            -v {args.root}:/opt/sematic_torch_test:z \
            -w /opt/sematic_torch_test \
            -v /:/outer_root \
              {args.image_name} \
                sleep infinity \
                  || docker start {args.dev_container_name} || true
    """)

    log.info("Dropping into Dockerized bash ...")
    EXEC_CMD = f'docker exec -it {args.dev_container_name} bash'
    os.execvp("docker", EXEC_CMD.split(' '))

  elif args.shell_rm:
    try:
      run_cmd(f'docker rm -f {args.dev_container_name}')
    except Exception:
      pass
    log.info(f"Removed container {args.dev_container_name}")

  if args.auto_format:
    try:
      run_cmd("which yapf")
    except Exception as e:
      raise ValueError(
        f"""Try running auto-formatting inside a Dockerized shell,
        which will have yapf.  Error: {e}
        """)

    run_cmd(f"""
      find {args.root} -name '*.py' -print0 | xargs -0 yapf -i
    """)
  elif args.local_test:
    assert False, 'TODO'

if __name__ == '__main__':
  main()
