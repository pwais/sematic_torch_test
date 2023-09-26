# `sematic_torch_test`: A Simple pytorch-based Pipeline to demo Sematic K8S Execution

Please see the accompanying article here for how to set up a bare metal
K8S+Sematic cluster for use this this demo: TODO LINK

## Quickstart

To run this code, you'll need a machine with Docker and
[`nvidia-docker`](https://github.com/nvidia/nvidia-container-runtime#installation)
installed and working properly.

For cluster execution find instances of these strings and configure them for
your own cluster:
 * Find `my-master:30500` and set it to the hostname (and optionally port) of
    your K8S cluster's (private) Docker registry.
 * Find `http://my-master` and set it to the URL of your Sematic Server.
 * Find `192.168.0.100` and set it to the IP address of a machine serving
    the K8S Ingress for the Sematic WebUI.  E.g. use the IP address of your
    K8S master.


On your local machine, first drop into a Dockerized development shell:
  ```bash
  mycomputer $$ devtool.py --docker-build --shell
  ```

Now see the `main.py`` docs and run through the examples provided there:
  ```bash
  indocker $ python3 main.py --help
  ```

## Development

Before a release use:
```bash
indocker $ devtool.py --auto-format
indocker $ devtool.py --local-test
```
