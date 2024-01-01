## Overiew

Scripts and Docker images for developing on or with `graphpatch`. All scripts assume they are being run from the root of the repository.

## Docker images
### base
`dev/containers/base.Dockerfile`

Base container for running `graphpatch`; installs only the required dependencies. Uses [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda/) as the base image.

#### Building
```
./dev/build_base.sh
```
Builds the base Docker image and tags it `graphpatch-base`.

#### Basic usage
```
docker run -it --gpus all --rm graphpatch-base bash
```
Starts a shell within the image, granting access to host GPU's. If the host system lacks CUDA-supported GPU's, omit `--gpus all`. If you are developing experiments or an application using `graphpatch`, consider instead creating your own Docker image with this as a base using the [FROM](https://docs.docker.com/engine/reference/builder/#from) instruction.

### dev
`dev/containers/dev.Dockerfile`

Container which I used during development on [RunPod](https://www.runpod.io/) instances. No affiliation, nor is this an endorsement, but I did find this service, along with [Tailscale](https://tailscale.com/) to join to a private VPN, convenient for testing on multi-GPU setups. This container's setup is not RunPod-specific, but it does assume the use of Tailscale; you will need to set the `TAILSCALE_AUTH_KEY` and `TAILSCALE_HOST_NAME` environment variables in the container launch configuration.

Installs all dev and optional dependencies, as well as Tailscale for joining to a private VPN. `TAILSCALE_AUTH_KEY` should be set to a [Tailscale auth key](https://tailscale.com/kb/1085/auth-keys) for your Tailnet. I found using ephemeral, reusable keys useful for development. The value of `TAILSCALE_HOST_NAME` will be used to set the name of the host when it joins your Tailnet. For example, during development I used the name `graphpatch`, which allowed me to do things like
```
ssh graphpatch
```
from any other device on my Tailnet.

#### Building
```
./dev/build_dev.sh
```

Builds the dev Docker image and tags it `graphpatch-dev`.

#### Usage
```
./dev/dev_shell.sh
```

Runs a bash shell inside the dev image, locally. Mounts the repository directory as a volume mapped to `/graphpatch` inside the container.

```
./dev/dev_local.sh
```

Runs the container locally, using the same configuration as when running it on a hosting service. Connects to your Tailnet using the `TAILSCALE_AUTH_KEY` and `TAILSCALE_HOST_NAME` environment variables and launches the Tailscale SSH server. You can then connect to the container over SSH from any other device on your Tailnet. Also mounts the repository directory as a volume mapped to `/graphpatch` inside the container.

#### Other recommendations
A workflow I found useful was launching this image on RunPod, then setting up file synchronization between my local machine and the running container via [Mutagen](https://mutagen.io/). I created a Docker image to streamline this process, available [here](https://github.com/evan-lloyd/codesync/tree/main).

## Utility Scripts
`dev/toxin.sh` &mdash; run a command inside the designated `tox` environment; by [Daniel Pryden](https://github.com/dpryden), script taken from https://gist.github.com/dpryden/92a9a94ed21207bba549bbe7ac41ca9f

Example usage:
```
./dev/toxin.sh -e test-py38-torch21-extranone bash
```
Starts a shell within the Python 3.8, PyTorch 2.1, no-`transformers` test environment. Note that the environment must have been previously initialized. I found this extremely useful for debugging the dependency setup within the Tox testing matrix.
