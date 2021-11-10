# Testing locally with Docker

Testing the package locally will help you diagnose problems before you put in a pull request.

Importantly, it mimics doing a fresh install without you having to change your dev environment.  
It _should_ behave similarly to a GitHub CI runner.


## On host

First, cd into the root of the git repo.

`cd $(git rev-parse --show-toplevel)`

Then, make sure the docker image is built.

`docker build .docker -t qfitdev:latest`

Launch an interactive docker container, feeding in the git repo.

```bash
docker run --rm -i -t \
       --mount type=bind,src=$(pwd),dst=/mnt/qfitsrc \
       qfitdev
```

## In container

Then, run the tests, avoiding writing (as much as possible) to the `/mnt/qfitsrc` directory. This is helped by the `alwayscopy = True` in `tox.ini`.

```bash
tox --workdir=/tmp \
    -c /mnt/qfitsrc
```
