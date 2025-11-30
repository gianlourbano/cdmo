# CDMO Project

To run all models with docker, use the following command:

```bash
chmod +x build-image.sh
./build-image.sh
chmod +x docker_run.sh
./docker_run.sh
```

This will build the docker image and execute all the models (with run_all.sh).

To run a single model, use:

```bash
chmod +x run.sh
./run.sh <n> <approach> --model <model_name> [--solver <solver_name>] [--optimization]
```

For instance,

```bash
./run.sh 26 SMT --model presolve_2 --optimization
```

The solver option is only available for CP.
Also, for optimization models, the `--optimization` flag is not necessary for CP given that optimization models are separated from decisional ones (they all start with opt_).