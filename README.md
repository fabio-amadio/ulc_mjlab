# ulc_mjlab

Implementation of **ULC** (**Unified Loco-manipulation Controller**) in `mjlab`.

Main characteristics:

- unified 29-DoF G1 controller
- staged curriculum for velocity, height, and upper-body control
- quintic command interpolation
- stochastic arm-command delay release
- residual arm actions around the delayed/interpolated arm target

For furter details refer to the original paper at [https://arxiv.org/abs/2507.06905](https://arxiv.org/abs/2507.06905).

## Usage

List the registered environments with the upstream `mjlab` CLI:

```bash
uv run list_envs --keyword ULC
```

Train the ULC G1 task:

```bash
uv run train Mjlab-ULC-Flat-Unitree-G1 --env.scene.num-envs 4096
```

W&B uploads use a rolling `model_latest.pt` checkpoint by default.

Play a trained checkpoint with the stock mjlab viewers:

```bash
uv run play Mjlab-ULC-Flat-Unitree-G1 --wandb-run-path your-org/project/run-id
```
