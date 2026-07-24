# Vendored LeRobot source

This directory contains the AV-piper project's local LeRobot fork.

- Upstream project: <https://github.com/huggingface/lerobot>
- Declared upstream package version: `0.1.0`
- Local source at migration time: `/home/dc/dc_project/lerobot/lerobot`
- Migration date: `2026-07-16`
- License: Apache License 2.0; see `LICENSE` in this directory.
- Base commit: unavailable because the source checkout's submodule Git metadata
  pointed to a missing `.git/modules/lerobot` directory.

The migration initially copied the complete Python package and runtime
resources while excluding bytecode caches and `scripts/outputs` evaluation
artifacts. Before the integration-only pruning and version fallback described
below, the SHA-256 digest of the sorted, relative-path source snapshot manifest
was:

```text
a127e44c61def0c6cec543b2bce420e8d013221625528813c1adc2dd8de5e72d
```

Known AV-piper-specific changes include:

- single-head Diffusion Policy changes used by DPPO;
- dual-head, coupled dual-head, and two-model Diffusion policies;
- Diffusion configuration and policy factory extensions for those policies;
- checkpoint/config compatibility fixes;
- W&B resume and video logging fixes in `common/logger.py`;
- a `0.1.0` version fallback when no standalone `lerobot` distribution
  metadata exists, as expected for the vendored installation;
- removal of unused upstream CLI scripts, their Hydra configs and HTML
  template, plus the legacy batch encoder that depended on the removed CLI.

The copied files retain their original copyright and license notices. Some
files also retain third-party notices embedded in their source, including the
MIT notice in `common/policies/vqbet/vqbet_utils.py` and the BSD-3-Clause
notice in
`common/datasets/push_dataset_to_hub/_umi_imagecodecs_numcodecs.py`.
