# Third-party notices

- [FastWAM-JAX](https://github.com/iPPLab/FastWAM-JAX) by the FastWAM-JAX
  contributors. HOG-WAM extends its JAX/Flax training and inference code
  under the [MIT license](LICENSE).

- [Fast-WAM](https://github.com/yuantianyuan01/FastWAM) by the FastWAM Authors.
  The model and RoboTwin adapter follow Fast-WAM. See the
  [Fast-WAM MIT license](LICENSES/FastWAM-MIT.txt).
- [MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion). JAX Wan layer,
  sharding, and checkpointing implementations reference MaxDiffusion. See its
  [Apache-2.0 license](LICENSES/Apache-2.0.txt).
- The evaluation client derives from RoboTwin's client under the
  [MIT license](LICENSES/RoboTwin-MIT.txt). It supports explicit scene seeds
  and per-trial reports. The adapter follows the upstream Fast-WAM policy.
  The setup script pins RoboTwin to `c3ddfa8b97d5519efa828b075999bd0006778e5e`
  and cuRobo to `v0.7.8`.
  [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) ·
  [cuRobo](https://github.com/NVlabs/curobo) ·
  [RoboTwin assets](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)
- [Wan2.2](https://github.com/Wan-Video/Wan2.2). The original Wan model weights
  and VAE are distributed independently under
  [Apache-2.0](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B).
