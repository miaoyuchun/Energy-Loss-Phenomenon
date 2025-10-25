# The Energy Loss Phenomenon in RLHF: A New Perspective on Mitigating Reward Hacking
This repository contains the official implementation of this article

**[[ICML 2025] The Energy Loss Phenomenon in RLHF: A New Perspective on Mitigating Reward Hacking][1]**

[Yuchun Miao][myc], [Sen Zhang][zs], [Liang Ding][dl], [Yuqi Zhang][zyq], [Lefei Zhang][zlf], [Dacheng Tao][tdc]


## Installation
The runtime environment and Docker container dependencies are specified in `packages.txt`, `requirements.txt`, and `system_packages.txt`.

## Energy Loss Phenomenon Demo

To help readers easily reproduce the **energy loss phenomenon**—which constitutes the core insight of our paper—we provide a demo located in the `Energy_Loss_Phenomenon_Demo` folder.
You can run it directly with the following command:

```bash
python ./Energy_Loss_Phenomenon_Demo/llama_avg_energyloss_last_layer.py \
    --rlhf_model ${rlhf_model_path} \
    --sft_model ${sft_model_path}
```

* `rlhf_model_path` and `sft_model_path` refer to the checkpoints of the RLHF and SFT models, respectively.
  We provide an example using Llama2-7B model weights available at [XXXX](#).
* The corresponding responses on the AlpacaFarm dataset are provided in
  `Energy_Loss_Phenomenon_Demo/response_new.jsonl`,
  and the associated reward hacking labels are available in
  `Energy_Loss_Phenomenon_Demo/hacking_label_new.json`.


## Energy Loss-Aware PPO (EPPO) Algorithm 
The implementation of our **EPPO** algorithm is provided under the `training/` directory.
To reproduce our experimental results, simply run the following commands:

```bash
# PPO
bash training/step3_rlhf_finetuning/slurm_scripts_miao/ppo_llama2.sh

# EPPO
bash training/step3_rlhf_finetuning/slurm_scripts_miao/eppo_llama2.sh
```

* Before running the above commands, please replace `WORKSPACE`, `actor_model`, `critic_model`, and `recipes` with the corresponding project path, SFT model path, RM model path, and dataset path, respectively.
* The core implementation of the algorithm can be found in
  `training/step3_rlhf_finetuning/ppo_trainer.py`, specifically from Line 346 to Line 365.
* To efficiently compute the energy loss of training prompts under the SFT model during the RLHF process, we additionally deploy a separate SFT model.
  The related deployment code is provided under the `lmdeploy_deploy/` directory.

## Citation
If you find our work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{miao2025the,
title={The Energy Loss Phenomenon in {RLHF}: A New Perspective on Mitigating Reward Hacking},
author={Yuchun Miao and Sen Zhang and Liang Ding and Yuqi Zhang and Lefei Zhang and Dacheng Tao},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=82A81az3V5}
}
```

[1]: https://arxiv.org/abs/2501.19358
[myc]: https://scholar.google.com/citations?user=-ec3mwUAAAAJ&hl=en
[zs]: https://scholar.google.com/citations?user=-bJJNV0AAAAJ&hl=en
[dl]: https://scholar.google.com/citations?user=lFCLvOAAAAAJ&hl=en
[zyq]: https://scholar.google.com/citations?user=GfiZkoAAAAAJ&hl=en
[zlf]: https://scholar.google.com/citations?user=BLKHwNwAAAAJ&hl=en
[tdc]: https://scholar.google.com/citations?user=RwlJNLcAAAAJ&hl=en


# Thanks
This project is based on [DeepSpeed-Chat](https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-chat). Thanks for this wonderful work!<br>