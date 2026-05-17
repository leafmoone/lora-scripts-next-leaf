## Open Source Notices

### Upstream GUI / community packaging

This repository is maintained as **[wochenlong/lora-scripts-next](https://github.com/wochenlong/lora-scripts-next)** and traces its UX and packaging to **Akegarasu SD-Trainer** / **秋叶一键训练包**: **[Akegarasu/lora-scripts](https://github.com/Akegarasu/lora-scripts)** (training backend integration: **[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)**).

### Rectified Flow (SDXL LoRA)

This project includes Rectified Flow training support for SDXL LoRA inspired by and adapted from:

- `bluvoll/Akegarasu-lora-scripts-RF`: https://github.com/bluvoll/Akegarasu-lora-scripts-RF

The referenced repository is licensed under AGPL-3.0. Its RF training ideas include the Rectified Flow objective, sigma timestep sampling, optional cosine optimal transport pairing, and SDXL resolution-dependent timestep shift controls.

### Anima LoRA

Active Anima backend maintenance is based on:

- `kohya-ss/sd-scripts`: https://github.com/kohya-ss/sd-scripts

Earlier Anima integration work also referenced:

- `WhitecrowAurora/lora-rescripts` (**SD-reScripts** — historical fork / continuation of the LoRA-scripts line): https://github.com/WhitecrowAurora/lora-rescripts

The historical reference repository is licensed under AGPL-3.0. Current Anima training should be synchronized from `kohya-ss/sd-scripts`; local code is limited to the WebUI compatibility wrapper, config adapter, defaults, and launch orchestration.
