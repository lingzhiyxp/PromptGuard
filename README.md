<!---
Copyright 2022 - The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<!-- <p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/diffusers/main/docs/source/en/imgs/diffusers_library.jpg" width="400"/>
    <br>
<p> -->

# Introduction
This is the official code for paper "[PromptGuard : Soft Prompt-Guided Unsafe Content Moderation for Text-to-Image Models](https://arxiv.org/abs/2501.03544)".
You could check our [Project Website](https://prompt-guard.github.io/) for more information.
We have released our pretrained model on [Hugging Face](https://huggingface.co/Prompt-Guard/PromptGuard_weights). Please check out how to use it for inference.
This implementation can be regarded as an example that can be integrated into the Diffusers library.

# Training Dataset
You could download our training dataset from [this link](https://drive.google.com/file/d/1czQL3-H-Z83XAZuTmJdgTIX2altah6A6/view?usp=sharing). The training dataset is **not permitted for any commercial use**.

# Environments and Installation
```bash
conda create -n promptguard python=3.9
conda activate promptguard
pip install -r requirements.txt
```

# Individual Safety Embedding Training
```bash
bash training.sh
```
You could modify the parameters in training.sh file. Normally, we just need to modify the coefficient, max_train_steps and the file and folder paths.

# Inference
```python
from diffusers import StableDiffusionPipeline
import torch
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# remove the safety checker
def dummy_checker(images, **kwargs):
    return images, [False] * len(images)
pipe.safety_checker = dummy_checker

safety_embedding_list = [${embedding_path_1}, ${embedding_path_2}, ...] # the save paths of your embeddings
token1 = "<prompt_guard_1>"
token2 = "<prompt_guard_2>"
...
token_list = [token1, token2, ...] # the corresponding tokens of your embeddings

pipe.load_textual_inversion(pretrained_model_name_or_path=safe_embedding_list, token=token_list)

origin_prompt = "a photo of a dog"
prompt_with_system = origin_prompt + " " + token1 + " " + token2 + ...
image = pipe(prompt).images[0]
image.save("example.png")
```

To get a better balance between unsafe content moderation and benign content preservation, we recommend you to load Sexual, Political and Disturbing these three safe embeddings.

# Acknowledgement

This work is based on the amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [🤗 Diffusers](https://github.com/huggingface/diffusers)
```latex
@misc{von-platen-etal-2022-diffusers,
    author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Thomas Wolf},
    title = {Diffusers: State-of-the-art diffusion models},
    year = {2022},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/huggingface/diffusers}}
}
```

- [Unsafe Diffusion](https://github.com/YitingQu/unsafe-diffusion)
```latex
@inproceedings{QSHBZZ23,
    author = {Yiting Qu and Xinyue Shen and Xinlei He and Michael Backes and Savvas Zannettou and Yang Zhang},
    title = {{Unsafe Diffusion: On the Generation of Unsafe Images and Hateful Memes From Text-To-Image Models}},
    booktitle = {{ACM SIGSAC Conference on Computer and Communications Security (CCS)}},
    publisher = {ACM},
    year = {2023}
}
```

- [Textual Inversion](https://github.com/rinongal/textual_inversion)
```latex
@misc{gal2022textual,
      doi = {10.48550/ARXIV.2208.01618},
      url = {https://arxiv.org/abs/2208.01618},
      author = {Gal, Rinon and Alaluf, Yuval and Atzmon, Yuval and Patashnik, Or and Bermano, Amit H. and Chechik, Gal and Cohen-Or, Daniel},
      title = {An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion},
      publisher = {arXiv},
      year = {2022},
      primaryClass={cs.CV}
}
```