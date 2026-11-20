# R3GAN2 project page

The project page for *Magnitude-preserving Discriminators Enable Efficient GANs* (NeurIPS 2026), served at
[visual.cs.brown.edu/r3gan2](https://visual.cs.brown.edu/r3gan2). Code: [brownvc/R3GAN2](https://github.com/brownvc/R3GAN2).

- `index.html`: the project page (samples, speed, both live generators, abstract, citation).
- `map.html`: the ImageNet GANdscape, a zoomable map of all 1,000 classes generated live with WebGPU.
- `generator.html`: the live sample generator, embedded in the project page.

The model weights, class embeddings and decoders are fetched from `DATA` in `config.js`, which points at the Hugging
Face repository [brownvc/R3GAN2-WebGPU](https://huggingface.co/brownvc/R3GAN2-WebGPU). Set it to an empty string to
read them from this folder instead.

Before pushing, run `python check.py`: it checks section order, local links and images, and that every data file is
fetched through `DATA`.
