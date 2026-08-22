# Usage

`chuchichaestli` provides various modules that are essential
throughout the creation of neural network models, from training to
evaluation. It is meant as a repository of building blocks with which
you can build your own neural network models.

!!!note "Note"
	
	The framework integrates into the PyTorch ecosystem and as any artificial 
	intelligence application is most efficiently used on GPU-based hardware.

We recommend combining the package with a configuration framework such
as [hydra](https://hydra.cc/). It can easily configure and instantiate
`chuchichaestli` modules such as data loaders, neural network models,
loss functions, and evaluation metrics.


### Datasets
The [data][chuchichaestli.data] module provides a
[HDF5Dataset][chuchichaestli.data.HDF5Dataset] which efficiently
caches PyTorch tensors in shared memory. With sufficient RAM,
subsequent training epochs can be considerably accelerated.


#### Example

Say, you have several HDF5 files with image datasets stored as
```console
data
└── images
    ├── dodos
    │   └── images.h5
    ├── dragons
    │   └── images.h5
    └── wolpertinger
        └── images.h5
```

then the following creates a dataset with 8 GB of memory allocation to
cache image tensors read from the dataset

```python
from chuchichaestli.data import HDF5Dataset

dataset = HDF5Dataset("data/images/**/*.h5", cache="8G")
dataset.info()
sample_image = dataset[0]
```


### Models
The [models][chuchichaestli.models] module provides various neural
network models ready to be instantiated such as
[UNet][chuchichaestli.models.unet.UNet],
[models.autoencoder][chuchichaestli.models.autoencoder], or built with 
the components implemented in [diffusion][chuchichaestli.diffusion],
[models.attention][chuchichaestli.models.attention],
[models.adversarial][chuchichaestli.models.adversarial], and more.

These models are **not pre-trained**, meaning for proper functioning
they have to be trained using appropriate data and objectives (loss
functions).


#### Example
The U-Net architecture consists of an encoder-decoder structure with
skip connections which ensure spatial information is passed through
the network (even for higher compression levels). The building blocks
of the U-Net can have various forms, but generally consist of
convolutional layers. In this example, the U-Net's higher levels are purely
convolutional, whereas the lowest levels include a mixture of attention and
(transposed) convolutional layers.

```python
--8<-- "examples/unet_visualization.py:unet"
```

[`summary`][chuchichaestli.utils.info.summary] provides a torchinfo-style 
text table (no extra dependency) that allows for detailed inspection of any
model

```python
--8<-- "examples/unet_visualization.py:summary"
```

Similarly, you can build other models such as a Variational Auto-encoder (VAE)
using autoencoding-specific building blocks. The model is quite similar to a
U-Net, but misses skip connections and instead includes a stochastic regularization prior 
in the lowest layer (the latent)

```python
--8<-- "examples/vae_visualization.py:vae"
```

### Visualization

The [visualization][chuchichaestli.utils.visualization] utilities turn a model
into a schematic. A backend-agnostic semantic graph
([`build_ir`][chuchichaestli.utils.ir.build_ir]) understands the
architecture hierarchy, components (encoder/bottleneck/decoder), spatial levels,
blocks, and layers, including skip and residual connections.
Architecture-aware adapters handle U-Nets and autoencoders (VAE/VQVAE/DCAE); any
other PyTorch model (e.g. a torchvision ResNet or ViT) falls back to a generic
adapter that mirrors the module tree.

Two entry points consume the graph:
- [`matplotlib_diagram`][chuchichaestli.utils.visualization.matplotlib_diagram]
  — vector figures (PDF/SVG/PNG); requires the optional `viz` extra.
- [`mermaid_diagram`][chuchichaestli.utils.visualization.mermaid_diagram] —
  quick markdown/web diagrams.


#### Matplotlib figures

Matplotlib diagrams are most versatile and are recommended as starting point
to produce publication-ready illustrations.
`level` selects the abstraction (0=components, 1=levels, 2=blocks, 3=layers).
`label_fields` chooses what each node shows (`name`, `channels`, `kernel`,
`resolution`, `params`); `color_by` chooses what the fill colours and legend
encode (`component`, `type`, or `name`); `node_size` is `small`/`medium`/`large`.

```python
--8<-- "examples/unet_visualization.py:matplotlib"
```

An *exemplary zoom* callout expands a block into its layers: `zoom=True`
auto-picks a representative block, or pass a node id / `ZoomSpec`. `zoom_loc`
places the inset (right/left/top/bottom, the four corners, or center); pass a
list of `ZoomSpec`s to draw several at once.

```python
--8<-- "examples/unet_visualization.py:zoom"
```

![U-Net block-level diagram with two exemplary-zoom insets expanding an encoder
and a decoder block into their layers](assets/unet_zoom.svg){ .diagram }

The same block-level view of a VAE has no skip arcs and swaps the U-Net's
concatenation for a latent bottleneck (the hourglass node); here a single zoom
inset expands an encoder block into its layers:

![VAE block-level diagram with the latent rendered as an hourglass node and an
exemplary-zoom inset expanding an encoder block into its layers](assets/vae_zoom.svg){ .diagram }


#### Mermaid diagrams

The Mermaid backend compiles a file that can be imported in a mermaid web-editor.
Skip connections render as dashed edges; `group_by` nests per-block subgraphs
inside the encoder/decoder subgraphs.

```python
--8<-- "examples/unet_visualization.py:mermaid"
```

Runnable end-to-end scripts for a U-Net, a VAE, a PatchGAN discriminator, and
torchvision models (ResNet + ViT) live in
[`examples/`](https://github.com/CAIIVS/chuchichaestli/tree/main/examples).


### Metrics

The [metrics][chuchichaestli.metrics] module provides various metrics
and losses to measure and compare image quality of fake and real
samples. In contrast to many other image quality metric libraries,
`chuchichaestli`'s only dependency for this module (besides `torch`
itself) is `torchvision`. This makes `chuchichaestli` still very
lightweight and avoids package conflicts during installs.


#### Example
This example demonstrates how to use a whole battery of metrics.  Each
metric has a
[`.update`][chuchichaestli.metrics.base.EvalMetric.update] method
which registers samples and adds them to the aggregate
state. Typically, this method is used while iterating through the
evaluation set to build aggregate statistics for the entire evaluation
set. The [`.compute`][chuchichaestli.metrics.mse.MSE.compute] method
computes the metric value for the current aggregate state. This method
is typically used after iterating through an evaluation set to trigger
the actual computation (reduction).

```python
from chuchichaestli.metrics import MSE, PSNR, SSIM, FID

batch_size, num_channels, width, height = 4, 3, 512, 512
sample_images = torch.rand(batch_size, num_channels, width, height)

metrics = [
	MSE(),
	PSNR(min_value=0, max_value=1), 
	SSIM(min_value=0, max_value=1, kernel_size=7, kernel_type="gaussian"),
	FID()
]

model.eval()
with torch.no_grad():
	fake_images = model(sample_images)
	evaluations = []
	for metric in metrics:
		metric.update(fake_images, sample_images)
		val = metric.compute()
		evaluations.append(val)
		metric.reset()
print(evaluations)
```
