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
The [data][chuchichaestli.data] module provides dataset classes for
common file types: [HDF5Dataset][chuchichaestli.data.HDF5Dataset],
[ImageDataset][chuchichaestli.data.ImageDataset],
[NumpyDataset][chuchichaestli.data.NumpyDataset], and
[SafetensorsDataset][chuchichaestli.data.SafetensorsDataset].
Each takes a single file or a wildcard pattern spanning an entire
directory tree, and presents the whole collection as one indexable
dataset. Accessed samples are cached as PyTorch tensors in shared memory
up to a budget you set, with anything beyond it read from disk. With
sufficient RAM, subsequent training epochs can be considerably
accelerated.

For paired data, each class has a `Zip*` counterpart
([ZipHDF5Dataset][chuchichaestli.data.ZipHDF5Dataset],
[ZipImageDataset][chuchichaestli.data.ZipImageDataset],
[ZipNumpyDataset][chuchichaestli.data.ZipNumpyDataset], and
[ZipSafetensorsDataset][chuchichaestli.data.ZipSafetensorsDataset]).
Much like Python's built-in `zip`, these read several sources in
lockstep and yield one sample per index as a tuple or dict, which is the
usual setup for inputs and their labels or masks. Use
`ZipHDF5Dataset.from_groups(path, "image/*", "mask/*")` to pair groups
within the same file(s), or [ZipDataset][chuchichaestli.data.ZipDataset]
directly to combine datasets you have already built.

#### Example

Say, you have several HDF5 files with image datasets stored as
```console
data
└── mnist_h5_tests
    ├── test_scenario_1
    │   └── mnist_test_1.h5
    ├── test_scenario_2
    │   └── mnist_test_2.h5
    ⋮
    └── test_scenario_13
        ├── test
        │   ├── mnist_test_test_00.h5
        │   └── mnist_test_test_01.h5
        ├── train
        │   ├── mnist_test_train_00.h5
        │   └── mnist_test_train_01.h5
        └── val
            ├── mnist_test_val_00.h5
            └── mnist_test_val_01.h5
```

then the following creates a dataset with 16 MiB of shared memory allocation to
cache image tensors read from the H5 MNIST dataset (test scenario 13)

```python
--8<-- "examples/hdf5_dataset.py:build"
```

[ZipHDF5Dataset][chuchichaestli.data.ZipHDF5Dataset] reads multiple sample
groups in parallel
```python
--8<-- "examples/zip_hdf5_dataset.py:build"
```

#### Procedural toy datasets

For quick experiments and sanity checks, the
[ProceduralDataset][chuchichaestli.data.ProceduralDataset] subclasses
synthesize their samples on the fly instead of reading them from disk:
[HalfMoonsDataset][chuchichaestli.data.HalfMoonsDataset],
[SpiralsDataset][chuchichaestli.data.SpiralsDataset],
[CheckerboardDataset][chuchichaestli.data.CheckerboardDataset],
[RingsDataset][chuchichaestli.data.RingsDataset],
[ConcentricSpheresDataset][chuchichaestli.data.ConcentricSpheresDataset],
[GaussiansDataset][chuchichaestli.data.GaussiansDataset], and
[SwissRollDataset][chuchichaestli.data.SwissRollDataset]. Each takes a
sample count, a feature dimensionality, a `noise` level, and a `seed`
for reproducibility, generating in pure PyTorch into the same
shared-memory cache as the file-backed datasets.

```python
--8<-- "examples/procedural_datasets.py:build"
```

The shapes are not restricted to the plane; raise `dim` to embed them in
a volume instead.

```python
--8<-- "examples/procedural_datasets.py:swissroll"
```

Finally, you can also wrap your own generator function with
[generate_procedural_dataset][chuchichaestli.data.generate_procedural_dataset].

Running `examples/procedural_datasets.py` renders all preset generators:

![Scatter plots of six procedural toy datasets: two interleaving half
moons, two interlocking spiral arms, two concentric rings, and six
circularly arranged Gaussian blobs in 2D, plus a Swiss roll sheet and two
concentric spheres in 3D](assets/procedural_datasets.png)


### Models
The [models][chuchichaestli.models] module provides various neural
network models ready to be instantiated such as
[UNet][chuchichaestli.models.unet.UNet],
[VAE][chuchichaestli.models.autoencoder.VAE], or built with the
components implemented in [diffusion][chuchichaestli.diffusion],
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

Autoencoders take their two components as arguments, so an encoder and a decoder
can be configured, reused, or swapped independently. When you would rather
describe the architecture in one call, `build` assembles both components for you:
arguments shared by them stay flat, while anything specific to one component goes
into `encoder_args` or `decoder_args`, whose keys are the parameter names of
[Encoder][chuchichaestli.models.autoencoder.Encoder] and
[Decoder][chuchichaestli.models.autoencoder.Decoder]. Keys you leave out fall
back to the defaults of the component itself, which is where each variant keeps its
architecture: `VAE` builds a `VAEEncoder` that doubles the latent channels, and
`DCAE` a `DCEncoder`/`DCDecoder` pair of deep-compression blocks.

```python
from chuchichaestli.models.autoencoder import DCAE, DCDecoder, DCEncoder

model = DCAE.build(dimensions=3, latent_dim=32)
# equivalently, with the components built by hand
encoder = DCEncoder(dimensions=3, out_channels=32)
decoder = DCDecoder(
    dimensions=3, in_channels=32, n_channels=encoder.bottleneck_channels
)
model = DCAE(encoder, decoder)
```

Both forms map directly onto a configuration framework such as hydra, either as
nested groups you can override one at a time, or as a single flat group

```yaml
model:
  _target_: chuchichaestli.models.autoencoder.DCAE
  encoder:
    _target_: chuchichaestli.models.autoencoder.DCEncoder
    dimensions: 3
    out_channels: 32
  decoder:
    _target_: chuchichaestli.models.autoencoder.DCDecoder
    dimensions: 3
    in_channels: 32
    n_channels: 1024
# or
model:
  _target_: chuchichaestli.models.autoencoder.DCAE.build
  dimensions: 3
  latent_dim: 32
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

The Mermaid backend compiles a `.mmd` file that can be imported into a mermaid
web-editor (e.g. [mermaid.live](https://mermaid.live)), where individual nodes
can be dragged to fine-tune the layout. The overall flow is controlled with
`direction`/`group_direction`, and `group_by` nests per-block subgraphs inside
the encoder/decoder subgraphs; skip connections render as dashed edges.

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
All metrics share the same interface:
[`.update`][chuchichaestli.metrics.base.EvalMetric.update] registers a
batch into the aggregate state while iterating over the evaluation set,
and [`.compute`][chuchichaestli.metrics.mse.MSE.compute] reduces that
aggregate to a value once the loop is done.

A toy stand-in for a model and its evaluation set:

```python
--8<-- "examples/metrics_evaluation.py:setup"
```

```python
--8<-- "examples/metrics_evaluation.py:battery"
```

Note the argument order: `update` takes the observed data first, the
prediction second. [FID][chuchichaestli.metrics.FID] shares the
interface but compares feature distributions instead of pixels, keeping
real and fake statistics apart; its default InceptionV3 extractor is
downloaded on first use.

```python
--8<-- "examples/metrics_evaluation.py:fid"
```
