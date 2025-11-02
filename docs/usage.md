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
[UNet][chuchichaestli.models.unet.UNet] or built with the components
implemented in [diffusion][chuchichaestli.diffusion], [models.attention][chuchichaestli.models.attention], [models.adversarial][chuchichaestli.models.adversarial], and more.

These models are **not pre-trained**, meaning for proper functioning
they have to be trained using appropriate data and objectives (loss
functions).


#### Example
The U-Net architecture consists of an encoder-decoder structure with
skip connections which ensure spatial information is passed through
the network (even for higher compression levels). The building blocks
of the U-Net can have various forms, but generally consist of
convolutional layers. In this example, the encoder is purely
convolutional, whereas the decoder includes a mixture of attention and
(transposed) convolutional layers.

```python
from chuchichaestli.models.unet import UNet

model = UNet(
	dimensions=2,        # spatial dimensions
	in_channels=3,       # input image channels such as RGB
	n_channels=64,       # channels of first hidden layer
	out_channels=3,      # output image channels such as RGB
	down_block_types=("DownBlock",)*4,     # simple residual blocks
	up_block_types=("AttnUpBlock",)*4,     # residual blocks with attention heads in front
	block_out_channel_mults=(1, 2, 2, 4),  # channel multipliers with each level
	res_act_fn="prelu",  # parametric ReLU
	res_dropout=0.4,     # dropout for residual blocks
	attn_n_heads=2,      # number of attention heads per block,
	skip_connection_action="concat",       # skip connections are concatenated in decoder
)
print(model)
```


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
