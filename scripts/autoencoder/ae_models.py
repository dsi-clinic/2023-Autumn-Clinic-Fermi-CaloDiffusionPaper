import copy
import math
from enum import Enum
from fractions import Fraction
from functools import partial
from inspect import isfunction

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def exists(x):
    """Checks if x is not None.
    """
    return x is not None


def default(val, d):
    """Returns val if it exists, otherswise returns d.
    If d is a function, call it and return its value.
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cosine_beta_schedule(nsteps, s=0.008):
    """Generate a cosine schedule for beta values as proposed in https://arxiv.org/abs/2102.09672.

    Parameters:
    - nsteps (int): Number of steps in the schedule.
    - s (float): Small constant to adjust the schedule.

    Returns:
    - betas (torch.Tensor): Beta schedule tensor.
    """
    x = torch.linspace(0, nsteps, nsteps + 1)
    alphas_cumprod = torch.cos(((x / nsteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


# Custom Convolutional Layers.


class CylindricalConvTrans(nn.Module):
    """Cylindrical 3D Transposed Convolution layer.

    Assumes input tensor format: (batch_size, channels, z_bin, phi_bin, r_bin)
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size=(3, 4, 4),
        stride=(1, 2, 2),
        groups=1,
        padding=1,
        output_padding=0,
    ):
        super().__init__()
        # Adjust padding for circular dimension
        if not isinstance(padding, int):
            self.padding_orig = copy.copy(padding)
            padding = list(padding)
        else:
            padding = [padding] * 3
            self.padding_orig = copy.copy(padding)

        padding[1] = kernel_size[1] - 1  # Adjust padding for phi_bin dimension
        self.convTrans = nn.ConvTranspose3d(
            dim_in,
            dim_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

    def forward(self, x):
        """Forward pass for the cylindrical transposed convolution.

        Pads the phi_bin dimension circularly before applying convolution.
        """
        # Circular padding for the phi_bin dimension
        # Out size is : O = (i-1)*S + K - 2P
        # To achieve 'same' use padding P = ((S-1)*W-S+F)/2, with F = filter size, S = stride, W = input size
        # Pad last dim with nothing, 2nd to last dim is circular one
        circ_pad = self.padding_orig[1]
        x = F.pad(x, pad=(0, 0, circ_pad, circ_pad, 0, 0), mode="circular")
        x = self.convTrans(x)
        return x


class FractionalResizeLayer(nn.Module):
    """Applies a Transposed Convolution, then a Convolution to achieve the effect
    of resizing by a fractional amount. The output size's spatial dimensions is
    the fraction times the input size.

    Assumes input tensor format: (batch_size, channels, z_bin, phi_bin, r_bin)

    Upsampling Example: 256x256x256 image by 3/2  -> 384x384x384
    N x C x 256 x 256 x 256 -(numerator=3, denominator=2)> N x C x 384 x 384 x 384

    Downsampling Example: 384x384x384 image by 2/3  -> 256x256x256
    N x C x 384 x 384 x 384 -(numerator=2, denominator=3)> N x C x 256 x 256 x 256
    """

    def __init__(
        self,
        in_channels,
        kernel_size=(3, 4, 4),
        padding=0,  # TODO: do we need different paddings for learned upsampling and learned downsampling
        output_padding=0,
        numerator=3,
        denominator=2,
    ):
        super().__init__()
        self.numerator = numerator
        self.denominator = denominator

        self.cylinConvTrans = CylindricalConvTrans(
            in_channels,
            in_channels,  # not changing the number of channels
            kernel_size=kernel_size,
            stride=(numerator, numerator, numerator),
            padding=padding,
            output_padding=output_padding,
        )

        self.cylinConv = CylindricalConv(
            in_channels,
            in_channels,  # not changing the number of channels
            kernel_size=kernel_size,
            stride=denominator,
            padding=padding,
        )

    def forward(self, x):
        input_shape = x.shape
        expected_bigger_shape = torch.Size(
            [
                input_shape[0],
                input_shape[1],
                input_shape[2] * self.numerator,
                input_shape[3] * self.numerator,
                input_shape[4] * self.numerator,
            ]
        )
        x = self.cylinConvTrans(x)
        assert x.shape == expected_bigger_shape, (x.shape, expected_bigger_shape)
        x = self.cylinConv(x)
        input2_shape = x.shape
        expected_smaller_shape = torch.Size(
            [
                input2_shape[0],
                input2_shape[1],
                input2_shape[2] // self.denominator,
                input2_shape[3] // self.denominator,
                input2_shape[4] // self.denominator,
            ]
        )
        assert x.shape == expected_smaller_shape, (x.shape, expected_smaller_shape)

        return x


class FractionalResizeTrilinear(nn.Module):
    """A module for fractional resizing using trilinear interpolation in a U-Net autoencoder.

    This module computes a sequence of target spatial dimensions (depth, height, width) by scaling
    the input shape by a factor (numerator/denominator). The sequence is generated by progressively
    downsampling until one or more dimensions become too small (<=2) or a specified number of steps
    (num_samples) is reached, and then mirroring these shapes (excluding the smallest) to form an upsampling
    path. These computed shapes are determined during the first forward pass and stored as internal,
    class-level attributes—meaning they are not provided at instantiation but are computed when processing
    the first input.

    The reset_step() method is provided for training scenarios, such as during autoencoder training with
    multiple epochs. Resetting the internal step counter at the start of each epoch ensures that the
    resizing sequence starts from the beginning for every new epoch, maintaining consistency across batches.
    """

    _start_shape = None
    _all_output_shapes = None
    _is_first = True
    _current_step = 0

    def _calculate_all_shapes(self, start_shape):
        """Compute the sequence of target shapes based on the input's spatial dimensions.

        This method scales the input shape iteratively by (numerator/denominator) for up to num_samples
        iterations or until any dimension becomes too small (<=2). It then appends the reversed sequence
        (excluding the smallest shape) to create an upsampling path.

        Args:
            start_shape (tuple): The initial spatial dimensions of the input tensor.

        Returns:
            list of tuple: The list of starting shape and target shapes for each interpolation step
            (downsampling then upsampling).
        """
        shapes = [start_shape]
        current_shape = start_shape
        scale = self.numerator / self.denominator

        # Calculate downsampling shapes
        for _ in range(self.num_samples):
            if any(x <= 2 for x in current_shape):
                # Prevent compresing to super small sizes
                shapes.append(current_shape)
                continue

            next_shape = tuple(int(x * scale) for x in current_shape)

            shapes.append(next_shape)
            current_shape = next_shape

        # Add upsampling shapes (reverse path excluding the smallest shape)
        shapes.extend(shapes[-2::-1])
        print("Data dimensions will be: ", shapes)
        return shapes

    def __init__(self, numerator=2, denominator=3, num_samples=-1):
        """Initialize FractionalResizeTrilinear.

        Args:
            numerator (int, optional): Numerator of the scaling factor. Defaults to 2.
            denominator (int, optional): Denominator of the scaling factor. Defaults to 3.
            num_samples (int, optional): Maximum number of downsampling steps.
                A negative value indicates no preset limit.
        """
        super().__init__()
        self.numerator = numerator
        self.denominator = denominator
        self.num_samples = num_samples

    def forward(self, x):
        """Resize the input tensor to a target shape using trilinear interpolation.

        On the first call, the method computes and caches the sequence of target
        shapes based on the input's spatial dimensions. For subsequent calls,
        it selects the next shape from the sequence and increments
        the internal step counter.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, z_bin, phi_bin, r_bin)

        Returns:
            torch.Tensor: The resized tensor.
        """
        FractionalResizeTrilinear._current_step += 1

        spatial_dims = tuple(x.shape[-3:])

        # Only executed the first forward pass
        if FractionalResizeTrilinear._is_first:
            FractionalResizeTrilinear._start_shape = spatial_dims
            FractionalResizeTrilinear._all_output_shapes = self._calculate_all_shapes(
                spatial_dims
            )
            FractionalResizeTrilinear._is_first = False

        # Increment step counter based on the number of times it's called
        if (
            FractionalResizeTrilinear._current_step
            == len(FractionalResizeTrilinear._all_output_shapes) - 1
        ):
            # Reset current step counter back to 0 if all the downsampling/upsampling steps are done
            FractionalResizeTrilinear._current_step = 0

        try:
            desired_shape = FractionalResizeTrilinear._all_output_shapes[
                FractionalResizeTrilinear._current_step
            ]
        except:
            return x

        x = F.interpolate(x, size=desired_shape)

        return x


class ResizeMethod(Enum):
    """Enum for different tensor resizing methods. CYLIN_FRAC_LEARNED and
    CYLIN_FRAC_INTERPOLATE are options that allow for more flexibility in
    upsampling and downsampling.

    SIMPLE_INT_CONV: Uses pytorch's nn.Conv3d for a convolution layer or
        nn.ConvTranspose3d for a transposed convolution layer. No padding applied
        beforehand, and only allows compression by an integer factor.
    CYLIN_INT_CONV: Uses CylindricalConv for a convolution layer or
        CylindricalConvTrans for a transposed convolution layer. Padding is applied
        beforehand, and only allows compression by an integer factor.
    CYLIN_FRAC_LEARNED: Applies both convolution and transposed convolutions
        for one resizing step. Padding is applied, and this combo allows for
        the effect of fractional strides.
    CYLIN_FRAC_INTERPOLATE: For one resizing step, uses only pytorch's
        nn.interpolate, but this function does not learn any parameters. Padding
        is applied, and this combo allows for the effect of fractional strides.
    """

    SIMPLE_INT_CONV = "simple-int-conv"
    CYLIN_INT_CONV = "cylindrical-int-conv"
    CYLIN_FRAC_LEARNED = "cylindrical-frac-learned"
    CYLIN_FRAC_INTERPOLATE = "cylindrical-frac-interpolate"

    def __str__(self):
        return self.value


class CylindricalConv(nn.Module):
    """Cylindrical 3D Convolution layer.

    Assumes input tensor format: (batch_size, channels, z_bin, phi_bin, r_bin)
    """

    def __init__(
        self, dim_in, dim_out, kernel_size=3, stride=1, groups=1, padding=0, bias=True
    ):
        super().__init__()
        # Adjust padding for circular dimension
        if isinstance(padding, int):
            padding = [padding] * 3
        else:
            self.padding_orig = copy.copy(padding)
            padding = list(padding)
            padding[1] = 0  # No padding for phi_bin dimension; will pad manually

        self.kernel_size = kernel_size
        self.padding_orig = copy.copy(padding)
        padding[1] = 0  # Remove padding for phi_bin dimension
        self.conv = nn.Conv3d(
            dim_in,
            dim_out,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        """Forward pass for the cylindrical convolution.

        Pads the phi_bin dimension circularly before applying convolution.
        """
        # Circular padding for the phi_bin dimension
        # To achieve 'same' use padding P = ((S-1)*W-S+F)/2, with F = filter size, S = stride, W = input size
        # Pad last dim with nothing, 2nd to last dim is circular one
        circ_pad = self.padding_orig[1]
        x = F.pad(x, pad=(0, 0, circ_pad, circ_pad, 0, 0), mode="circular")
        x = self.conv(x)
        return x


# Residual Block


class Residual(nn.Module):
    """Residual connection module.

    Applies a function and adds the input to the output (residual connection).
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def extract(a, t, x_shape):
    """Extract values from a tensor and reshape for broadcasting.

    Parameters:
    - a (torch.Tensor): Tensor to extract from.
    - t (torch.Tensor): Indices tensor.
    - x_shape (tuple): Desired output shape.

    Returns:
    - out (torch.Tensor): Extracted and reshaped tensor.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# Sinusoidal Position Embeddings


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embedding module.

    Generates embeddings for time steps.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """Forward pass to generate embeddings.

        Parameters:
        - time (torch.Tensor): Time steps tensor.

        Returns:
        - embeddings (torch.Tensor): Sinusoidal embeddings.
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# Basic Convolutional Block


class Block(nn.Module):
    """Basic convolutional block with optional group normalization and activation.
    """

    def __init__(self, dim, dim_out, groups=8, cylindrical=False):
        super().__init__()
        if not cylindrical:
            self.proj = nn.Conv3d(dim, dim_out, kernel_size=3, padding=1)
        else:
            self.proj = CylindricalConv(dim, dim_out, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()  # Swish activation function

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift  # Apply scaling and shifting

        x = self.act(x)
        return x


# ResNet Block


class ResnetBlock(nn.Module):
    """ResNet block with optional conditional embedding.

    Reference: https://arxiv.org/abs/1512.03385
    """

    def __init__(self, dim, dim_out, *, cond_emb_dim=None, groups=8, cylindrical=False):
        super().__init__()
        # Conditional embedding MLP
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(cond_emb_dim, dim_out))
            if exists(cond_emb_dim)
            else None
        )

        # Convolution layers
        conv = (
            CylindricalConv(dim, dim_out, kernel_size=1)
            if cylindrical
            else nn.Conv3d(dim, dim_out, kernel_size=1)
        )
        self.block1 = Block(dim, dim_out, groups=groups, cylindrical=cylindrical)
        self.block2 = Block(dim_out, dim_out, groups=groups, cylindrical=cylindrical)
        self.res_conv = conv if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        # this would apply conditional embedding if available
        """
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1 1")
            h = h + time_emb
        """  # Not sure why this is commented out; investigate

        h = self.block2(h)
        return h + self.res_conv(x)


# ConvNeXt Block


class ConvNextBlock(nn.Module):
    """ConvNeXt block with optional conditional embedding.

    Reference: https://arxiv.org/abs/2201.03545
    """

    def __init__(
        self, dim, dim_out, *, cond_emb_dim=None, mult=2, norm=True, cylindrical=False
    ):
        super().__init__()
        # Conditional embedding MLP
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(cond_emb_dim, dim))
            if exists(cond_emb_dim)
            else None
        )

        # Choose convolution operation based on cylindrical
        conv_op = CylindricalConv if cylindrical else nn.Conv3d

        # Depthwise convolution
        self.ds_conv = conv_op(dim, dim, kernel_size=7, padding=3, groups=dim)

        # Convolutional layers and normalization
        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            conv_op(dim, dim_out * mult, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            conv_op(dim_out * mult, dim_out, kernel_size=3, padding=1),
        )

        self.res_conv = (
            conv_op(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()
        )

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        # Apply conditional embedding if available
        if exists(self.mlp) and exists(time_emb):
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)
        return h + self.res_conv(x)  # Residual connection


# Attention Mechanisms


class Attention(nn.Module):
    """Multi-head self-attention module.
    """

    def __init__(self, dim, heads=4, dim_head=32, cylindrical=False):
        super().__init__()
        self.scale = dim_head**-0.5  # Scaling factor
        self.heads = heads
        hidden_dim = dim_head * heads

        # Define convolutional operations based on cylindrical flag
        if cylindrical:
            self.to_qkv = CylindricalConv(
                dim, hidden_dim * 3, kernel_size=1, bias=False
            )
            self.to_out = CylindricalConv(hidden_dim, dim, kernel_size=1)
        else:
            self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, kernel_size=1, bias=False)
            self.to_out = nn.Conv3d(hidden_dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, n, h, w = x.shape
        # Compute queries, keys, and values
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y z -> b h c (x y z)", h=self.heads), qkv
        )
        q = q * self.scale  # Scale queries

        # Compute attention scores
        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()  # Stabilize softmax
        attn = sim.softmax(dim=-1)

        # Compute attention output
        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y z) d -> b (h d) x y z", x=n, y=h, z=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    """Linear attention module for computational efficiency.
    """

    def __init__(self, dim, heads=1, dim_head=32, cylindrical=False):
        super().__init__()
        self.scale = dim_head**-0.5  # Scaling factor
        self.heads = heads
        hidden_dim = dim_head * heads

        # Define convolution operations based on cylindrical flag
        if cylindrical:
            self.to_qkv = CylindricalConv(
                dim, hidden_dim * 3, kernel_size=1, bias=False
            )
            self.to_out = nn.Sequential(
                CylindricalConv(hidden_dim, dim, kernel_size=1), nn.GroupNorm(1, dim)
            )
        else:
            self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, kernel_size=1, bias=False)
            self.to_out = nn.Sequential(
                nn.Conv3d(hidden_dim, dim, kernel_size=1), nn.GroupNorm(1, dim)
            )

    def forward(self, x):
        b, c, n, h, w = x.shape
        # Compute queries, keys, and values
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y z -> b h c (x y z)", h=self.heads), qkv
        )

        # Apply softmax normalization
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale  # Scale queries
        # Compute context
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        # Compute attention output
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(
            out, "b h c (x y z) -> b (h c) x y z", h=self.heads, x=n, y=h, z=w
        )
        return self.to_out(out)


class PreNorm(nn.Module):
    """Module to apply normalization before a function."""

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# Up and down sample in 2 dims but keep z dimm


def Upsample(
    dim,
    extra_upsample=[0, 0, 0],
    resize_method=ResizeMethod.CYLIN_INT_CONV,
    compress_Z=False,
    compress=2,
    num_of_samples=-1,
):
    """Upsampling layer in 2 dimensions while optionally compressing the Z dimension.

    Parameters:
    - dim (int): Input dimension.
    - extra_upsample (list): Additional output padding for upsampling.
    - resize_method (enum): which conv/interpolate method to use for upsampling.
    - compress_Z (bool): Whether to adjust Z-dimension upsampling.
    - compress (float): compression factor
    - num_of_samples (int): Number of upsampling steps to take

    Returns:
    - nn.Module: Upsampling layer.
    """
    Z_stride = compress if compress_Z else 1
    Z_kernel = 4 if extra_upsample[0] > 0 else 3

    extra_upsample[0] = 0  # Ensure Z-dimension extra upsample is zero
    if resize_method == ResizeMethod.CYLIN_INT_CONV:
        return CylindricalConvTrans(
            dim,
            dim,
            kernel_size=(Z_kernel, 4, 4),
            stride=(Z_stride, compress, compress),
            padding=1,
            output_padding=extra_upsample,
        )

    if resize_method == ResizeMethod.SIMPLE_INT_CONV:
        return nn.ConvTranspose3d(
            dim,
            dim,
            kernel_size=(Z_kernel, 4, 4),
            stride=(Z_stride, compress, compress),
            padding=1,
            output_padding=extra_upsample,
        )

    # Methods to achieve a non-integer compression factor
    fraction = Fraction(compress).limit_denominator()
    if resize_method == ResizeMethod.CYLIN_FRAC_LEARNED:
        return FractionalResizeLayer(
            in_channels=dim,
            kernel_size=(3, 4, 4),
            padding=0,
            output_padding=0,
            numerator=fraction.numerator,
            denominator=fraction.denominator,
        )

    return FractionalResizeTrilinear(
        numerator=fraction.numerator,
        denominator=fraction.denominator,
        num_samples=num_of_samples,
    )


def Downsample(
    dim,
    resize_method=ResizeMethod.CYLIN_INT_CONV,
    compress_Z=False,
    compress=2,
    num_of_samples=-1,
):
    """Downsampling layer in 2 dimensions while optionally compressing the Z dimension.

    Parameters:
    - dim (int): Input dimension.
    - resize_method (enum): which conv/interpolate method to use for downsampling.
    - compress_Z (bool): Whether to adjust Z-dimension downsampling.
    - compress (float): Compression factor
    - num_of_samples(int): Number of downsampling steps to take in total.

    Returns:
    - nn.Module: Downsampling layer.
    """
    Z_stride = compress if compress_Z else 1
    if resize_method == ResizeMethod.SIMPLE_INT_CONV:
        return nn.Conv3d(
            dim,
            dim,
            kernel_size=(3, 4, 4),
            stride=(Z_stride, compress, compress),
            padding=1,
        )

    if resize_method == ResizeMethod.CYLIN_INT_CONV:
        return CylindricalConv(
            dim,
            dim,
            kernel_size=(3, 4, 4),
            stride=(Z_stride, compress, compress),
            padding=1,
        )
    # Methods to achieve a non-integer compression factor
    fraction = Fraction(compress).limit_denominator()

    if resize_method == ResizeMethod.CYLIN_FRAC_LEARNED:
        return FractionalResizeLayer(
            in_channels=dim,
            kernel_size=(3, 4, 4),
            padding=0,
            output_padding=0,
            numerator=fraction.numerator,
            denominator=fraction.denominator,
        )
    # we swap numerator and denom. to downsample. 1.5 as compress -> num.=2, denom.=3
    return FractionalResizeTrilinear(
        numerator=fraction.denominator,
        denominator=fraction.numerator,
        num_samples=num_of_samples,
    )

    # Alternative using average pooling
    # return nn.AvgPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=0)


# Fully Connected Network (FCN)


class FCN(nn.Module):
    """Fully Connected Network with optional time and condition embeddings.

    Parameters:
    - dim_in (int): Input dimension.
    - num_layers (int): Number of layers.
    - cond_dim (int): Conditional embedding dimension.
    - time_embed (bool): Whether to use time embeddings.
    - cond_embed (bool): Whether to use condition embeddings.
    """

    def __init__(
        self,
        dim_in=356,
        num_layers=4,
        cond_dim=64,
        time_embed=True,
        cond_embed=True,
    ):
        super().__init__()

        # Time and energy embeddings
        half_cond_dim = cond_dim // 2

        # Time embedding layers
        time_layers = []
        if time_embed:
            time_layers = [SinusoidalPositionEmbeddings(half_cond_dim // 2)]
        else:
            time_layers = [
                nn.Unflatten(-1, (-1, 1)),
                nn.Linear(1, half_cond_dim // 2),
                nn.GELU(),
            ]
        time_layers += [
            nn.Linear(half_cond_dim // 2, half_cond_dim),
            nn.GELU(),
            nn.Linear(half_cond_dim, half_cond_dim),
        ]

        # Energy embedding layers
        cond_layers = []
        if cond_embed:
            cond_layers = [SinusoidalPositionEmbeddings(half_cond_dim // 2)]
        else:
            cond_layers = [
                nn.Unflatten(-1, (-1, 1)),
                nn.Linear(1, half_cond_dim // 2),
                nn.GELU(),
            ]
        cond_layers += [
            nn.Linear(half_cond_dim // 2, half_cond_dim),
            nn.GELU(),
            nn.Linear(half_cond_dim, half_cond_dim),
        ]

        self.time_mlp = nn.Sequential(*time_layers)
        self.cond_mlp = nn.Sequential(*cond_layers)

        # Main MLP layers
        out_layers = [nn.Linear(dim_in + cond_dim, dim_in)]
        for _ in range(num_layers - 1):
            out_layers.append(nn.GELU())
            out_layers.append(nn.Linear(dim_in, dim_in))

        self.main_mlp = nn.Sequential(*out_layers)

    def forward(self, x, cond, time):
        """Forward pass of the FCN.

        Parameters:
        - x (torch.Tensor): Input tensor.
        - cond (torch.Tensor): Condition tensor.
        - time (torch.Tensor): Time tensor.

        Returns:
        - x (torch.Tensor): Output tensor.
        """
        t = self.time_mlp(time)
        c = self.cond_mlp(cond)
        x = torch.cat([x, t, c], axis=-1)
        x = self.main_mlp(x)
        return x


# Conditional Autoencoder


class CondAE(nn.Module):
    # Unet with conditional layers
    """Conditional autoencoder derived from the u-net structure that encodes original data into latent space
    to reduce dimensionality and reconstruct encoded data back into its original shape (decoding)

    Parameters:
    - out_dim (int): number of output channel dimensions
    - layer_sizes (list): primary method to control downsampling, list of input channel dimensions which is zipped for UNet ResNet blocks
    - channels (int): number of input channels for initial cylindrical convolution
    - cond_dim (int): dimensionality of conditional embedding vectors
    - resnet_block_groups (int): number of groups to separate channels into for group norm operation in Block()
    - use_convnext (bool): whether to use ConvNextBlocks in UNet
    - mid_attn (bool): whether to add attention blocks in between Resnet + Downsample combos in UNet
    - compress_Z (bool): indicates need to adjust z-dimension downsampling for consistency
    - convnext_mult (int): groupnorm output dimension multiplier
    - resize_method (str): method of resizing that specifies if convolution layers vs. interpolate will be used and if padding will be added beforehand
    - data_shape (tuple): shape of input data samples
    - time_embed (bool): whether to embed time for UNet
    - cond_embed (bool): whether to embed energy
    - resnet_set (list): alternate method to control downsampling, allows removal of resnet + downsample block combinations from UNet architecture
    - compress (int): compression factor when dividing dimensions size, default 2
    """

    def __init__(
        self,
        out_dim=1,
        layer_sizes=None,
        channels=1,
        cond_dim=64,
        resnet_block_groups=8,
        use_convnext=False,
        mid_attn=False,
        block_attn=False,
        compress_Z=False,
        convnext_mult=2,
        resize_method="cylindrical-int-conv",
        data_shape=(-1, 1, 45, 16, 9),
        time_embed=True,
        cond_embed=True,
        resnet_set=[0, 1, 2],
        compress=2,
    ):
        super().__init__()

        # Determine dimensions
        self.channels = channels
        self.block_attn = block_attn
        self.mid_attn = mid_attn
        self.resnet_set = resnet_set

        # Adjust dimensionality based on resnet_set
        if self.resnet_set != [0, 1, 2]:
            self.plus_one = True
        else:
            self.plus_one = False

        in_out = list(zip(layer_sizes[:-1], layer_sizes[1:]))

        self.resize_method = ResizeMethod(resize_method)

        # Initial convolution as Conv3d or CylindricalConv
        if self.resize_method == ResizeMethod.SIMPLE_INT_CONV:
            self.init_conv = nn.Conv3d(
                channels, layer_sizes[0], kernel_size=3, padding=1
            )
            cylindrical = False
        else:
            self.init_conv = CylindricalConv(
                channels, layer_sizes[0], kernel_size=3, padding=1
            )
            cylindrical = True  # Used to inform instantiation of blocks

        # Chose block type (ResNet or ConvNeXt)
        if use_convnext:
            block_klass = partial(
                ConvNextBlock, mult=convnext_mult, cylindrical=cylindrical
            )
        else:
            block_klass = partial(
                ResnetBlock, groups=resnet_block_groups, cylindrical=cylindrical
            )

        # Time and energy embeddings
        half_cond_dim = cond_dim // 2

        # Time embedding layers
        time_layers = []
        if time_embed:
            time_layers = [SinusoidalPositionEmbeddings(half_cond_dim // 2)]
        else:
            time_layers = [
                nn.Unflatten(-1, (-1, 1)),
                nn.Linear(1, half_cond_dim // 2),
                nn.GELU(),
            ]
        time_layers += [
            nn.Linear(half_cond_dim // 2, half_cond_dim),
            nn.GELU(),
            nn.Linear(half_cond_dim, half_cond_dim),
        ]

        # Energy embedding layers
        cond_layers = []
        if cond_embed:
            cond_layers = [SinusoidalPositionEmbeddings(half_cond_dim // 2)]
        else:
            cond_layers = [
                nn.Unflatten(-1, (-1, 1)),
                nn.Linear(1, half_cond_dim // 2),
                nn.GELU(),
            ]
        cond_layers += [
            nn.Linear(half_cond_dim // 2, half_cond_dim),
            nn.GELU(),
            nn.Linear(half_cond_dim, half_cond_dim),
        ]

        self.time_mlp = nn.Sequential(*time_layers)
        self.cond_mlp = nn.Sequential(*cond_layers)

        # Initialize lists for downsampling and upsampling layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.downs_attn = nn.ModuleList([])
        self.ups_attn = nn.ModuleList([])
        self.extra_upsamples = []
        self.Z_even = []
        num_resolutions = len(in_out)

        cur_data_shape = data_shape[-3:]  # Get (Z, H, W) dimensions

        self.compress = compress  # compression factor for Z, H, W dimensions
        print(f"CaloEnco compress factor config: {self.compress}", flush=True)
        # Build the downsampling layers
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            if not is_last:
                # Added +1 for extra upsampling to match dimensionality for pytorch model
                if self.plus_one:
                    extra_upsample_dim = [
                        (cur_data_shape[0] + 1) % self.compress,
                        cur_data_shape[1] % self.compress,
                        (cur_data_shape[2] + 1) % self.compress,
                    ]
                else:
                    extra_upsample_dim = [
                        (cur_data_shape[0] + 1) % self.compress,
                        cur_data_shape[1] % self.compress,
                        (cur_data_shape[2]) % self.compress,
                    ]
                # Update current data shape based on downsampling
                Z_dim = (
                    cur_data_shape[0]
                    if not compress_Z
                    else math.ceil(cur_data_shape[0] / self.compress)
                )
                cur_data_shape = (
                    Z_dim,
                    cur_data_shape[1] // self.compress,
                    cur_data_shape[2] // self.compress,
                )
                self.extra_upsamples.append(extra_upsample_dim)

            # Append downsampling blocks
            no_identity_needed = (
                self.resize_method == ResizeMethod.CYLIN_FRAC_INTERPOLATE
            )
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, cond_emb_dim=cond_dim),
                        block_klass(dim_out, dim_out, cond_emb_dim=cond_dim),
                        Downsample(
                            dim_out,
                            resize_method=self.resize_method,
                            compress_Z=compress_Z,
                            compress=self.compress,
                            num_of_samples=len(in_out),
                        )
                        if not is_last or no_identity_needed
                        else nn.Identity(),
                    ]
                )
            )
            if self.block_attn:
                self.downs_attn.append(
                    Residual(
                        PreNorm(
                            dim_out, LinearAttention(dim_out, cylindrical=cylindrical)
                        )
                    )
                )

        # Middle (bottleneck) layers
        mid_dim = layer_sizes[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, cond_emb_dim=cond_dim)
        if self.mid_attn:
            self.mid_attn = Residual(
                PreNorm(mid_dim, LinearAttention(mid_dim, cylindrical=cylindrical))
            )
        self.mid_block2 = block_klass(mid_dim, mid_dim, cond_emb_dim=cond_dim)

        # Build the upsampling layers
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            if not is_last:
                extra_upsample = self.extra_upsamples.pop()

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out, dim_in, cond_emb_dim=cond_dim),
                        block_klass(dim_in, dim_in, cond_emb_dim=cond_dim),
                        Upsample(
                            dim_in,
                            extra_upsample,
                            self.resize_method,
                            compress_Z=compress_Z,
                            compress=self.compress,
                            num_of_samples=len(in_out),
                        )
                        if not is_last or no_identity_needed
                        else nn.Identity(),
                    ]
                )
            )
            if self.block_attn:
                self.ups_attn.append(
                    Residual(
                        PreNorm(
                            dim_in, LinearAttention(dim_in, cylindrical=cylindrical)
                        )
                    )
                )

        # Final convolution layer
        if not cylindrical:
            final_lay = nn.Conv3d(layer_sizes[0], out_dim, 1)
        else:
            final_lay = CylindricalConv(layer_sizes[0], out_dim, 1)
        self.final_conv = nn.Sequential(
            block_klass(layer_sizes[1], layer_sizes[0]), final_lay
        )

    def forward(self, x, cond, time):
        """Class function that performs the full forward pass the both encodes the original data, passes it through a linear attention,
        and decodes the encoded data to reconstruct it back to its original shape

        Parameters:
        - x (torch.tensor): data to be encoded
        - cond (torch.tensor): energies
        - time (torch.tensor): time embedding

        Returns:
        - torch.Tensor: Reconstructed data tensor
        """
        # Generate energy embeddings
        # print(f"Start: {x.shape}", flush=True)
        conditions = self.cond_mlp(cond)
        x = self.init_conv(x)  # Convolution
        # print(f"Initial Conv: {x.shape}", flush=True)

        # Downsample
        for i, (block1, block2, downsample) in enumerate(self.downs):
            x = block1(x, conditions)
            x = block2(x, conditions)
            if self.block_attn:
                x = self.downs_attn[i](x)
            x = downsample(x)
            # print(f"Downsample {i + 1}: {x.shape}", flush=True)

        # Bottleneck
        x = self.mid_block1(x, conditions)
        # print(f"Bottleneck 1: {x.shape}", flush=True)
        if self.mid_attn:
            x = self.mid_attn(x)
            # print(f"Bottleneck Attention: {x.shape}", flush=True)
        x = self.mid_block2(x, conditions)
        # print(f"Bottleneck 2: {x.shape}", flush=True)

        # Upsample
        for i, (block1, block2, upsample) in enumerate(self.ups):
            x = block1(x, conditions)
            x = block2(x, conditions)
            if self.block_attn:
                x = self.ups_attn[i](x)
            x = upsample(x)
            # print(f"Upsample {i + 1}: {x.shape}", flush=True)

        x = self.final_conv(x)
        # print(f"Final Conv: {x.shape}", flush=True)
        return x

    def encode(self, x, cond):
        """Class function that performs only the encoding step in the conditional u-net to transform original data into a lower
        dimensional space to generated a latent space

        Parameters:
        - x (torch.tensor): data to be encoded
        - cond (torch.tensor): energies

        Returns:
        - torch.Tensor: Encoded data (latent space representation)
        """
        conditions = self.cond_mlp(cond)

        x = self.init_conv(x)  # Initial convolution

        # downsample
        for i, (block1, block2, downsample) in enumerate(self.downs):
            x = block1(x, conditions)
            x = block2(x, conditions)
            if self.block_attn:
                x = self.downs_attn[i](x)
            x = downsample(x)

        # bottleneck convolution and attention
        x = self.mid_block1(x, conditions)
        if self.mid_attn:
            x = self.mid_attn(x)

        return x  # Return latent representation

    def decode(self, x, cond):
        """Class function that performs only the decoding step in the conditional u-net to transform encoded data into its original dimension size
        or otherwise transforming latent space shape into original shape

        Parameters:
        - x (torch.tensor): Encoded data tensor (latent space).
        - cond (torch.tensor): energies

        Returns:
        - torch.Tensor: Decoded data tensor
        """
        conditions = self.cond_mlp(cond)

        # bottleneck attention and convolution
        if self.mid_attn:
            x = self.mid_attn(x)
        x = self.mid_block2(x, conditions)

        # upsample
        for i, (block1, block2, upsample) in enumerate(self.ups):
            x = block1(x, conditions)
            x = block2(x, conditions)
            if self.block_attn:
                x = self.ups_attn[i](x)
            x = upsample(x)

        return self.final_conv(x)  # final convolution to reconstruct data
