from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRocm
from helion._testing import skipIfTileIR
import helion.language as hl

# -----------------------------------------------------------------------------
# Basic Operations (no mutation, return new tensor)
# -----------------------------------------------------------------------------


@helion.kernel(autotune_effort="none")
def k_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(autotune_effort="none")
def k_add_mul(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return both add and mul results."""
    add_out = torch.empty_like(x)
    mul_out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        add_out[tile] = x[tile] + y[tile]
        mul_out[tile] = x[tile] * y[tile]
    return add_out, mul_out


@helion.kernel(autotune_effort="none")
def k_scale_with_scalar_output(
    x: torch.Tensor, scale: float
) -> tuple[torch.Tensor, int]:
    """2D elementwise kernel: returns x * scale and scalar 42."""
    m, n = x.size()
    out = torch.empty_like(x)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :]
        out[tile_m, :] = x_tile * scale

    return out, 42


@helion.kernel(autotune_effort="none")
def k_sum_rows(x: torch.Tensor) -> torch.Tensor:
    """Sum each row of x."""
    m, _ = x.size()
    out = torch.empty([m], device=x.device, dtype=x.dtype)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].to(torch.float32).sum(-1).to(x.dtype)
    return out


@helion.kernel(autotune_effort="none")
def k_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """RMS normalization: returns (out, residual, 42)."""
    m, n = x.size()
    assert weight.size(0) == n
    out = torch.empty_like(x)
    residual = torch.empty_like(x)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        x_squared = x_tile * x_tile
        mean_x_squared = torch.mean(x_squared, dim=-1)
        inv_rms_tile = torch.rsqrt(mean_x_squared + eps)
        normalized = x_tile * inv_rms_tile[:, None]
        result = normalized * weight[:].to(torch.float32)
        out[tile_m, :] = result.to(out.dtype)
        residual[tile_m, :] = normalized.to(out.dtype)

    return out, residual, 42


@helion.kernel(autotune_effort="none")
def k_inline_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Add using inline_triton."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        x_val = x[tile]
        y_val = y[tile]
        result = hl.inline_triton(
            """
            tmp = {lhs} + {rhs}
            tmp
            """,
            args={"lhs": x_val, "rhs": y_val},
            output_like=x_val,
        )
        out[tile] = result
    return out


# -----------------------------------------------------------------------------
# Mutations
# -----------------------------------------------------------------------------


@helion.kernel(autotune_effort="none")
def k_add_inplace(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size()):
        x[tile] = x[tile] + y[tile]
    return x


@helion.kernel(autotune_effort="none")
def k_mutate_both(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    for tile in hl.tile(x.size(0)):
        x[tile], y[tile] = x[tile] + 1, y[tile] * 2
    return x, y


@helion.kernel(autotune_effort="none")
def k_mutate_via_view(x: torch.Tensor) -> torch.Tensor:
    """Create view internally and mutate through it."""
    y = x.view(x.size())
    for tile in hl.tile(y.size()):
        y[tile] = y[tile] + 1
    return x


@helion.kernel(autotune_effort="none")
def k_add_to_both(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Add 1 to x and 2 to y (which may alias)."""
    for tile in hl.tile(x.size()):
        x[tile] = x[tile] + 1
        y[tile] = y[tile] + 2
    return x


@helion.kernel(autotune_effort="none")
def k_store(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size(0)):
        hl.store(x, [tile], y[tile] * 2)
    return x


@helion.kernel(autotune_effort="none")
def k_atomic_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size(0)):
        hl.atomic_add(x, [tile], y[tile])
    return x


@helion.kernel(autotune_effort="none")
def k_mutate_with_out(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        x[tile] = x[tile] + 1
        out[tile] = x[tile] + y[tile]
    return x, out


@helion.kernel(autotune_effort="none")
def k_mutate_return_new(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        x[tile] = x[tile] + 1
        out[tile] = y[tile] * 2
    return out


@helion.kernel(autotune_effort="none")
def k_mutate_two_return_new(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """Mutate both x and y, return a new tensor computed from z."""
    out = torch.empty_like(z)
    for tile in hl.tile(x.size()):
        x[tile] = x[tile] + 1.0
        y[tile] = y[tile] * 2.0
        out[tile] = z[tile] + x[tile] + y[tile]
    return out


# -----------------------------------------------------------------------------
# Pre-allocated/External Output
# -----------------------------------------------------------------------------


@helion.kernel(autotune_effort="none")
def k_add_into_out(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Add x and y, store result in pre-allocated out tensor."""
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(autotune_effort="none")
def k_atomic_add_to_out(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """Atomically add x and y to out."""
    for tile in hl.tile(x.size()):
        hl.atomic_add(out, tile, x[tile])
        hl.atomic_add(out, tile, y[tile])
    return out


# -----------------------------------------------------------------------------
# View/Slice Operations
# -----------------------------------------------------------------------------


@helion.kernel(autotune_effort="none")
def k_slice_mutate(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mutate a slice of x and return that slice (indirect alias)."""
    x_slice = x[:2, :4]
    for tile in hl.tile(x_slice.size()):
        x_slice[tile] = x_slice[tile] + y[tile]
    return x_slice


@helion.kernel(autotune_effort="none")
def k_slice_return_other(
    x_slice: torch.Tensor, y: torch.Tensor, x_full: torch.Tensor
) -> torch.Tensor:
    """Process one slice but return a different slice of the same tensor."""
    for tile in hl.tile(x_slice.size()):
        x_slice[tile] = x_slice[tile] + y[tile]
    return x_full[2:4, 4:8]


@helion.kernel(autotune_effort="none")
def k_mutate_permuted(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mutate 3D tensor and return permuted view."""
    for tile in hl.tile(x.size()):
        x[tile] = x[tile] + y[tile]
    return x.permute(2, 0, 1)


@helion.kernel(autotune_effort="none")
def k_mutate_return_view(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mutate x and return both x and a view of x."""
    for tile in hl.tile(x.size()):
        x[tile] = x[tile] + y[tile]
    return x, x.view(-1)


@helion.kernel(autotune_effort="none")
def k_create_return_view(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Create intermediate tensor, mutate it, return a view."""
    intermediate = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        intermediate[tile] = x[tile] + y[tile]
    return intermediate.view(-1)


# -----------------------------------------------------------------------------
# Special Operations (signal/wait)
# -----------------------------------------------------------------------------


@helion.kernel(autotune_effort="none")
def k_signal(
    signal_pad: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Signal kernel using hl.signal."""
    out = torch.empty_like(x)
    (n,) = x.shape
    for i in hl.grid(n):
        hl.signal(signal_pad, [i], signal=2)
        out[i] = x[i] * 2
    return out, signal_pad


@helion.kernel(autotune_effort="none")
def k_wait_update(
    signal_pad: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wait kernel using hl.wait with update."""
    out = torch.empty_like(x)
    (n,) = x.shape
    for i in hl.grid(n):
        hl.wait(signal_pad, [i], signal=1, update=2)
        out[i] = x[i] * 2
    return out, signal_pad


# =============================================================================
# View and Base Tensor Edge Cases
# =============================================================================


@helion.kernel(autotune_effort="none")
def k_mutate_view_and_read_base(
    x_view: torch.Tensor, y: torch.Tensor, x_base: torch.Tensor
) -> torch.Tensor:
    """Mutate x_view (a view of x_base) and also read from x_base.

    This tests the edge case where a kernel receives both a view and its base
    tensor as separate arguments. The mutation through the view should be
    visible in the base tensor.
    """
    for tile in hl.tile(x_view.size()):
        x_view[tile] = x_view[tile] + y[tile]
    # Return a different slice of the base tensor
    return x_base[2:4, 4:8]


# =============================================================================
# Captured Global Variables
# =============================================================================

CAPTURED_SCALE_FACTOR = 2.5


@helion.kernel(autotune_effort="none")
def k_scale_with_captured(x: torch.Tensor) -> torch.Tensor:
    """Scale x by a captured global variable."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] * CAPTURED_SCALE_FACTOR
    return out


# =============================================================================
# Test Class
# =============================================================================


_ALL_KERNELS = [
    k_add, k_add_mul, k_scale_with_scalar_output, k_sum_rows, k_rms_norm,
    k_inline_add, k_add_inplace, k_mutate_both, k_mutate_via_view, k_add_to_both,
    k_store, k_atomic_add, k_mutate_with_out, k_mutate_return_new,
    k_mutate_two_return_new, k_add_into_out, k_atomic_add_to_out, k_slice_mutate,
    k_slice_return_other, k_mutate_permuted, k_mutate_return_view,
    k_create_return_view, k_signal, k_wait_update, k_scale_with_captured,
    k_mutate_view_and_read_base,
]


class TestTorchCompile(RefEagerTestDisabled, TestCase):
    def setUp(self):
        super().setUp()
        for kernel in _ALL_KERNELS:
            kernel.settings._wip_experimental_allow_torch_compile_fusion = True

    def _run_compile_test(
        self,
        f,
        kernel,
        test_args: tuple,
        warmup_args: tuple | None = None,
        rtol: float | None = None,
        atol: float | None = None,
    ):
        """Run torch.compile test comparing eager vs compiled execution."""
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        kernel.reset()
        _ = kernel(*(warmup_args or test_args))

        # Clone tensor args for expected computation
        expected_args = tuple(
            a.clone() if isinstance(a, torch.Tensor) else a for a in test_args
        )
        expected = f(*expected_args)

        # Clone tensor args for compiled computation
        compiled_args = tuple(
            a.clone() if isinstance(a, torch.Tensor) else a for a in test_args
        )
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(*compiled_args)

        # Verify no graph breaks occurred during compilation
        graph_breaks = torch._dynamo.utils.counters["graph_break"]
        self.assertEqual(len(graph_breaks), 0, f"Graph breaks: {dict(graph_breaks)}")

        # Compare results
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    def _run_clone_then_view_mutate_test(
        self,
        view_fn,
        y_fn=None,
        input_shape=(4, 8),
        warmup_shape=None,
    ):
        """
        Helper for clone-then-view-mutate tests.

        Args:
            view_fn: Function that takes x_clone and returns a view of it.
                     Can be a lambda like `lambda x: x.t()` or `lambda x: torch.positive(x)`.
            y_fn: Optional function to transform y to match the view shape.
                  If None, y is used as-is. Example: `lambda y: y.t()` for transpose.
            input_shape: Shape of input tensors (default (4, 8)).
            warmup_shape: Shape for warmup tensors. If None, uses same as input.
        """
        y_transform = y_fn if y_fn is not None else (lambda y: y)
        warmup_shape = warmup_shape or input_shape

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            x_clone = x.clone()
            x_view = view_fn(x_clone)
            y_view = y_transform(y)
            result = k_add_inplace(x_view, y_view)
            result = torch.relu(result) + 1.0
            return result, x.sum()

        x = torch.randn(*input_shape, device=DEVICE, dtype=torch.float16)
        y = torch.randn(*input_shape, device=DEVICE, dtype=torch.float16)
        warmup = (
            torch.randn(*warmup_shape, device=DEVICE, dtype=torch.float16),
            torch.randn(*warmup_shape, device=DEVICE, dtype=torch.float16),
        )
        self._run_compile_test(f, k_add_inplace, (x, y), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_add_kernel(self):
        """Test: basic addition kernel with prologue/epilogue ops."""
        k_add.settings._wip_experimental_allow_torch_compile_fusion = False

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            result = k_add(x, y)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_add, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_basic_elementwise_kernel(self):
        """Test: multi-input elementwise ops with complex prologue/epilogue."""

        def f(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            z = z * 2.0
            a = x * 2.0
            b = y + z
            result = k_add(a, b)
            result = result * 0.5
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_add, (x, y, z), warmup_args=(x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_multi_input_return_used(self):
        """Test: kernel with multiple inputs that mutates and returns one."""

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_y = y * scale
            result = k_add_inplace(x, scaled_y)
            result = result + 1.0
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_add_inplace, (x, y, scale), warmup_args=(x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_multiple_outputs(self):
        """Test: kernel with multiple outputs."""

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            a = x.abs()
            b = y.neg()
            add_result, mul_result = k_add_mul(a, b)
            add_result = add_result * 2.0
            mul_result = mul_result + 1.0
            add_result = torch.relu(add_result) + 1.0
            mul_result = torch.relu(mul_result) + 1.0
            return add_result, mul_result

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(f, k_add_mul, (x, y), atol=1e-3, rtol=1e-3)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_keyword_arg_styles_all_keyword(self):
        """Test: all keyword argument passing."""

        def f(x, y, z):
            x = x * 2.0
            y = y * 2.0
            z = z * 2.0
            result = k_add(y=y + z, x=x) * 0.5
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_add, (x, y, z), warmup_args=(x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_keyword_arg_styles_mixed(self):
        """Test: mixed positional/keyword argument passing."""

        def f(x, y, z):
            x = x * 2.0
            y = y * 2.0
            z = z * 2.0
            result = k_add(x, y=y + z) - 1.0
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_add, (x, y, z), warmup_args=(x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_default_params(self):
        """Test: kernel with default vs custom parameter values."""

        def f_with_default_scale(
            x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            bias = bias * 2.0
            biased_x = x + bias
            # Inline scaling (default scale=2.0) before kernel
            result = k_add(biased_x, y * 2.0)
            result = result * 0.5
            return torch.relu(result) + 1.0

        def f_with_custom_scale(
            x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            bias = bias * 2.0
            biased_x = x + bias
            # Inline scaling (custom scale=3.0) before kernel
            result = k_add(biased_x, y * 3.0)
            result = result * 0.5
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)

        # Test with default scale
        self._run_compile_test(
            f_with_default_scale,
            k_add,
            (x, y, bias),
            warmup_args=(x, y),
            rtol=1e-3,
            atol=1e-3,
        )

        # Test with custom scale
        self._run_compile_test(
            f_with_custom_scale,
            k_add,
            (x, y, bias),
            warmup_args=(x, y),
            rtol=1e-3,
            atol=1e-3,
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_constant_scalar_args(self):
        """Test: scalar constants in prologue/epilogue operations."""

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            # Apply scale and shift as prologue operations
            a = x * 2.5 + 1.0
            b = y * 2.5 + 1.0
            result = k_add(a, b)
            result = result - 0.5
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f, k_add, (x, y), warmup_args=(x, y), rtol=1e-3, atol=1e-3
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_transposed_input(self):
        """Test: transposed (non-contiguous) tensor input."""

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            a = x.T * scale
            b = y.T
            result = k_add(a, b)
            result = result + 1.0
            return torch.relu(result) + 1.0

        x = torch.randn(8, 4, device=DEVICE, dtype=torch.float16)
        y = torch.randn(8, 4, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        warmup = (
            torch.randn(4, 8, device=DEVICE, dtype=torch.float16),
            torch.randn(4, 8, device=DEVICE, dtype=torch.float16),
        )
        self._run_compile_test(f, k_add, (x, y, scale), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_called_twice(self):
        """Test: same kernel called twice with different inputs."""

        def f(
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, scale: torch.Tensor
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            z = z * 2.0
            scale = scale * 2.0
            scaled_x = x * scale
            a = k_add(scaled_x, y)
            b = k_add(a, z)
            result = b + 1.0
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_add, (x, y, z, scale), warmup_args=(x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_same_tensor_as_two_different_args(self):
        """Test: same tensor passed as two different arguments."""

        def f(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            bias = bias * 2.0
            scaled = x * 2.0 + bias
            result = k_add(scaled, scaled)
            result = result.mean(dim=-1)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_add, (x, bias), rtol=1e-2, atol=1e-2)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_atomic_add_mutation(self):
        """Test: mutation via atomic operations."""

        def f(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            a = x * 0.5
            b = y.abs()
            result = k_atomic_add_to_out(a, b, out)
            result = result + 1.0
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        out = torch.zeros(4, 8, device=DEVICE, dtype=torch.float32)
        warmup = (x, y, torch.zeros_like(x))
        self._run_compile_test(f, k_atomic_add_to_out, (x, y, out), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_indirect_output_alias(self):
        """Test: output is a slice/view of input (indirect alias with different shape)."""

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_y = y * scale
            result = k_slice_mutate(x, scaled_y)
            result = result + 1.0
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)
        warmup = (
            torch.randn(4, 8, device=DEVICE, dtype=torch.float16),
            torch.randn(2, 4, device=DEVICE, dtype=torch.float16),
        )
        self._run_compile_test(f, k_slice_mutate, (x, y, scale), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_empty_tensor(self):
        """Test: tensors with zero-size dimensions."""

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_x = x * scale
            result = k_add(scaled_x, y)
            result = result + 1.0
            return torch.relu(result) + 1.0

        # Test with zero-size first dimension
        x = torch.randn(0, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(0, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(0, 8, device=DEVICE, dtype=torch.float16)
        warmup = (
            torch.randn(4, 8, device=DEVICE, dtype=torch.float16),
            torch.randn(4, 8, device=DEVICE, dtype=torch.float16),
        )
        self._run_compile_test(f, k_add, (x, y, scale), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_reduction_sum(self):
        """Test: kernel with reduction dimension (sum along axis)."""

        def f(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            weight = weight * 2.0
            scaled = x * weight
            # Helion kernel with reduction
            row_sums = k_sum_rows(scaled)
            result = row_sums.softmax(dim=0)
            return torch.relu(result) + 1.0

        x = torch.randn(8, 16, device=DEVICE, dtype=torch.float32)
        weight = torch.randn(8, 16, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f, k_sum_rows, (x, weight), warmup_args=(x,), rtol=1e-3, atol=1e-3
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_inline_triton_mutation(self):
        """Test: kernel using inline_triton marks all inputs as potentially mutated."""

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            a = x.exp()
            b = y.log1p()
            # Helion kernel with inline_triton
            result = k_inline_add(a, b)
            result = result * 2.0
            return torch.relu(result) + 1.0

        x = torch.randn(32, device=DEVICE, dtype=torch.float32).abs()
        y = torch.randn(32, device=DEVICE, dtype=torch.float32).abs()
        self._run_compile_test(f, k_inline_add, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_single_argument_kernel_mutation(self):
        """Test: kernel with mutation on first argument (tests mutation pattern)."""

        def f(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            bias = bias * 2.0
            scaled = (x * 2.0 + bias).contiguous()
            ones = torch.ones_like(scaled)
            # Helion kernel with mutation
            result = k_add_inplace(scaled, ones)
            result = result * 0.5
            return torch.relu(result) + 1.0

        x = torch.randn(8, 8, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(8, 8, device=DEVICE, dtype=torch.float32)
        warmup_x = torch.randn(8, 8, device=DEVICE, dtype=torch.float32)
        warmup = (warmup_x, torch.ones_like(warmup_x))
        self._run_compile_test(f, k_add_inplace, (x, bias), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @skipIfNotCUDA()
    def test_signal_mutation(self):
        """Test: kernel using hl.signal correctly tracks mutation."""

        def f(
            signal_pad: torch.Tensor, x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            processed_x = x * y
            out, sig = k_signal(signal_pad, processed_x)
            out = out + 1.0
            out = torch.relu(out) + 1.0
            return out, sig

        signal_pad = torch.zeros(4, device=DEVICE, dtype=torch.int32)
        warmup_pad = torch.zeros(4, device=DEVICE, dtype=torch.int32)
        x = torch.randn(4, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, device=DEVICE, dtype=torch.float32)
        warmup = (warmup_pad, torch.randn(4, device=DEVICE, dtype=torch.float32))
        self._run_compile_test(f, k_signal, (signal_pad, x, y), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @skipIfNotCUDA()
    def test_wait_mutation(self):
        """Test: kernel using hl.wait correctly tracks mutation."""

        def f(
            signal_pad: torch.Tensor, x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            processed_x = x + y
            out, sig = k_wait_update(signal_pad, processed_x)
            out = out - 0.5
            out = torch.relu(out) + 1.0
            return out, sig

        signal_pad = torch.ones(4, device=DEVICE, dtype=torch.int32)
        warmup_pad = torch.ones(4, device=DEVICE, dtype=torch.int32)
        x = torch.randn(4, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, device=DEVICE, dtype=torch.float32)
        warmup = (warmup_pad, torch.randn(4, device=DEVICE, dtype=torch.float32))
        self._run_compile_test(f, k_wait_update, (signal_pad, x, y), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_view_input_mutate_different_view(self):
        """Test: input is a slice, mutate and return a different slice of the same base."""

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_y = y * scale
            # Pass slice as first arg, but also pass full tensor
            result = k_slice_return_other(x[:2, :4], scaled_y, x)
            result = result + 1.0
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)
        warmup_x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        warmup = (warmup_x[:2, :4], y, warmup_x)
        self._run_compile_test(
            f, k_slice_return_other, (x, y, scale), warmup_args=warmup
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_permute_view(self):
        """Test: output is permuted view of input."""

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_y = y * scale
            result = k_mutate_permuted(x, scaled_y)
            result = result + 1.0
            return torch.relu(result) + 1.0

        x = torch.randn(2, 4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(2, 4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(2, 4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_mutate_permuted, (x, y, scale), warmup_args=(x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_alias_view_as_two_args(self):
        """Test: passing x and aliased view of x as two different arguments."""

        def f(a: torch.Tensor) -> torch.Tensor:
            a = a * 2.0
            x = a * 2
            y = x.view(-1).view(8, 8)  # Reshape to maintain 2D, aliased with x
            result = k_add_inplace(x, y)
            result = result + 1.0
            return torch.relu(result) + 1.0

        a = torch.randn(8, 8, device=DEVICE, dtype=torch.float16)
        warmup = (
            torch.randn(8, 8, device=DEVICE, dtype=torch.float16),
            torch.randn(8, 8, device=DEVICE, dtype=torch.float16),
        )
        self._run_compile_test(f, k_add_inplace, (a,), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_aliasing_inputs_used_after(self):
        """Test: view of input used after kernel mutation."""

        def f(x: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = x.view(-1)  # View before kernel
            ones = torch.ones_like(x)
            _ = k_add_inplace(x, ones)  # Mutate x
            result = y + 1  # Use view after - should see mutation
            return torch.relu(result) + 1.0

        x = torch.randn(8, 8, device=DEVICE, dtype=torch.float16)
        warmup_x = torch.randn(8, 8, device=DEVICE, dtype=torch.float16)
        warmup = (warmup_x, torch.ones_like(warmup_x))
        self._run_compile_test(f, k_add_inplace, (x,), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_through_internal_view(self):
        """Test: kernel that creates a view inside the kernel and mutates through it."""

        def f(x: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            x = x - 1  # Prologue
            result = k_mutate_via_view(x)
            result = result * 2  # Epilogue
            return torch.relu(result) + 1.0

        x = torch.randn(64, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(f, k_mutate_via_view, (x,))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_multiple_mutated_inputs(self):
        """Test: kernel that mutates multiple input tensors independently."""

        def f(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            z = z * 2.0
            result = k_mutate_two_return_new(x, y, z)
            result = result - 1.0
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_mutate_two_return_new, (x, y, z))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_detached_input(self):
        """Test: input is detached from grad-tracking tensor."""

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            y = y * 2.0
            # Detach x before passing to kernel
            x_detached = x.detach()
            result = k_add(x_detached, y)
            result = result * 2.0
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32, requires_grad=True)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(f, k_add, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_module_forward_with_kernel(self):
        """Test: Helion kernel called inside nn.Module.forward()."""

        class SimpleModule(torch.nn.Module):
            def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
                super().__init__()
                self.weight = torch.nn.Parameter(weight)
                self.bias = torch.nn.Parameter(bias)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return k_add(x * self.weight, self.bias)

        def f(module, x):
            x = x * 2.0
            result = module(x)
            return torch.relu(result) + 1.0

        weight = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        module = SimpleModule(weight.clone(), bias.clone())
        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        warmup = (x, x)
        self._run_compile_test(f, k_add, (module, x), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_with_chained_prologue_epilogue_ops(self):
        """Test: mutation with prologue/epilogue operations."""

        def f(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            bias = bias * 2.0
            biased = x + bias
            ones = torch.ones_like(biased)
            result = k_add_inplace(biased, ones)
            result = result + 1.0
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        warmup_x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        warmup = (warmup_x, torch.ones_like(warmup_x))
        self._run_compile_test(f, k_add_inplace, (x, bias), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate(self):
        """Test: clone tensor, mutate clone, verify original unchanged."""

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            x_clone = x.clone()
            result = k_add_inplace(x_clone, y)
            # Apply epilogue only to result, not to x (which we're verifying stayed unchanged)
            result = torch.relu(result) + 1.0
            return result, x

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_add_inplace, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_preallocated_output(self):
        """Test: kernel fills pre-allocated output tensor passed as argument."""

        def f(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            a = x * 2.0
            b = y + 1.0
            result = k_add_into_out(a, b, out)
            result = result * 0.5
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        out = torch.empty(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_add_into_out, (x, y, out))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_multiple_outputs_same_storage(self):
        """Test: multiple outputs that share the same underlying storage."""

        def f(
            x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_y = y * scale
            out_2d, out_1d = k_mutate_return_view(x, scaled_y)
            out_2d = out_2d + 1.0
            out_1d = out_1d * 2.0
            out_2d = torch.relu(out_2d) + 1.0
            out_1d = torch.relu(out_1d) + 1.0
            return out_2d, out_1d

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f, k_mutate_return_view, (x, y, scale), warmup_args=(x, y)
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_aliased_storage_different_shape(self):
        """Test: inputs share storage but have different shapes."""

        def f(base: torch.Tensor) -> torch.Tensor:
            base = base * 2.0
            # Create two views of base with different strides
            x = base[::2]  # Every other element: shape [16]
            y = base[1::2]  # Every other element offset by 1: shape [16]
            result = k_add(x, y)
            result = result + 1.0
            return torch.relu(result) + 1.0

        base = torch.randn(32, device=DEVICE, dtype=torch.float16)
        warmup = (
            torch.randn(16, device=DEVICE, dtype=torch.float16),
            torch.randn(16, device=DEVICE, dtype=torch.float16),
        )
        self._run_compile_test(f, k_add, (base,), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_partial_tensor_mutation(self):
        """Test: mutate only a slice of tensor, rest remains unchanged."""

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            # Take a slice, mutate it, return both slice result and full tensor
            x_slice = x[:2, :4]  # First 2 rows, first 4 cols
            result = k_add_inplace(x_slice, y)
            # Apply epilogue only to result, not to x (which shows mutation pattern)
            result = torch.relu(result) + 1.0
            return result, x  # x should have mutation in slice region

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)
        warmup = (
            torch.randn(2, 4, device=DEVICE, dtype=torch.float16),
            torch.randn(2, 4, device=DEVICE, dtype=torch.float16),
        )
        self._run_compile_test(f, k_add_inplace, (x, y), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_output_aliases_intermediate(self):
        """Test: output aliases tensor created inside the kernel."""

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            a = x * scale
            b = y + 1.0
            result = k_create_return_view(a, b)
            result = result * 2.0
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f, k_create_return_view, (x, y, scale), warmup_args=(x, y)
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_inference_mode(self):
        """Test: kernel works correctly inside inference_mode context."""

        def f(x, y):
            x = x * 2.0
            y = y * 2.0
            z = x + 0.5  # prologue
            result = k_add_inplace(z, y)
            result = result * 2  # epilogue
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)

        with torch.inference_mode():
            self._run_compile_test(f, k_add_inplace, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_identical_aliased_inputs(self):
        """Test: same tensor passed twice as different mutated arguments."""

        def f(z):
            z = z * 2.0
            # Pass same tensor as both x and y
            a = z.clone()
            result = k_add_to_both(a, a)
            # Apply epilogue only to result, not to z (which we're verifying stayed unchanged)
            result = torch.relu(result) + 1.0
            return result, z  # original should be unchanged

        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        warmup = (
            torch.randn(4, 8, device=DEVICE, dtype=torch.float16),
            torch.randn(4, 8, device=DEVICE, dtype=torch.float16),
        )
        self._run_compile_test(f, k_add_to_both, (z,), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_graph_input_is_view_with_kernel(self):
        """Test: graph input is a view, kernel operates on derived view."""

        def f(x, y):
            x = x * 2.0
            y = y * 2.0
            # x is already a view (passed from outside), take a 2D slice
            a = x[:2]  # 2D slice of view
            result = k_add_inplace(a.clone(), y[:2])
            return torch.relu(result) + 1.0

        base = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        x = base[1:]  # view with shape [3, 8]
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        warmup = (
            torch.randn(2, 8, device=DEVICE, dtype=torch.float16),
            torch.randn(2, 8, device=DEVICE, dtype=torch.float16),
        )
        self._run_compile_test(f, k_add_inplace, (x, y), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_return_assigned(self):
        """Test: mutation where return value is assigned to a variable."""

        def fn(x):
            x = x * 2.0
            x = x * 2
            ones = torch.ones_like(x)
            x = k_add_inplace(x, ones)
            result = x + 1
            return torch.relu(result) + 1.0

        x = torch.randn(64, device=DEVICE)
        warmup_x = torch.randn(64, device=DEVICE)
        warmup = (warmup_x, torch.ones_like(warmup_x))
        self._run_compile_test(fn, k_add_inplace, (x,), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_return_discarded(self):
        """Test: mutation where return value is discarded (not assigned)."""

        def fn(x):
            x = x * 2.0
            x = x * 2
            ones = torch.ones_like(x)
            k_add_inplace(x, ones)  # return ignored; still mutates x
            result = x + 1
            return torch.relu(result) + 1.0

        x = torch.randn(64, device=DEVICE)
        warmup_x = torch.randn(64, device=DEVICE)
        warmup = (warmup_x, torch.ones_like(warmup_x))
        self._run_compile_test(fn, k_add_inplace, (x,), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_two_mutated(self):
        """Test: kernel that mutates two inputs and returns both."""

        def fn(x, y):
            x = x * 2.0
            y = y * 2.0
            x, y = x + 1, y - 1
            x, y = k_mutate_both(x, y)
            rx, ry = x * 2, y * 2
            rx = torch.relu(rx) + 1.0
            ry = torch.relu(ry) + 1.0
            return rx, ry

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        self._run_compile_test(fn, k_mutate_both, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_one_mutated(self):
        """Test: kernel that mutates one input."""

        def fn(x, y):
            x = x * 2.0
            y = y * 2.0
            y = y * 2
            x = k_add_inplace(x, y)
            result = x - 1
            return torch.relu(result) + 1.0

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        self._run_compile_test(fn, k_add_inplace, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mut_and_out(self):
        """Test: kernel that mutates input and also returns new tensor."""

        def fn(x, y):
            x = x * 2.0
            y = y * 2.0
            x, y = x + 1, y + 1
            x, out = k_mutate_with_out(x, y)
            x = torch.relu(x) + 1.0
            out = torch.relu(out) + 1.0
            return x, out

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        self._run_compile_test(fn, k_mutate_with_out, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_input_reused_after_call(self):
        """Test: mutated input is used after kernel call, but kernel returns a different tensor."""

        def fn(x, y):
            x = x * 2.0
            y = y * 2.0
            x = x + 1
            out = k_mutate_return_new(x, y)
            rx, rout = x + 1, out  # use mutated input after kernel
            rx = torch.relu(rx) + 1.0
            rout = torch.relu(rout) + 1.0
            return rx, rout

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        self._run_compile_test(fn, k_mutate_return_new, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_store_operation(self):
        """Test hl.store write operation."""

        def fn(x, y):
            x = x * 2.0
            y = y * 2.0
            y = y + 1
            x = k_store(x, y)
            result = x - 1
            return torch.relu(result) + 1.0

        x = torch.zeros(64, device=DEVICE)
        y = torch.randn(64, device=DEVICE)
        warmup_y = torch.randn(64, device=DEVICE)
        warmup = (torch.zeros(64, device=DEVICE), warmup_y)
        self._run_compile_test(fn, k_store, (x, y), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_atomic_add_operation(self):
        """Test hl.atomic_add write operation."""

        def fn(x, y):
            x = x * 2.0
            y = y * 2.0
            y = y * 2
            x = k_atomic_add(x, y)
            result = x + 1
            return torch.relu(result) + 1.0

        x = torch.zeros(64, device=DEVICE)
        y = torch.ones(64, device=DEVICE)
        warmup_y = torch.ones(64, device=DEVICE)
        warmup = (torch.zeros(64, device=DEVICE), warmup_y)
        self._run_compile_test(fn, k_atomic_add, (x, y), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_no_mutation(self):
        """Test: pure function kernel with no input mutations."""

        def fn(x, y):
            x = x * 2.0
            y = y * 2.0
            x, y = x * 2, y * 2
            out = k_add(x, y)
            result = out + 1
            return torch.relu(result) + 1.0

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        self._run_compile_test(fn, k_add, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_basic_prologue_epilogue_tuple(self):
        """Test: prologue/epilogue with tuple (tensor, scalar) output."""
        kernel_scale = 2.0

        def f(x, out_bias):
            # Prologue: ops before kernel
            x_processed = torch.sigmoid(x) * 1.5
            out, info = k_scale_with_scalar_output(x_processed, kernel_scale)
            # Epilogue: ops after kernel
            out_processed = torch.relu(out) + out_bias
            return out_processed, info

        m, n = 64, 128
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        out_bias = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        warmup = (torch.sigmoid(x) * 1.5, kernel_scale)
        self._run_compile_test(
            f,
            k_scale_with_scalar_output,
            (x, out_bias),
            warmup_args=warmup,
            rtol=1e-3,
            atol=1e-3,
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_basic_prologue_epilogue_single(self):
        """Test: prologue/epilogue with single tensor output."""

        def f(x, out_bias):
            # Prologue: ops before kernel
            x_processed = torch.tanh(x) * 2.0
            # Use k_add with processed input added to itself (equivalent to *2)
            out = k_add(x_processed, x_processed)
            # Epilogue: ops after kernel
            return torch.relu(out) + out_bias

        m, n = 64, 128
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        out_bias = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        warmup = (torch.tanh(x) * 2.0, torch.tanh(x) * 2.0)
        self._run_compile_test(
            f, k_add, (x, out_bias), warmup_args=warmup, rtol=1e-3, atol=1e-3
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_prologue_epilogue_chained_ops(self):
        """Test: prologue/epilogue with chained ops on both sides."""
        kernel_scale = 2.0

        def f(x, out_bias, out_scale):
            # Prologue: chained ops before kernel
            x_processed = torch.relu(x) + 0.1
            out, info = k_scale_with_scalar_output(x_processed, kernel_scale)
            # Epilogue: chained ops after kernel
            out_relu = torch.relu(out)
            out_tanh = torch.tanh(out_relu)
            out_biased = out_tanh + out_bias
            out_scaled = out_biased * out_scale
            return out_scaled, info

        m, n = 64, 128
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        out_bias = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        out_scale = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        warmup = (torch.relu(x) + 0.1, kernel_scale)
        self._run_compile_test(
            f,
            k_scale_with_scalar_output,
            (x, out_bias, out_scale),
            warmup_args=warmup,
            rtol=1e-3,
            atol=1e-3,
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_rms_norm_prologue_epilogue(self):
        """Test: prologue/epilogue with multi-output RMS norm kernel."""

        def f(x, weight, out_bias, res_bias):
            # Prologue: ops before kernel
            x_processed = torch.relu(x) + 0.5
            out, residual, info = k_rms_norm(x_processed, weight, 1e-5)
            # Epilogue: ops after kernel (different epilogue per output)
            return torch.relu(out) + out_bias, torch.sigmoid(residual) + res_bias, info

        m, n = 128, 256
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        weight = torch.randn(n, device=DEVICE, dtype=torch.float32)
        out_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
        res_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
        warmup = (torch.relu(x) + 0.5, weight, 1e-5)
        self._run_compile_test(
            f,
            k_rms_norm,
            (x, weight, out_bias, res_bias),
            warmup_args=warmup,
            rtol=1e-3,
            atol=1e-3,
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_transpose_then_view_to_3d_epilogue(self):
        """Test: prologue/epilogue with transpose and reshape view ops."""
        d1, d2, d3 = 8, 16, 32
        kernel_scale = 2.0

        def f(x, epilogue_bias):
            # Prologue view ops: mirror of epilogue (3D->2D then transpose)
            x_3d = x.T.reshape(d3, d1, d2)  # (m, n) -> (n, m) -> (d3, d1, d2)
            x_2d = x_3d.reshape(
                d3, d1 * d2
            ).T  # (d3, d1, d2) -> (d3, m) -> (m, d3) = (m, n)
            out, info = k_scale_with_scalar_output(x_2d, kernel_scale)
            # Epilogue view ops: transpose then 2D->3D
            out_t = out.T
            out_3d = out_t.reshape(d3, d1, d2)
            out_processed = torch.relu(out_3d) + epilogue_bias
            return out_processed, info

        m, n = d1 * d2, d3
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        epilogue_bias = torch.randn(d3, d1, d2, device=DEVICE, dtype=torch.float32)
        warmup = (x, kernel_scale)
        self._run_compile_test(
            f,
            k_scale_with_scalar_output,
            (x, epilogue_bias),
            warmup_args=warmup,
            rtol=1e-3,
            atol=1e-3,
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_fp16_prologue_epilogue_dtype_promotion_simple(self):
        """Test: fp16 kernel with fp32 epilogue (simple dtype promotion)."""
        m, n = 64, 128

        def f(x, y):
            # Prologue: ops before kernel (stays fp16)
            x_processed = torch.relu(x) * 1.2
            # Use k_add with x_processed added to itself (equivalent to *2)
            out = k_add(x_processed, x_processed)
            # Epilogue: simple ops with dtype promotion
            out_sigmoid = torch.sigmoid(out)
            return out_sigmoid + y  # fp16 + fp32 -> fp32

        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
        y = torch.randn(n, device=DEVICE, dtype=torch.float32)
        x_proc = torch.relu(x) * 1.2
        warmup = (x_proc, x_proc)
        self._run_compile_test(
            f, k_add, (x, y), warmup_args=warmup, rtol=1e-3, atol=1e-3
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_fp16_prologue_epilogue_dtype_promotion_chained(self):
        """Test: fp16 kernel with fp32 epilogue (chained dtype promotion)."""
        m, n = 64, 128

        def f(x, y):
            # Prologue: ops before kernel (stays fp16)
            x_processed = torch.sigmoid(x) + 0.1
            # Use k_add with x_processed added to itself (equivalent to *2)
            out = k_add(x_processed, x_processed)
            # Epilogue: chained ops after kernel
            out = torch.sigmoid(out)
            out = torch.relu(out)
            out = torch.tanh(out)
            return out * y  # fp16 * fp32 -> fp32

        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
        y = torch.randn(n, device=DEVICE, dtype=torch.float32)
        x_proc = torch.sigmoid(x) + 0.1
        warmup = (x_proc, x_proc)
        self._run_compile_test(
            f, k_add, (x, y), warmup_args=warmup, rtol=1e-3, atol=1e-3
        )


    # -------------------------------------------------------------------------
    # Extended clone-then-mutate / aliasing edge case tests
    # -------------------------------------------------------------------------

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate_original_twice_in_output(self):
        """Test: same unmutated original appears twice in output tuple.

        This tests that when the same FX node appears multiple times as a graph
        output, all references correctly read from the preserved original buffer.
        """

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            x_clone = x.clone()
            result = k_add_inplace(x_clone, y)
            result = torch.relu(result) + 1.0
            # Return x twice - both should be unchanged
            return result, x, x

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_add_inplace, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate_view_of_original_as_output(self):
        """Test: view of original is output alongside clone-then-mutate.

        This tests that views derived from the original also see the preserved
        pre-mutation value.
        """

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            x_view = x.view(-1)  # view of original
            x_clone = x.clone()
            result = k_add_inplace(x_clone, y)
            result = torch.relu(result) + 1.0
            # Both x and view of x should be unchanged
            return result, x, x_view

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_add_inplace, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate_transform_original(self):
        """Test: original undergoes computation before being output.

        This tests that computations on the original (like x + 1) use the
        pre-mutation value, not the mutated value.
        """

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            x_clone = x.clone()
            result = k_add_inplace(x_clone, y)
            result = torch.relu(result) + 1.0
            # x + 1 should use pre-mutation value of x
            return result, x + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_add_inplace, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate_chained_kernels(self):
        """Test: two kernel calls, each with clone-then-mutate pattern.

        This tests complex graphs with multiple HOPs where each needs
        independent cloning to preserve originals.
        """

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            # First kernel: mutate clone of x
            x_clone1 = x.clone()
            result1 = k_add_inplace(x_clone1, y)
            # Second kernel: mutate the result of first kernel
            ones = torch.ones_like(result1)
            result2 = k_add_inplace(result1, ones)
            result2 = torch.relu(result2) + 1.0
            # Both x and y should be unchanged
            return result2, x, y

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        warmup = (
            torch.randn(4, 8, device=DEVICE, dtype=torch.float16),
            torch.randn(4, 8, device=DEVICE, dtype=torch.float16),
        )
        self._run_compile_test(f, k_add_inplace, (x, y), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_of_view_then_mutate(self):
        """Test: clone a view, mutate the clone, original unchanged.

        This tests that cloning a view and mutating the clone doesn't affect
        the original base tensor.
        """

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            # Create a view, clone it, mutate the clone
            x_view = x.view(-1)
            x_view_clone = x_view.clone()
            result = k_add_inplace(x_view_clone, y.view(-1))
            result = torch.relu(result) + 1.0
            # Original x should be unchanged
            return result, x

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_add_inplace, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_multiple_clones_same_tensor(self):
        """Test: multiple clones of same tensor, each mutated independently.

        This tests that when the same original is cloned multiple times and
        each clone is mutated, the original remains unchanged.
        """

        def f(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x = x * 2.0
            # Create two clones
            x_clone1 = x.clone()
            x_clone2 = x.clone()
            # Mutate first clone
            ones = torch.ones_like(x_clone1)
            result1 = k_add_inplace(x_clone1, ones)
            # Mutate second clone
            twos = ones * 2
            result2 = k_add_inplace(x_clone2, twos)
            result1 = torch.relu(result1) + 1.0
            result2 = torch.relu(result2) + 1.0
            # x should be unchanged
            return result1, result2, x

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        warmup = (
            torch.randn(4, 8, device=DEVICE, dtype=torch.float16),
            torch.randn(4, 8, device=DEVICE, dtype=torch.float16),
        )
        self._run_compile_test(f, k_add_inplace, (x,), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate_transposed(self):
        """Test: clone transposed tensor, mutate clone, original unchanged.

        This tests non-contiguous tensor handling in the clone-then-mutate pattern.
        """

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            # Transpose x, clone the transpose, mutate
            x_t = x.T
            x_t_clone = x_t.clone()
            y_t = y.T
            result = k_add_inplace(x_t_clone, y_t)
            result = torch.relu(result) + 1.0
            # Original x should be unchanged (not transposed)
            return result, x

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_add_inplace, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate_with_inplace_epilogue(self):
        """Test: in-place PyTorch op on mutated result.

        This tests interaction between Helion mutation and PyTorch in-place ops.
        """

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            x_clone = x.clone()
            result = k_add_inplace(x_clone, y)
            result = result.mul_(2.0)  # in-place PyTorch op
            result = torch.relu(result) + 1.0
            return result, x

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_add_inplace, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate_result_used_twice(self):
        """Test: mutated result is used in multiple computations.

        This tests that the mutated clone can be used multiple times while
        the original remains unchanged.
        """

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            x_clone = x.clone()
            result = k_add_inplace(x_clone, y)
            # Use result in two different computations
            out1 = torch.relu(result) + 1.0
            out2 = result.sum()
            return out1, out2, x

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_add_inplace, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_identical_aliased_three_args(self):
        """Test: same tensor passed as three different mutated arguments.

        Extension of test_identical_aliased_inputs with three arguments.
        """

        @helion.kernel(autotune_effort="none")
        def k_add_to_three(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            """Add 1 to x, 2 to y, 3 to z (which may all alias)."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + 1
                y[tile] = y[tile] + 2
                z[tile] = z[tile] + 3
            return x

        k_add_to_three.settings._wip_experimental_allow_torch_compile_fusion = True

        def f(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            w = w * 2.0
            a = w.clone()
            # Pass same tensor as all three arguments
            result = k_add_to_three(a, a, a)
            result = torch.relu(result) + 1.0
            return result, w  # w should be unchanged

        w = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        warmup = (
            torch.randn(4, 8, device=DEVICE, dtype=torch.float16),
            torch.randn(4, 8, device=DEVICE, dtype=torch.float16),
            torch.randn(4, 8, device=DEVICE, dtype=torch.float16),
        )
        k_add_to_three.reset()
        _ = k_add_to_three(*warmup)
        self._run_compile_test(f, k_add_to_three, (w,), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate_original_reduction_as_output(self):
        """Test: reduction of original as output alongside mutation.

        This tests that reductions (like sum) on the original use the
        pre-mutation value.
        """

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            x_clone = x.clone()
            result = k_add_inplace(x_clone, y)
            result = torch.relu(result) + 1.0
            # sum of x should use pre-mutation value
            return result, x.sum()

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_add_inplace, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate_both_inputs_as_outputs(self):
        """Test: clone x, mutate clone, return result along with both x and y unchanged.

        This tests that non-mutated inputs (y) are also correctly handled
        when mutated inputs have clone-then-mutate pattern.
        """

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            x_clone = x.clone()
            result = k_add_inplace(x_clone, y)
            result = torch.relu(result) + 1.0
            # Both x and y should be unchanged
            return result, x, y

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_add_inplace, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_multiple_chained_views_mutate(self):
        """Test: clone then many chained view ops, mutate, original unchanged.

        This tests that clone detection correctly traces through multiple
        consecutive view operations: clone -> t -> contiguous -> view -> flatten -> mutate
        """

        @helion.kernel(autotune_effort="none")
        def k_add_inplace_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """1D in-place add."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + y[tile]
            return x

        k_add_inplace_1d.settings._wip_experimental_allow_torch_compile_fusion = True

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            # Clone then multiple chained views
            x_clone = x.clone()
            x_t = x_clone.t()  # (8, 4)
            x_contig = x_t.contiguous()  # Makes a copy since t() is non-contiguous!
            x_view = x_contig.view(32)  # (32,)
            y_flat = y.flatten()
            result = k_add_inplace_1d(x_view, y_flat)
            result = torch.relu(result) + 1.0
            # x.sum() should use pre-mutation value of x
            return result, x.sum()

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        warmup = (x.t().contiguous().view(32).clone(), y.flatten().clone())
        k_add_inplace_1d.reset()
        _ = k_add_inplace_1d(*warmup)
        self._run_compile_test(f, k_add_inplace_1d, (x, y), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_with_multiple_views_one_mutated(self):
        """Test: clone with multiple views, only one is mutated.

        This tests that when a clone has multiple views and only one is mutated,
        the other view correctly sees the mutation (since both views share the
        same clone's storage in eager mode).

        The fix works by detecting whether the original tensor (before clone) has
        direct (non-view) uses in the output. If all uses of the original go through
        views (sibling views of the mutated input), we don't clone at Inductor level,
        allowing the mutation to propagate correctly to sibling views.
        """

        @helion.kernel(autotune_effort="none")
        def k_add_inplace_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """1D in-place add."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + y[tile]
            return x

        k_add_inplace_1d.settings._wip_experimental_allow_torch_compile_fusion = True

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            # Clone then create two different views
            x_clone = x.clone()
            x_flat = x_clone.flatten()  # view 1 - will be mutated
            x_transposed = x_clone.t()  # view 2 - not mutated, used in output
            y_flat = y.flatten()
            result = k_add_inplace_1d(x_flat, y_flat)
            result = torch.relu(result) + 1.0
            # x_transposed should use pre-mutation value (same as x.t() since clone was made)
            return result, x_transposed.sum()

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        warmup = (x.flatten().clone(), y.flatten().clone())
        k_add_inplace_1d.reset()
        _ = k_add_inplace_1d(*warmup)
        self._run_compile_test(f, k_add_inplace_1d, (x, y), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_two_clones_of_same_tensor_both_mutated(self):
        """Test: create two clones of same tensor, pass both to kernel, both mutated.

        This tests that when two independent clones are made from the same tensor
        and both are passed to the kernel as different arguments, the original
        tensor remains unchanged and both clones receive independent mutations.
        """

        @helion.kernel(autotune_effort="none")
        def k_add_two_inplace(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Add 1 to x and 2 to y (mutates both)."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + 1
                y[tile] = y[tile] + 2
            return x + y

        k_add_two_inplace.settings._wip_experimental_allow_torch_compile_fusion = True

        def f(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            # Create two independent clones of x
            clone1 = x.clone()
            clone2 = x.clone()
            # Both clones are mutated
            result = k_add_two_inplace(clone1, clone2)
            result = torch.relu(result) + 1.0
            # x should be unchanged (both mutations happened to clones)
            return result, x.sum()

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        warmup = (
            torch.randn(4, 8, device=DEVICE, dtype=torch.float16),
            torch.randn(4, 8, device=DEVICE, dtype=torch.float16),
        )
        k_add_two_inplace.reset()
        _ = k_add_two_inplace(*warmup)
        self._run_compile_test(f, k_add_two_inplace, (x,), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_passed_to_two_kernels(self):
        """Test: same clone passed to two different kernels in sequence.

        The first kernel mutates the clone, then a second kernel uses it.
        The original tensor should remain unchanged.

        Uses graph-level clone cache to share clones across kernels.
        """

        @helion.kernel(autotune_effort="none")
        def k_add_one(x: torch.Tensor) -> torch.Tensor:
            """Add 1 to x."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + 1
            return x

        @helion.kernel(autotune_effort="none")
        def k_mul_two(x: torch.Tensor) -> torch.Tensor:
            """Multiply x by 2."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] * 2
            return x

        k_add_one.settings._wip_experimental_allow_torch_compile_fusion = True
        k_mul_two.settings._wip_experimental_allow_torch_compile_fusion = True

        def f(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            x_clone = x.clone()
            # First kernel mutates clone
            _ = k_add_one(x_clone)
            # Second kernel mutates same clone
            result = k_mul_two(x_clone)
            result = torch.relu(result) + 1.0
            # x.sum() should use pre-mutation value of x
            return result, x.sum()

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        warmup1 = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        warmup2 = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        k_add_one.reset()
        k_mul_two.reset()
        _ = k_add_one(warmup1)
        _ = k_mul_two(warmup2)
        self._run_compile_test(f, k_add_one, (x,), warmup_args=(warmup1,))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_repeat_mutate(self):
        """Test: clone then repeat (expansion), mutate.

        repeat(1,1) is a no-op that may be optimized away. Clone detection
        traces through no-op repeats to find the underlying clone.
        """

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            # Clone then repeat (repeat creates a new tensor, not a view)
            x_clone = x.clone()
            x_repeated = x_clone.repeat(1, 1)  # Same shape, but new tensor
            result = k_add_inplace(x_repeated, y)
            result = torch.relu(result) + 1.0
            # x.sum() should use pre-mutation value of x
            return result, x.sum()

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        warmup = (x.clone(), y.clone())
        self._run_compile_test(f, k_add_inplace, (x, y), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @unittest.expectedFailure
    def test_dynamic_shapes_basic(self):
        """Test: kernel with dynamic shapes enabled.

        KNOWN LIMITATION: When torch.compile is used with dynamic=True, tensor
        dimensions become symbolic SymInt values. Currently fails with:
            TypeError: unhashable type: non-nested SymInt
        in kernel.py:_get_bound_kernel_cache_key() because it tries to hash
        SymInt values when creating the bound kernel cache key.

        This test documents the limitation for future improvement.
        """

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            result = k_add(x, y)
            return torch.relu(result) + 1.0

        # Warmup with one size
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        k_add.reset()
        warmup_x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        warmup_y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        _ = k_add(warmup_x.clone(), warmup_y.clone())

        # Compile with dynamic shapes
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor", dynamic=True)

        # Test with first size
        x1 = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y1 = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        expected1 = f(x1.clone(), y1.clone())
        actual1 = compiled_f(x1.clone(), y1.clone())
        torch.testing.assert_close(actual1, expected1)

        # Test with different size (should use same compiled graph with symbolic shapes)
        x2 = torch.randn(8, 16, device=DEVICE, dtype=torch.float16)
        y2 = torch.randn(8, 16, device=DEVICE, dtype=torch.float16)
        expected2 = f(x2.clone(), y2.clone())
        actual2 = compiled_f(x2.clone(), y2.clone())
        torch.testing.assert_close(actual2, expected2)


    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_with_no_return(self):
        """Test: kernel with no return statement (pure mutation, returns None).

        This tests that when a kernel only mutates inputs and has no explicit
        return statement, the compilation handles it correctly.
        """

        @helion.kernel(autotune_effort="none")
        def k_mutate_no_return(x: torch.Tensor, y: torch.Tensor) -> None:
            """Mutate x in-place with no return."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + y[tile]

        k_mutate_no_return.settings._wip_experimental_allow_torch_compile_fusion = True

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            k_mutate_no_return(x, y)
            # Use x after mutation
            return torch.relu(x) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        warmup = (
            torch.randn(4, 8, device=DEVICE, dtype=torch.float16),
            torch.randn(4, 8, device=DEVICE, dtype=torch.float16),
        )
        k_mutate_no_return.reset()
        k_mutate_no_return(*warmup)
        self._run_compile_test(f, k_mutate_no_return, (x, y), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_with_optional_tensor_parameter(self):
        """Test: kernel with Optional[torch.Tensor] parameter.

        Verifies that kernels with Optional[torch.Tensor] parameters work correctly
        with torch.compile. The typing import is added dynamically when Optional
        is detected in the generated code.
        """
        from typing import Optional

        @helion.kernel(autotune_effort="none")
        def k_add_optional(
            x: torch.Tensor, y: torch.Tensor, bias: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """Add x + y, optionally adding bias."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                result = x[tile] + y[tile]
                if bias is not None:
                    result = result + bias[tile]
                out[tile] = result
            return out

        k_add_optional.settings._wip_experimental_allow_torch_compile_fusion = True

        def f(x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            bias = bias * 2.0
            result = k_add_optional(x, y, bias)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        warmup = (x.clone(), y.clone(), bias.clone())
        k_add_optional.reset()
        k_add_optional(*warmup)
        self._run_compile_test(f, k_add_optional, (x, y, bias), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_same_kernel_different_shapes(self):
        """Test: same kernel called twice with different input shapes.

        This tests that when the same Helion kernel is called multiple times with
        different input shapes, each instance gets unique inner Triton kernel names.
        Without proper name uniquification, the second inner kernel would overwrite
        the first in the generated code, causing incorrect results.
        """

        @helion.kernel(autotune_effort="none")
        def k_scale(x: torch.Tensor) -> torch.Tensor:
            """Scale x by 2."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 2.0
            return out

        k_scale.settings._wip_experimental_allow_torch_compile_fusion = True

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # Apply same kernel to two tensors of different shapes
            scaled1 = k_scale(x)  # 4x8
            scaled2 = k_scale(y)  # 2x4
            return scaled1.sum() + scaled2.sum()

        # Warmup both shapes separately (kernel takes single tensor)
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        k_scale.reset()
        _ = k_scale(torch.randn(4, 8, device=DEVICE, dtype=torch.float32))
        _ = k_scale(torch.randn(2, 4, device=DEVICE, dtype=torch.float32))

        # Test inputs
        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(2, 4, device=DEVICE, dtype=torch.float32)

        # Expected (eager)
        expected = f(x.clone(), y.clone())

        # Actual (compiled)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x.clone(), y.clone())

        # Verify no graph breaks
        graph_breaks = torch._dynamo.utils.counters["graph_break"]
        self.assertEqual(len(graph_breaks), 0, f"Graph breaks: {dict(graph_breaks)}")

        # Compare results
        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_captured_global_variable(self):
        """Test: kernel using captured global variable from module scope.

        This tests that when a Helion kernel references a global variable defined
        in the module scope (CAPTURED_SCALE_FACTOR), the generated Inductor code
        correctly imports _source_module to resolve the captured variable.
        """

        def f(x: torch.Tensor) -> torch.Tensor:
            return k_scale_with_captured(x)

        # Warmup
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        k_scale_with_captured.reset()
        warmup_x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        _ = k_scale_with_captured(warmup_x)

        # Test input
        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)

        # Expected (eager)
        expected = f(x.clone())

        # Actual (compiled)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x.clone())

        # Verify no graph breaks
        graph_breaks = torch._dynamo.utils.counters["graph_break"]
        self.assertEqual(len(graph_breaks), 0, f"Graph breaks: {dict(graph_breaks)}")

        # Compare results
        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_view_and_base_as_separate_args(self):
        """Test: kernel receives both a view and its base tensor as separate args.

        This tests a subtle edge case where the kernel mutates a view while also
        receiving the base tensor as a separate argument. The mutation through
        the view should be visible in the base tensor after the call.

        Bug scenario: The compiled code might clone the view before mutation,
        breaking the aliasing relationship with the base tensor.
        """

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x_view = x[:2, :4]  # View of x
            # Pass both view and base to kernel - view gets mutated
            result = k_mutate_view_and_read_base(x_view, y, x)
            return result, x  # Return result and base (which should show mutation)

        # Warmup
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        k_mutate_view_and_read_base.reset()
        warmup_x = torch.randn(8, 8, device=DEVICE, dtype=torch.float16)
        warmup_y = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)
        _ = k_mutate_view_and_read_base(warmup_x[:2, :4], warmup_y, warmup_x)

        # Test inputs
        x = torch.randn(8, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)

        # Expected (eager)
        x_eager = x.clone()
        y_eager = y.clone()
        expected = f(x_eager, y_eager)

        # Actual (compiled)
        x_compiled = x.clone()
        y_compiled = y.clone()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x_compiled, y_compiled)

        # Verify no graph breaks
        graph_breaks = torch._dynamo.utils.counters["graph_break"]
        self.assertEqual(len(graph_breaks), 0, f"Graph breaks: {dict(graph_breaks)}")

        # Compare results - this currently fails due to a bug
        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @unittest.expectedFailure  # Known limitation: cross-kernel view mutation propagation
    def test_overlapping_views_both_mutated(self):
        """Test: two overlapping views of the same tensor, both mutated.

        This tests aliasing between overlapping views. When view1 and view2
        overlap, mutating view2 should be visible through view1 since they
        share underlying storage.

        Bug scenario: The compiled code returns result1 (which is view1) with
        only the first mutation applied, missing the second mutation that
        happened through the overlapping view2.

        NOTE: This is a known limitation. When two separate kernel calls mutate
        overlapping views, the mutation from the second call doesn't propagate
        to the first view. This is because Dynamo traces the view operations
        before the kernel calls, so view2 points to the original x, not the
        functionally updated version after the first kernel mutates view1.
        Fixing this would require propagating mutations through base tensors
        at functionalization time.
        """

        def f(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            view1 = x[:3, :]  # First 3 rows
            view2 = x[1:4, :]  # Rows 1-3 (overlaps with view1)
            ones = torch.ones_like(view1)
            # Mutate view1 (+1)
            result1 = k_add_inplace(view1, ones)
            # Mutate view2 (+2) - overlaps with result1
            twos = torch.ones_like(view2) * 2
            result2 = k_add_inplace(view2, twos)
            # result1 should show rows 1-2 with +1+2=+3 effect
            return result1, result2

        # Warmup
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        k_add_inplace.reset()
        warmup_x = torch.randn(5, 4, device=DEVICE, dtype=torch.float16)
        _ = k_add_inplace(
            warmup_x[:3, :], torch.ones(3, 4, device=DEVICE, dtype=torch.float16)
        )

        # Test inputs
        x = torch.randn(5, 4, device=DEVICE, dtype=torch.float16)

        # Expected (eager)
        x_eager = x.clone()
        expected = f(x_eager)

        # Actual (compiled)
        x_compiled = x.clone()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x_compiled)

        # Verify no graph breaks
        graph_breaks = torch._dynamo.utils.counters["graph_break"]
        self.assertEqual(len(graph_breaks), 0, f"Graph breaks: {dict(graph_breaks)}")

        # Compare results - this currently fails due to a bug
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
