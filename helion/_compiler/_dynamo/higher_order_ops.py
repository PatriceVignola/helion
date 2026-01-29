from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch
from torch._higher_order_ops import effects as hop_effects
from torch._higher_order_ops.utils import register_fake
from torch._library.effects import EffectType
from torch._ops import HigherOrderOperator
import torch.fx.experimental.proxy_tensor
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
from torch.fx.experimental.proxy_tensor import track_tensor_tree
import torch.utils._pytree as pytree

if TYPE_CHECKING:
    from torch._subclasses.functional_tensor import BaseFunctionalizeAPI

    from helion.runtime.kernel import Kernel


class HelionKernelWrapperMutation(HigherOrderOperator):
    """HOP that wraps a Helion kernel call, deferring compilation to codegen."""

    def __init__(self) -> None:
        super().__init__("helion_kernel_wrapper_mutation", cacheable=True)

    def __call__(
        self,
        *,
        kernel_idx: int,
        constant_args: dict[str, object],
        tensor_args: dict[str, object],
        output_spec: dict[str, object],
    ) -> tuple[object, ...]:
        return super().__call__(
            kernel_idx=kernel_idx,
            constant_args=constant_args,
            tensor_args=tensor_args,
            output_spec=output_spec,
        )


helion_kernel_wrapper_mutation = HelionKernelWrapperMutation()
hop_effects._register_effectful_op(helion_kernel_wrapper_mutation, EffectType.ORDERED)


class HelionKernelWrapperFunctional(HigherOrderOperator):
    """Functional version of Helion kernel wrapper.

    This HOP takes a tensors_to_clone parameter, clones specified inputs
    before mutation, and returns both the kernel outputs and the cloned
    tensors (for functionalization to track mutations).

    Returns:
        tuple of (kernel_outputs: tuple, cloned_tensors: dict[str, Tensor])
    """

    def __init__(self) -> None:
        super().__init__("helion_kernel_wrapper_functional", cacheable=True)

    def __call__(
        self,
        *,
        kernel_idx: int,
        constant_args: dict[str, object],
        tensor_args: dict[str, object],
        output_spec: dict[str, object],
        tensors_to_clone: list[str],
    ) -> tuple[tuple[object, ...], dict[str, torch.Tensor]]:
        return super().__call__(
            kernel_idx=kernel_idx,
            constant_args=constant_args,
            tensor_args=tensor_args,
            output_spec=output_spec,
            tensors_to_clone=tensors_to_clone,
        )


helion_kernel_wrapper_functional = HelionKernelWrapperFunctional()


def get_helion_kernel(kernel_idx: int) -> Kernel:
    from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

    return cast("Kernel", kernel_side_table.get_kernel(kernel_idx))


# =============================================================================
# Mutation HOP dispatch implementations
# =============================================================================


@helion_kernel_wrapper_mutation.py_impl(torch._C.DispatchKey.CompositeExplicitAutograd)
def helion_kernel_wrapper_mutation_dense(
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, torch.Tensor],
    output_spec: dict[str, object],
) -> tuple[torch.Tensor | object, ...]:
    kernel, all_args = get_helion_kernel(kernel_idx), {**constant_args, **tensor_args}
    args = [
        all_args.get(n, p.default)
        for n, p in kernel.signature.parameters.items()
        if n in all_args or p.default is not p.empty
    ]
    result = kernel(*args)
    return (result,) if not isinstance(result, tuple) else result


@register_fake(helion_kernel_wrapper_mutation)
def helion_kernel_wrapper_mutation_fake(
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, torch.Tensor],
    output_spec: dict[str, object],
) -> tuple[torch.Tensor | object, ...]:
    # Create output tensors/scalars from spec
    results: list[torch.Tensor | object] = []
    for spec in cast(
        "list[dict[str, object] | None]", output_spec.get("output_specs", [])
    ):
        if spec is None:
            results.append(None)
        elif "scalar_value" in spec:
            results.append(spec["scalar_value"])
        else:
            results.append(
                torch.empty(  # pyrefly: ignore[no-matching-overload]
                    spec["shape"], dtype=spec["dtype"], device=spec["device"]
                )
            )
    return tuple(results)


@helion_kernel_wrapper_mutation.py_impl(
    torch.fx.experimental.proxy_tensor.ProxyTorchDispatchMode
)
def helion_kernel_wrapper_mutation_proxy(
    mode: ProxyTorchDispatchMode,
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, torch.Tensor],
    output_spec: dict[str, object],
) -> tuple[torch.Tensor | object, ...]:
    with disable_proxy_modes_tracing():
        out = helion_kernel_wrapper_mutation(
            kernel_idx=kernel_idx,
            constant_args=constant_args,
            tensor_args=tensor_args,  # pyrefly: ignore[bad-argument-type]
            output_spec=output_spec,
        )
    # pyrefly: ignore[missing-attribute]
    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, tensor_args)
    out_proxy = mode.tracer.create_proxy(
        "call_function",
        helion_kernel_wrapper_mutation,
        (),
        {
            "kernel_idx": kernel_idx,
            "constant_args": constant_args,
            "tensor_args": proxy_args,
            "output_spec": output_spec,
        },
        name="helion_kernel_wrapper_mutation",
    )
    return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)


@helion_kernel_wrapper_mutation.py_functionalize_impl
def helion_kernel_wrapper_mutation_functionalize(
    ctx: BaseFunctionalizeAPI,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, torch.Tensor],
    output_spec: dict[str, object],
) -> tuple[torch.Tensor | object, ...]:
    """Convert mutation HOP to functional HOP during functionalization.

    This implements the two-HOP pattern from PyTorch's triton_kernel_wrap:
    1. Identify mutated inputs from output_spec
    2. Call the functional HOP which clones those inputs and runs the kernel
    3. Use ctx.replace() to propagate mutations back through functionalization
    """
    # pyrefly: ignore[bad-argument-type]
    unwrapped_tensor_args = ctx.unwrap_tensors(tensor_args)

    # Get mutated inputs from output_spec (already computed at Dynamo level)
    mutated_inputs = cast("list[str]", output_spec.get("mutated_inputs", []))

    # Clone ALL mutated inputs
    tensors_to_clone = list(mutated_inputs)

    with ctx.redispatch_to_next():
        # Call functional HOP which clones inputs, runs kernel, returns both
        kernel_outputs, cloned_tensors = helion_kernel_wrapper_functional(
            kernel_idx=kernel_idx,
            constant_args=constant_args,
            tensor_args=unwrapped_tensor_args,
            output_spec=output_spec,
            tensors_to_clone=tensors_to_clone,
        )

    # Propagate mutations back through functionalization context
    for key, cloned_tensor in cloned_tensors.items():
        if not isinstance(cloned_tensor, torch.Tensor):
            continue
        input_tensor = tensor_args.get(key)
        if not isinstance(input_tensor, torch.Tensor):
            continue

        ctx.replace(input_tensor, cloned_tensor)
        ctx.mark_mutation_hidden_from_autograd(input_tensor)
        ctx.commit_update(input_tensor)
        ctx.sync(input_tensor)

    return ctx.wrap_tensors(kernel_outputs)


# Fallthrough for dispatch keys
for key in [
    torch._C.DispatchKey.PythonDispatcher,
    torch._C.DispatchKey.PythonTLSSnapshot,
    torch._C.DispatchKey.ADInplaceOrView,
    torch._C.DispatchKey.BackendSelect,
    torch._C.DispatchKey.AutocastCPU,
    torch._C.DispatchKey.AutocastCUDA,
    torch._C.DispatchKey.AutogradCUDA,
    torch._C.DispatchKey.AutogradCPU,
]:
    helion_kernel_wrapper_mutation.fallthrough(key)


# =============================================================================
# Functional HOP dispatch implementations
# =============================================================================


@helion_kernel_wrapper_functional.py_impl(
    torch._C.DispatchKey.CompositeExplicitAutograd
)
def helion_kernel_wrapper_functional_dense(
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, torch.Tensor],
    output_spec: dict[str, object],
    tensors_to_clone: list[str],
) -> tuple[tuple[torch.Tensor | object, ...], dict[str, Any]]:
    """Clone specified inputs, call mutation HOP, return kernel outputs and cloned tensors."""
    # Use same_tensor_groups to share clones for args that had the same proxy at Dynamo time.
    # This distinguishes kernel(x, x) (share clone) from kernel(x.clone(), x.clone()) (separate clones).
    same_tensor_groups = cast("list[list[str]]", output_spec.get("same_tensor_groups", []))
    name_to_group: dict[str, int] = {}
    for group_idx, group in enumerate(same_tensor_groups):
        for name in group:
            name_to_group[name] = group_idx

    # Clone using (tensor_id, group_idx) as key. Args in the same group share a clone.
    clone_cache: dict[tuple[int, int], torch.Tensor] = {}
    cloned_tensor_args: dict[str, torch.Tensor] = {}
    unique_counter = 0
    for key, val in tensor_args.items():
        if key in tensors_to_clone:
            tid = id(val)
            group_idx = name_to_group.get(key)
            if group_idx is None:
                unique_counter -= 1
                group_idx = unique_counter
            clone_key = (tid, group_idx)
            if clone_key not in clone_cache:
                # Use .clone() (not clone_preserve_strides) to ensure contiguous storage.
                # For mutation targets, each element must have independent storage to avoid
                # write conflicts (e.g., broadcast tensors with stride 0 would alias writes).
                clone_cache[clone_key] = val.clone()
            cloned_tensor_args[key] = clone_cache[clone_key]
        else:
            cloned_tensor_args[key] = val

    kernel_outputs = helion_kernel_wrapper_mutation(
        kernel_idx=kernel_idx,
        constant_args=constant_args,
        tensor_args=cloned_tensor_args,
        output_spec=output_spec,
    )

    cloned_tensors = {key: cloned_tensor_args[key] for key in tensors_to_clone}
    return (kernel_outputs, cloned_tensors)


@register_fake(helion_kernel_wrapper_functional)
def helion_kernel_wrapper_functional_fake(
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, torch.Tensor],
    output_spec: dict[str, object],
    tensors_to_clone: list[str],
) -> tuple[tuple[torch.Tensor | object, ...], dict[str, Any]]:
    """Create fake outputs and cloned fake tensors."""
    kernel_outputs = helion_kernel_wrapper_mutation_fake(
        kernel_idx=kernel_idx,
        constant_args=constant_args,
        tensor_args=tensor_args,
        output_spec=output_spec,
    )
    # Use same_tensor_groups to share clones for args with same proxy at Dynamo time
    same_tensor_groups = cast("list[list[str]]", output_spec.get("same_tensor_groups", []))
    name_to_group: dict[str, int] = {}
    for group_idx, group in enumerate(same_tensor_groups):
        for name in group:
            name_to_group[name] = group_idx

    clone_cache: dict[tuple[int, int], torch.Tensor] = {}
    cloned_tensors: dict[str, torch.Tensor] = {}
    unique_counter = 0
    for key in tensors_to_clone:
        val = tensor_args[key]
        tid = id(val)
        group_idx = name_to_group.get(key)
        if group_idx is None:
            unique_counter -= 1
            group_idx = unique_counter
        clone_key = (tid, group_idx)
        if clone_key not in clone_cache:
            # Use .clone() (not clone_preserve_strides) to ensure contiguous storage.
            # For mutation targets, each element must have independent storage.
            clone_cache[clone_key] = val.clone()
        cloned_tensors[key] = clone_cache[clone_key]
    return (kernel_outputs, cloned_tensors)


@helion_kernel_wrapper_functional.py_impl(
    torch.fx.experimental.proxy_tensor.ProxyTorchDispatchMode
)
def helion_kernel_wrapper_functional_proxy(
    mode: ProxyTorchDispatchMode,
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, torch.Tensor],
    output_spec: dict[str, object],
    tensors_to_clone: list[str],
) -> tuple[tuple[torch.Tensor | object, ...], dict[str, Any]]:
    """Trace the functional HOP call."""
    with disable_proxy_modes_tracing():
        out = helion_kernel_wrapper_functional(
            kernel_idx=kernel_idx,
            constant_args=constant_args,
            tensor_args=tensor_args,  # pyrefly: ignore[bad-argument-type]
            output_spec=output_spec,
            tensors_to_clone=tensors_to_clone,
        )
    # pyrefly: ignore[missing-attribute]
    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, tensor_args)
    out_proxy = mode.tracer.create_proxy(
        "call_function",
        helion_kernel_wrapper_functional,
        (),
        {
            "kernel_idx": kernel_idx,
            "constant_args": constant_args,
            "tensor_args": proxy_args,
            "output_spec": output_spec,
            "tensors_to_clone": tensors_to_clone,
        },
        name="helion_kernel_wrapper_functional",
    )
    return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)


@helion_kernel_wrapper_functional.py_functionalize_impl
def helion_kernel_wrapper_functional_functionalize(
    ctx: BaseFunctionalizeAPI,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, torch.Tensor],
    output_spec: dict[str, object],
    tensors_to_clone: list[str],
) -> tuple[tuple[torch.Tensor | object, ...], dict[str, Any]]:
    """Simple pass-through for functional HOP - just wrap/unwrap tensors."""
    # pyrefly: ignore[bad-argument-type]
    unwrapped_tensor_args = ctx.unwrap_tensors(tensor_args)
    with ctx.redispatch_to_next():
        kernel_outputs, cloned_tensors = helion_kernel_wrapper_functional(
            kernel_idx=kernel_idx,
            constant_args=constant_args,
            tensor_args=unwrapped_tensor_args,
            output_spec=output_spec,
            tensors_to_clone=tensors_to_clone,
        )
    wrapped_outputs = ctx.wrap_tensors(kernel_outputs)
    wrapped_cloned = ctx.wrap_tensors(cloned_tensors)
    return (wrapped_outputs, wrapped_cloned)  # pyrefly: ignore[bad-return-type]


# Fallthrough for dispatch keys
for key in [
    torch._C.DispatchKey.PythonDispatcher,
    torch._C.DispatchKey.PythonTLSSnapshot,
    torch._C.DispatchKey.ADInplaceOrView,
    torch._C.DispatchKey.BackendSelect,
    torch._C.DispatchKey.AutocastCPU,
    torch._C.DispatchKey.AutocastCUDA,
    torch._C.DispatchKey.AutogradCUDA,
    torch._C.DispatchKey.AutogradCPU,
]:
    helion_kernel_wrapper_functional.fallthrough(key)


