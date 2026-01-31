from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch
from torch._higher_order_ops import effects as hop_effects
from torch._higher_order_ops.utils import register_fake
from torch._library.effects import EffectType
from torch._ops import HigherOrderOperator
from torch._prims_common import compute_required_storage_length
import torch.fx.experimental.proxy_tensor
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
from torch.fx.experimental.proxy_tensor import track_tensor_tree
import torch.utils._pytree as pytree

if TYPE_CHECKING:
    from torch._subclasses.functional_tensor import BaseFunctionalizeAPI

    from helion.runtime.kernel import Kernel


def _group_aliased_tensors(
    tensors_to_clone: list[str],
    tensor_args: dict[str, torch.Tensor],
    name_to_group: dict[str, int],
) -> list[list[tuple[str, torch.Tensor]]]:
    """Group tensors by storage aliasing using union-find.

    Tensors are grouped if they share storage (detected via torch._C._is_alias_of).
    Tensors with different Dynamo-time proxy groups are kept separate.

    Returns:
        List of groups, where each group is a list of (name, tensor) tuples.
    """
    if not tensors_to_clone:
        return []

    # Union-find implementation
    parent: dict[str, str] = {name: name for name in tensors_to_clone}

    def find(x: str) -> str:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: str, y: str) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Merge tensors only if they satisfy BOTH conditions:
    # 1. Same Dynamo-time proxy group (same_tensor_groups) - to avoid merging unrelated
    #    tensors that happen to share storage due to Inductor's memory optimization
    # 2. Actually share storage at runtime (torch._C._is_alias_of) - to avoid merging
    #    tensors where Inductor optimized away one of them (e.g., computed only a slice
    #    instead of the full tensor)
    #
    # The Dynamo-time grouping is based on:
    # 1. Proxy identity (kernel(x, x) puts both args in same group)
    # 2. Storage aliasing of fake tensors (kernel(x[:2], x) also groups them)
    for i, name1 in enumerate(tensors_to_clone):
        t1 = tensor_args[name1]
        group1 = name_to_group.get(name1)
        for name2 in tensors_to_clone[i + 1 :]:
            t2 = tensor_args[name2]
            group2 = name_to_group.get(name2)
            # Check if they are in the SAME Dynamo-time group
            same_dynamo_group = (
                group1 is not None and group2 is not None and group1 == group2
            )
            # Only merge if BOTH conditions are satisfied:
            # same Dynamo-time group AND actually share storage at runtime
            # pyrefly: ignore[missing-attribute]
            if same_dynamo_group and torch._C._is_alias_of(t1, t2):
                union(name1, name2)

    # Build groups from union-find result
    groups_dict: dict[str, list[tuple[str, torch.Tensor]]] = {}
    for name in tensors_to_clone:
        root = find(name)
        groups_dict.setdefault(root, []).append((name, tensor_args[name]))

    return list(groups_dict.values())


def _clone_tensors_preserving_aliasing(
    tensor_args: dict[str, torch.Tensor],
    tensors_to_clone: list[str],
    same_tensor_groups: list[list[str]],
) -> dict[str, torch.Tensor]:
    """Clone tensors while preserving aliasing relationships between them.

    When multiple tensors share the same underlying storage (e.g., a view and its base),
    we must clone them in a way that preserves this relationship. Otherwise, mutations
    to one tensor won't be visible through the other.

    Args:
        tensor_args: Dict mapping arg names to tensors
        tensors_to_clone: List of arg names that need to be cloned
        same_tensor_groups: Groups of args that had the same proxy at Dynamo time

    Returns:
        Dict mapping arg names to cloned tensors (preserving aliasing)
    """
    if not tensors_to_clone:
        return {}

    # Build name_to_group mapping for proxy-identity groups
    name_to_group: dict[str, int] = {}
    for group_idx, group in enumerate(same_tensor_groups):
        for name in group:
            name_to_group[name] = group_idx

    # Group tensors by storage aliasing
    alias_groups = _group_aliased_tensors(
        tensors_to_clone, tensor_args, name_to_group
    )

    cloned_tensors: dict[str, torch.Tensor] = {}

    for tensors in alias_groups:
        if len(tensors) == 1:
            # Single tensor, simple clone
            key, val = tensors[0]
            cloned_tensors[key] = val.clone()
        else:
            # Multiple tensors share storage - need to preserve aliasing
            # Find the range of storage elements covered by all tensors
            min_offset = min(t.storage_offset() for _, t in tensors)
            max_end = max(
                compute_required_storage_length(t.shape, t.stride(), t.storage_offset())
                for _, t in tensors
            )
            storage_size = max_end - min_offset

            # Create a 1D tensor covering the needed storage range using as_strided
            first_tensor = tensors[0][1]
            temp_1d = torch.as_strided(
                first_tensor,
                (storage_size,),
                (1,),
                min_offset,
            )
            # Clone the 1D tensor - this clones the storage region we need
            cloned_1d = temp_1d.clone()

            # Recreate each tensor as a view of the cloned storage
            for key, val in tensors:
                # Adjust offset relative to the cloned 1D tensor
                new_offset = val.storage_offset() - min_offset
                cloned_val = torch.as_strided(
                    cloned_1d,
                    val.size(),
                    val.stride(),
                    new_offset,
                )
                cloned_tensors[key] = cloned_val

    return cloned_tensors


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
    same_tensor_groups = cast("list[list[str]]", output_spec.get("same_tensor_groups", []))

    # Clone tensors while preserving aliasing relationships
    cloned_tensors = _clone_tensors_preserving_aliasing(
        tensor_args, tensors_to_clone, same_tensor_groups
    )

    # Build cloned_tensor_args: cloned tensors for those to clone, original for others
    cloned_tensor_args = {
        key: cloned_tensors.get(key, val) for key, val in tensor_args.items()
    }

    kernel_outputs = helion_kernel_wrapper_mutation(
        kernel_idx=kernel_idx,
        constant_args=constant_args,
        tensor_args=cloned_tensor_args,
        output_spec=output_spec,
    )

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
    same_tensor_groups = cast("list[list[str]]", output_spec.get("same_tensor_groups", []))

    # Clone tensors while preserving aliasing relationships
    cloned_tensors = _clone_tensors_preserving_aliasing(
        tensor_args, tensors_to_clone, same_tensor_groups
    )
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


