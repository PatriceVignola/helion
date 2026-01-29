"""Post-grad FX passes for Helion kernel HOPs."""

from __future__ import annotations

import torch
import torch.utils._pytree as pytree
from torch._inductor.pattern_matcher import (
    CallFunctionVarArgs,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)

from .higher_order_ops import (
    helion_kernel_wrapper_functional,
    helion_kernel_wrapper_functional_dense,
)


def decompose_helion_kernel_wrapper_functional(graph: torch.fx.Graph) -> None:
    """Decompose helion_kernel_wrapper_functional into clones + mutation HOP.

    This pass replaces functional HOP nodes with their dense implementation,
    which handles cloning and calls the mutation HOP. This mirrors PyTorch's
    decompose_triton_kernel_wrapper_functional pass.

    Should run after reinplace_inplaceable_ops if reinplace integration exists,
    otherwise runs with whatever tensors_to_clone was set during functionalization.
    """
    graph_pass = PatternMatcherPass()

    @register_graph_pattern(
        CallFunctionVarArgs(helion_kernel_wrapper_functional),
        pass_dict=graph_pass,
    )
    def _(match: Match, *args: object, **kwargs: object) -> None:
        flat_args, spec = pytree.tree_flatten((args, kwargs))

        def decomp(*flat_args: object) -> tuple[object, ...]:
            args, kwargs = pytree.tree_unflatten(flat_args, spec)
            return (helion_kernel_wrapper_functional_dense(*args, **kwargs),)

        match.replace_by_example(decomp, flat_args, run_functional_passes=False)

    graph_pass.apply(graph)

    # Verify decomposition complete
    for _ in graph.find_nodes(
        op="call_function", target=helion_kernel_wrapper_functional
    ):
        raise AssertionError("helion_kernel_wrapper_functional was not decomposed")
