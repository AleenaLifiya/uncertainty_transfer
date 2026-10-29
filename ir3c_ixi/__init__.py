"""
ir3c_ixi
========
`IR3C_IXItiny_08Sept` --- IR3C evaluated zero-shot on IXI-tiny and fine-tuned on it.

    existing frozen IR3C checkpoint
            |
            +-- zero-shot inference on the IXI-tiny TEST split
            |
            +-- fine-tune on the IXI-tiny TRAIN split
                    -> checkpoint selected on the IXI-tiny VALIDATION split
                    -> evaluated on the SAME IXI-tiny TEST split and the SAME masks

Every artefact of this run lives under `runs/IR3C_IXItiny_08Sept/`. Nothing under
`runs/brats_*`, `runs/IR3C_OAI_08Sept/`, `comparisons/`, `image_residual/`, `ir2_*`,
`ir3c_oai/` or `IXI_tiny/` is written to, renamed or deleted. NO downstream task.
"""
