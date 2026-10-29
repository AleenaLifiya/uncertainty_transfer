# Source from any working directory: source /path/to/uncertainty_transfer/env.sh
export UT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ISBI="$UT_ROOT/for_ISBI"
export PY="${PY:-python}"
export PYTHONPATH="$UT_ROOT:$ISBI/src:$ISBI/exp_ixi/scripts:$UT_ROOT/IR3C_more_comparisons:$UT_ROOT/IR3C_experiments_V2:$UT_ROOT/X-Diffusion/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$ISBI/cache/mpl"
export TORCH_HOME="$ISBI/cache/torch"
export XDG_CACHE_HOME="$ISBI/cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
