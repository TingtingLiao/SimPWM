# Environment of the H3 jobs (sourced after scripts/env.sh).
#   H3_PY         python of the H3 environment (pip install -e ".[h3]"; default: PY)
#   H3_DIFFUSERS  MiniMax-H3 Ref2VA in diffusers layout (default: models/MiniMax-H3-diffusers-ref2va)
H3_PY=${H3_PY:-$PY}
export PATH=$(dirname "$(command -v "$H3_PY")"):$PATH
