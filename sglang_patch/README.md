# SGLang Domino Patch

This directory contains the small SGLang-side patch needed for public Domino checkpoints that use:

```json
"projector_type": "domino"
```

The patch makes SGLang treat `domino` as the public alias of the existing DFlash v5 rollout path while keeping `causal_v5` as a backward-compatible alias.

The Domino-compatible SGLang branch includes this patch as commit:

```text
8f30fd9e8 feat(dflash): support Domino projector alias
```

The patch file was generated against the local DFlash-enabled SGLang base:

```text
db869c543ac3fc55a208737b25471173bfb04f35
```

Apply from the root of a DFlash-enabled SGLang checkout:

```bash
git apply /path/to/qwen8B-Domino-release/sglang_patch/domino_projector_alias.patch
python -m pip install -e ./python
```

This patch is only for the SGLang server path. The HF benchmark in this release does not require SGLang.
