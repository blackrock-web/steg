# Experimental Run Integrity Notice

All benchmark and ablation results generated prior to the 2026-08-31 neural pipeline fix are explicitly marked as **INVALID / SUPERSEDED**.

### Prior Defects Identified & Resolved:
1. **ONNX Protobuf Corruption**: Previous runs silently fell back to heuristic cost mapping ('Heuristic Fallback').
2. **Synthetic Cover Bias**: Synthetic procedural covers lacked photographic multi-band frequency distribution.
3. **Stability Bucket Collision**: Server-side OPAP/EMD distortion had inflated metrics before standardizing STABILITY_BITS=4.

The runs conducted from this timestamp onwards use the bit-exact, retrained `FullLFRINNModel` exported to `cost_map_lfrinn.onnx` and evaluated against the real 500-image photographic dataset in `datasets/covers/`.
