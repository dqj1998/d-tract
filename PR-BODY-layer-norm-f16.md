## Problem

`LayerNorm`'s `wire` expansion in `onnx/src/ops/nn/layer_norm.rs` casts
`normalized` back to `fact.datum_type` **before** the scale and bias are
applied, then multiplies it with `cast_scale` (which is still in
`self.datum_type`, typically F32).

With F16 inputs this becomes `F16 × F32` at the `normalized_scaled` mul
node. The mul downgrades the output to F32, which then conflicts with
the inference rule

```rust
s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
```

(line 57 — `outputs[0].datum_type == inputs[0].datum_type`, which is
F16). `into_typed()` fails with:

```
Translating node #N "/embeddings/LayerNorm/LayerNormalization" LayerNorm
  ToTypedTranslator
Caused by:
  Output mismatch after rewiring expansion for output #0:
  expected 1,256,384,F16 got 1,256,384,F32
```

## Reproduction

Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Export:
```python
from optimum.exporters.onnx import main_export
main_export(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "./out/",
    task="feature-extraction",
    dtype="fp16",
)
```

Bake fixed I/O shapes (independent of this issue, required for tract
static optimization):
```python
import onnx
from onnx import shape_inference
from onnx.tools import update_model_dims
m = onnx.load("./out/model.onnx")
m = update_model_dims.update_inputs_outputs_dims(m,
    {"input_ids":[1,256], "attention_mask":[1,256], "token_type_ids":[1,256]},
    {"last_hidden_state":[1,256,384]})
while len(m.graph.value_info) > 0:
    m.graph.value_info.pop()
m = shape_inference.infer_shapes(m)
onnx.save(m, "./model_f16.onnx")
```

Load in tract:
```rust
let plan = tract_onnx::onnx()
    .model_for_path("./model_f16.onnx")?
    .with_input_fact(0, InferenceFact::dt_shape(i64::datum_type(), tvec!(1, 256)))?
    .with_input_fact(1, InferenceFact::dt_shape(i64::datum_type(), tvec!(1, 256)))?
    .with_input_fact(2, InferenceFact::dt_shape(i64::datum_type(), tvec!(1, 256)))?
    .into_optimized()?            // ← panics here, before this patch
    .into_runnable()?;
```

## Fix

Defer the cast back to `fact.datum_type` until after the
scale and bias have been applied. The expansion stays entirely in
`self.datum_type` (F32) through `normalized * scale + bias`, then casts
only the final result.

**Behavior for F32 inputs is unchanged**: when
`fact.datum_type == self.datum_type`, the introduced final cast is a
no-op that the optimizer removes.

## Diff summary

```
- let cast_normalized = wire_node(cast(fact.datum_type), &normalized)
- let normalized_scaled = mul(cast_normalized, cast_scale)
- let y = if bias { add(normalized_scaled, bias) } else { normalized_scaled }
+ let normalized_scaled = mul(normalized, cast_scale)
+ let y_internal = if bias { add(normalized_scaled, bias) } else { normalized_scaled }
+ let y = wire_node(cast(fact.datum_type), &y_internal)
```

13 insertions, 8 deletions.

## Verification

After this patch:
- F16 `paraphrase-multilingual-MiniLM-L12-v2`: loads, embeds correctly
  on both English and Chinese queries. Output dim 384, L2 norm 1.000000.
- Latency on Apple M2 release build: ~100 ms p50, ~120 ms p95
  (MAX_SEQ_LEN=256).
- F32 case: smoke test passes with identical numeric output to before
  the patch (final cast collapses to no-op).

Happy to add a unit test in `onnx/src/ops/nn/layer_norm.rs` if you
prefer; pointer to a preferred test fixture would help.
