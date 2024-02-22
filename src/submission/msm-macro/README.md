# msm-macro

This crate provides proc macros used in `msm-wasm`. In particular, it provides
macro to generate code that splits a big integer into smaller ones, as required
by the Pippenger algorithm. See `../msm-wasm/src/split.rs` for its application.

E.g.,

```rust
define_msm_scalar_splitter! { Split16: [u32; 8] -> [16u32] }
```

defines a splitter struct called `Split16` that splits a big-endian 256-bit
integer consisting of 8`u32`s into several 16-bit integers where each integer is
represented as `u32`.
