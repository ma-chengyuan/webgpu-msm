# msm-bin

This is a very thin wrapper to allow the WGSL code to be run outside the browser
(thanks to the Rust `wgpu` library). A standalone binary is easier to profile
(e.g., with other third party tools like Nvidia Nsight)

## Usage

```
cargo run --bin msm-bin --release <power>
```

where `<power>` is between 16 and 20 inclusive.
