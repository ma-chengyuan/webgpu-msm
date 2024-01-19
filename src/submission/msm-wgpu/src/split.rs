use msm_split::define_msm_scalar_splitter;

/// Trait for splitting a multi-limb big scalar into smaller limbs of at most
/// WINDOW_SIZE bits each.
pub(crate) trait SplitterConstants {
    /// Window size in bits.
    const WINDOW_SIZE: usize;
    /// Number of windows.
    const N_WINDOWS: usize;
    /// Numeric type of each window.
    type Output;
}

define_msm_scalar_splitter! { Split16: [u32; 8] -> [16u32]  }
