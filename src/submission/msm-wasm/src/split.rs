use msm_macro::define_msm_scalar_splitter;

/// Trait for splitting a multi-limb big scalar into smaller limbs of at most
/// WINDOW_SIZE bits each.

pub trait SplitImpl {
    /// Window size in bits.
    const WINDOW_SIZE: usize;
    /// Number of windows.
    const N_WINDOWS: usize;
    /// Numeric type of each window.
    type Output;

    fn split(scalar: &[u32; 8]) -> [Self::Output; Self::N_WINDOWS];
}

define_msm_scalar_splitter! { Split8:  [u32; 8] -> [ 8u32] }
define_msm_scalar_splitter! { Split9:  [u32; 8] -> [ 9u32] }
define_msm_scalar_splitter! { Split10: [u32; 8] -> [10u32] }
define_msm_scalar_splitter! { Split11: [u32; 8] -> [11u32] }
define_msm_scalar_splitter! { Split12: [u32; 8] -> [12u32] }
define_msm_scalar_splitter! { Split13: [u32; 8] -> [13u32] }
define_msm_scalar_splitter! { Split14: [u32; 8] -> [14u32] }
define_msm_scalar_splitter! { Split15: [u32; 8] -> [15u32] }
define_msm_scalar_splitter! { Split16: [u32; 8] -> [16u32] }
define_msm_scalar_splitter! { Split20: [u32; 8] -> [20u32] }
