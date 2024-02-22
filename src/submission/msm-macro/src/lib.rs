use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{quote, TokenStreamExt};
use syn::{
    bracketed,
    parse::{Parse, ParseStream},
    parse_macro_input, parse_quote, Ident, LitInt, Token, Type,
};

struct SplitterSpec {
    name: Ident,
    input: SplitterInputSpec,
    output: LitInt,
}

struct SplitterInputSpec {
    limb_type: Type,
    n_limbs: LitInt,
}

impl Parse for SplitterInputSpec {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let ty = input.parse()?;
        input.parse::<Token![;]>()?;
        let size = input.parse()?;
        Ok(Self {
            limb_type: ty,
            n_limbs: size,
        })
    }
}

impl Parse for SplitterSpec {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name = input.parse()?;
        input.parse::<Token![:]>()?;

        let input_spec;
        let _ = bracketed!(input_spec in input);
        let input_spec = input_spec.parse::<SplitterInputSpec>()?;

        input.parse::<Token![->]>()?;

        let output;
        let _ = bracketed!(output in input);
        let output = output.parse::<LitInt>()?;

        Ok(Self {
            name,
            input: input_spec,
            output,
        })
    }
}

fn get_unsigned_type_width(ty: &Type) -> usize {
    match ty {
        Type::Path(path) if path.qself.is_none() && path.path.segments.len() == 1 => {
            let segment = &path.path.segments[0];
            match &segment.ident.to_string()[..] {
                "u8" => 8usize,
                "u16" => 16usize,
                "u32" => 32usize,
                "u64" => 64usize,
                "u128" => 128usize,
                _ => panic!("Expected unsigned integer type"),
            }
        }
        _ => panic!("Expected unsigned integer type"),
    }
}

#[proc_macro]
pub fn define_msm_scalar_splitter(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as SplitterSpec);

    let window_size = input.output.base10_parse::<u8>().unwrap();
    let output_type = match input.output.suffix() {
        "u8" => parse_quote! { u8 },
        "u16" => parse_quote! { u16 },
        "u32" => parse_quote! { u32 },
        "u64" => parse_quote! { u64 },
        "u128" => parse_quote! { u128 },
        "" => input.input.limb_type.clone(),
        _ => panic!("Unsupported output type"),
    };
    let input_limb_bitwidth = get_unsigned_type_width(&input.input.limb_type);
    let n_input_limbs = input.input.n_limbs.base10_parse::<usize>().unwrap();
    let n_total_input_bits = n_input_limbs * input_limb_bitwidth;
    let n_windows = (n_total_input_bits + window_size as usize - 1) / window_size as usize;

    let mut body = quote! {};
    let input_tok = quote! { input };

    body.append_separated(
        (0..n_windows).rev().map(|i| {
            // The ith window. 0 is the least significant and n_windows - 1 is the most significant.

            // Least significant input bit to contribute to this window.
            let bit_start = i * window_size as usize;
            // Least significant input limb to contribute to this window.
            let limb_start = bit_start / input_limb_bitwidth;
            // One-past-the-most-significant input bit to contribute to this window.
            let bit_end = (i + 1) * window_size as usize;
            // Most significant input limb to contribute to this window.
            let limb_end = ((bit_end - 1) / input_limb_bitwidth).min(n_input_limbs - 1);

            let mut expr = quote! {};
            expr.append_separated(
                (limb_start..=limb_end).map(|j| {
                    // The least significant `unused_part` bits of the `j`-th
                    // limb are not used to construct the ith window.
                    let unused_part = if j == limb_start {
                        bit_start - j * input_limb_bitwidth
                    } else {
                        0
                    };
                    // The least significant `used_part` bits, excluding the
                    // unused part above, are used to construct the ith window.
                    let used_part = if j == limb_end {
                        (bit_end - j * input_limb_bitwidth).min(input_limb_bitwidth)
                    } else {
                        input_limb_bitwidth
                    };
                    // Index of the `j`-th limb in the input. Assumes big-endian.
                    let idx = n_input_limbs - 1 - j;
                    // Shift out the unused part on the right.
                    let shifted = if unused_part == 0 {
                        quote! { #input_tok[#idx] }
                    } else {
                        quote! { (#input_tok[#idx] >> #unused_part) }
                    };
                    // Mask out the unused part on the left.
                    let masked = if used_part == input_limb_bitwidth {
                        shifted
                    } else {
                        let mask = format!(
                            "0x{:x}u{}",
                            (1u64 << (used_part - unused_part)) - 1,
                            input_limb_bitwidth
                        );
                        let mask = LitInt::new(&mask, Span::call_site());
                        quote! { (#shifted & #mask) }
                    };

                    let left_shift = j * input_limb_bitwidth + unused_part - bit_start;
                    if left_shift == 0 {
                        quote! { #masked as #output_type }
                    } else {
                        quote! { ((#masked as #output_type) << #left_shift) }
                    }
                }),
                quote! { | },
            );
            expr
        }),
        quote! { , },
    );

    let input_type = input.input.limb_type;
    let input_size = input.input.n_limbs;
    let name = input.name;
    let expanded = quote! {
        pub(crate) struct #name;

        impl SplitterConstants for #name {
            const WINDOW_SIZE: usize = (#window_size) as usize;
            const N_WINDOWS: usize = #n_windows;
            type Output = #output_type;
        }

        impl #name {
            pub const fn split(#input_tok: &[#input_type; #input_size]) -> [#output_type; #n_windows] {
                [ #body ]
            }
        }
    };

    TokenStream::from(expanded)
}
