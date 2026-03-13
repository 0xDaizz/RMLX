//! Proc macros for RMLX kernel registration.
//!
//! Provides `#[rmlx_kernel(...)]` to auto-generate kernel registration boilerplate.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, punctuated::Punctuated, ItemFn, Meta, Token};

/// Attribute macro for kernel registration functions.
///
/// # Usage
///
/// ```ignore
/// #[rmlx_kernel(name = "my_kernel")]
/// pub fn register_my_kernel(registry: &KernelRegistry) -> Result<(), KernelError> {
///     // Body is preserved as-is, macro just adds registration metadata
/// }
/// ```
///
/// This is an exploratory prototype. Currently it:
/// 1. Validates the function signature has exactly one parameter
/// 2. Generates a companion `_kernel_metadata()` function returning the kernel name
/// 3. Preserves the original function unchanged
#[proc_macro_attribute]
pub fn rmlx_kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let attrs = parse_macro_input!(attr with Punctuated::<Meta, Token![,]>::parse_terminated);

    let fn_name = &input.sig.ident;

    // Validate: function must have exactly one parameter (the registry).
    if input.sig.inputs.len() != 1 {
        return syn::Error::new_spanned(
            &input.sig,
            "rmlx_kernel: expected exactly one parameter (registry: &KernelRegistry)",
        )
        .to_compile_error()
        .into();
    }

    // Extract kernel name from attributes, fall back to function name.
    let mut kernel_name: Option<String> = None;
    for meta in &attrs {
        if let Meta::NameValue(nv) = meta {
            if nv.path.is_ident("name") {
                if let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(s),
                    ..
                }) = &nv.value
                {
                    kernel_name = Some(s.value());
                }
            }
        }
    }

    let kernel_name_str = kernel_name.unwrap_or_else(|| fn_name.to_string());
    let metadata_fn_name = syn::Ident::new(&format!("{fn_name}_kernel_metadata"), fn_name.span());

    let output = quote! {
        #input

        /// Auto-generated kernel metadata.
        #[doc(hidden)]
        pub fn #metadata_fn_name() -> &'static str {
            #kernel_name_str
        }
    };

    output.into()
}
