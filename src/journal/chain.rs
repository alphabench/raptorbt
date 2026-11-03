//! The hash chain over a journal's canonical record bytes.
//!
//! `h_0 = sha256(canonical header)`, or the chain head of the run this one
//! continues; `h_n = sha256(hex(h_{n-1}) || canonical_n)`. The previous link
//! is hashed as its 64-character lowercase hex *text*, so a verifier in any
//! language needs only a sha256 and the bytes of each record.

use sha2::{Digest, Sha256};

/// Lowercase hex sha256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// `h_0` of a journal that starts a chain: the hash of its header's canonical
/// bytes.
pub fn chain_start(header_canonical: &[u8]) -> String {
    sha256_hex(header_canonical)
}

/// One link. Pure: the same `prev_hex` and record bytes give the same link.
pub fn chain_step(prev_hex: &str, canonical: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(prev_hex.as_bytes());
    h.update(canonical);
    hex(&h.finalize())
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(DIGITS[(b >> 4) as usize] as char);
        out.push(DIGITS[(b & 15) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_answer_and_chain_order_matters() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let h0 = chain_start(b"{}");
        let a = chain_step(&chain_step(&h0, b"a"), b"b");
        let b = chain_step(&chain_step(&h0, b"b"), b"a");
        assert_ne!(a, b);
        assert_eq!(chain_step(&h0, b"a"), chain_step(&h0, b"a"));
    }
}
