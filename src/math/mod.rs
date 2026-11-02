//! Numerics the engine owns rather than borrows from a platform library, so two
//! machines give the same bits. `fsum` is Python's `math.fsum`: a sum taken with it in
//! Python lands on the same float here. The normal distribution is `pricing::norm`.

pub(crate) mod fsum;
