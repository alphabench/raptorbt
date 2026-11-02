//! Dense integer handles and symbol interning.
//!
//! Strategies and readers hold these small `Copy` ids instead of symbols or
//! references; text is resolved only at registration time.

use crate::core::error::RaptorError;
use crate::core::Result;
use crate::runtime::memory::HeapSize;

macro_rules! dense_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(u32);

        impl $name {
            #[inline]
            pub const fn new(index: u32) -> Self {
                Self(index)
            }

            /// Position in the owning dense table.
            #[inline]
            pub const fn index(self) -> usize {
                self.0 as usize
            }

            /// The raw integer.
            #[inline]
            pub const fn raw(self) -> u32 {
                self.0
            }
        }
    };
}

dense_id!(
    /// A registered instrument; ids follow registration order.
    InstrumentId
);
dense_id!(
    /// A registered strategy.
    StrategyId
);
dense_id!(
    /// A registered bar aggregation.
    AggId
);
dense_id!(
    /// A registered indicator.
    IndId
);

/// Symbol interning: insertion order is id order.
///
/// Lookup by text is meant for registration time only; the hot path carries
/// [`InstrumentId`]s.
#[derive(Debug, Default, Clone)]
pub struct InstrumentTable {
    symbols: Vec<Box<str>>,
    /// Ids sorted by symbol text, for binary-search lookup. Keeps the text
    /// stored once.
    by_symbol: Vec<InstrumentId>,
}

impl InstrumentTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// The id of `symbol`, registering it on first sight.
    pub fn intern(&mut self, symbol: &str) -> Result<InstrumentId> {
        match self.search(symbol) {
            Ok(pos) => Ok(self.by_symbol[pos]),
            Err(pos) => {
                let raw = u32::try_from(self.symbols.len()).map_err(|_| {
                    RaptorError::invalid_parameter("instrument table is full (u32 ids)")
                })?;
                let id = InstrumentId::new(raw);
                self.symbols.push(symbol.into());
                self.by_symbol.insert(pos, id);
                Ok(id)
            }
        }
    }

    /// The id of an already registered symbol.
    pub fn get(&self, symbol: &str) -> Option<InstrumentId> {
        self.search(symbol).ok().map(|pos| self.by_symbol[pos])
    }

    /// The symbol behind an id.
    pub fn symbol(&self, id: InstrumentId) -> Option<&str> {
        self.symbols.get(id.index()).map(|s| &**s)
    }

    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }

    fn search(&self, symbol: &str) -> std::result::Result<usize, usize> {
        self.by_symbol.binary_search_by(|id| (*self.symbols[id.index()]).cmp(symbol))
    }
}

impl HeapSize for InstrumentTable {
    fn heap_bytes(&self) -> usize {
        self.symbols.capacity() * std::mem::size_of::<Box<str>>()
            + self.symbols.iter().map(|s| s.len()).sum::<usize>()
            + self.by_symbol.capacity() * std::mem::size_of::<InstrumentId>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_dense_in_insertion_order_and_idempotent() {
        let mut t = InstrumentTable::new();
        let b = t.intern("BBB").unwrap();
        let a = t.intern("AAA").unwrap();
        assert_eq!((b.index(), a.index()), (0, 1));
        assert_eq!(t.intern("BBB").unwrap(), b);
        assert_eq!(t.get("AAA"), Some(a));
        assert_eq!(t.get("CCC"), None);
        assert_eq!(t.symbol(a), Some("AAA"));
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn ids_are_four_bytes_and_ordered() {
        assert_eq!(std::mem::size_of::<InstrumentId>(), 4);
        assert!(StrategyId::new(1) < StrategyId::new(2));
        assert_eq!(AggId::new(7).index(), 7);
        assert_eq!(IndId::new(3).raw(), 3);
    }
}
