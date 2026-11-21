//! The payloads of the records a node writes once per market input (a tick, a
//! bar, a depth snapshot, a clock move). Each builds its payload as a
//! [`Value`], as every other record does, and can also write the canonical
//! bytes of that value directly, keys in the order the canonical form sorts
//! them, so a journal that keeps no record values does not build one per
//! input. The two must give the same bytes; the tests below hold them to it.

use super::value::{write_str, Value};

pub(crate) trait Payload {
    /// The payload as a value: what a record that is kept holds.
    fn value(&self) -> Value;
    /// Exactly the bytes `self.value().write(out)` writes.
    fn write(&self, out: &mut String);
}

/// `sep`, then `"key":`.
fn key(out: &mut String, sep: char, key: &str) {
    out.push(sep);
    write_str(out, key);
    out.push(':');
}

fn float(out: &mut String, sep: char, name: &str, f: f64) {
    key(out, sep, name);
    Value::Float(f).write(out);
}

/// `push_tick`.
pub(crate) struct Tick<'a> {
    pub symbol: &'a str,
    pub ltp: f64,
    pub bid: f64,
    pub ask: f64,
    pub buy_qty_delta: f64,
    pub sell_qty_delta: f64,
    pub ltq: f64,
    pub bid_qty: f64,
    pub ask_qty: f64,
    pub oi: f64,
}

impl Payload for Tick<'_> {
    fn value(&self) -> Value {
        Value::map([
            ("symbol", Value::str(self.symbol)),
            ("ltp", Value::Float(self.ltp)),
            ("bid", Value::Float(self.bid)),
            ("ask", Value::Float(self.ask)),
            ("buy_qty_delta", Value::Float(self.buy_qty_delta)),
            ("sell_qty_delta", Value::Float(self.sell_qty_delta)),
            ("ltq", Value::Float(self.ltq)),
            ("bid_qty", Value::Float(self.bid_qty)),
            ("ask_qty", Value::Float(self.ask_qty)),
            ("oi", Value::Float(self.oi)),
        ])
    }

    fn write(&self, out: &mut String) {
        float(out, '{', "ask", self.ask);
        float(out, ',', "ask_qty", self.ask_qty);
        float(out, ',', "bid", self.bid);
        float(out, ',', "bid_qty", self.bid_qty);
        float(out, ',', "buy_qty_delta", self.buy_qty_delta);
        float(out, ',', "ltp", self.ltp);
        float(out, ',', "ltq", self.ltq);
        float(out, ',', "oi", self.oi);
        float(out, ',', "sell_qty_delta", self.sell_qty_delta);
        key(out, ',', "symbol");
        write_str(out, self.symbol);
        out.push('}');
    }
}

/// `push_bar`: `timestamp` is the bar's as the caller stamped it.
pub(crate) struct Bar<'a> {
    pub symbol: &'a str,
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub warmup: bool,
}

impl Payload for Bar<'_> {
    fn value(&self) -> Value {
        Value::map([
            ("symbol", Value::str(self.symbol)),
            ("timestamp", Value::Int(self.timestamp)),
            ("open", Value::Float(self.open)),
            ("high", Value::Float(self.high)),
            ("low", Value::Float(self.low)),
            ("close", Value::Float(self.close)),
            ("volume", Value::Float(self.volume)),
            ("warmup", Value::Bool(self.warmup)),
        ])
    }

    fn write(&self, out: &mut String) {
        float(out, '{', "close", self.close);
        float(out, ',', "high", self.high);
        float(out, ',', "low", self.low);
        float(out, ',', "open", self.open);
        key(out, ',', "symbol");
        write_str(out, self.symbol);
        key(out, ',', "timestamp");
        Value::Int(self.timestamp).write(out);
        float(out, ',', "volume", self.volume);
        key(out, ',', "warmup");
        Value::Bool(self.warmup).write(out);
        out.push('}');
    }
}

/// `push_depth`: the levels as given, best first.
pub(crate) struct Depth<'a> {
    pub symbol: &'a str,
    pub bids: &'a [(f64, f64)],
    pub asks: &'a [(f64, f64)],
}

fn side_value(levels: &[(f64, f64)]) -> Value {
    Value::List(
        levels.iter().map(|&(p, q)| Value::List(vec![Value::Float(p), Value::Float(q)])).collect(),
    )
}

fn write_side(out: &mut String, levels: &[(f64, f64)]) {
    out.push('[');
    for (i, &(p, q)) in levels.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('[');
        Value::Float(p).write(out);
        out.push(',');
        Value::Float(q).write(out);
        out.push(']');
    }
    out.push(']');
}

impl Payload for Depth<'_> {
    fn value(&self) -> Value {
        Value::map([
            ("symbol", Value::str(self.symbol)),
            ("bids", side_value(self.bids)),
            ("asks", side_value(self.asks)),
        ])
    }

    fn write(&self, out: &mut String) {
        key(out, '{', "asks");
        write_side(out, self.asks);
        key(out, ',', "bids");
        write_side(out, self.bids);
        key(out, ',', "symbol");
        write_str(out, self.symbol);
        out.push('}');
    }
}

/// `advance_time`.
pub(crate) struct Time {
    pub timestamp: i64,
}

impl Payload for Time {
    fn value(&self) -> Value {
        Value::map([("timestamp", Value::Int(self.timestamp))])
    }

    fn write(&self, out: &mut String) {
        key(out, '{', "timestamp");
        Value::Int(self.timestamp).write(out);
        out.push('}');
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::core::Journal;
    use crate::journal::record::{kind, source, Record};
    use crate::journal::sink::{parse_file, FileSink, Fsync};

    /// splitmix64: a fixed seed gives the same cases on every machine.
    struct Gen(u64);

    impl Gen {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }

        /// Ordinary prices most of the time, and every awkward float: signed
        /// zeros, subnormals, the extremes, NaN, infinities, integers past
        /// 2^53, very large and very small exponents, raw bit patterns.
        fn float(&mut self) -> f64 {
            const SPECIAL: [f64; 22] = [
                0.0,
                -0.0,
                f64::MIN_POSITIVE,
                -f64::MIN_POSITIVE,
                5e-324,
                -5e-324,
                2.225_073_858_507_201e-308,
                f64::MAX,
                f64::MIN,
                f64::EPSILON,
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                9_007_199_254_740_993.0,
                18_446_744_073_709_551_616.0,
                1e21,
                1e22,
                1e-7,
                123_456_789.123_456_79,
                0.1,
                100.5,
                1e300,
            ];
            match self.below(6) {
                0 => SPECIAL[self.below(SPECIAL.len() as u64) as usize],
                1 => f64::from_bits(self.next()),
                2 => (self.next() >> 11) as f64 * if self.below(2) == 0 { 1.0 } else { -1.0 },
                3 => self.below(10_000_000) as f64 / 100.0,
                4 => self.below(1_000_000) as f64,
                _ => {
                    (self.next() as f64 / u64::MAX as f64) * 10f64.powi(self.below(40) as i32 - 20)
                }
            }
        }

        fn int(&mut self) -> i64 {
            match self.below(4) {
                0 => [0, -1, 1, i64::MIN, i64::MAX][self.below(5) as usize],
                1 => self.next() as i64,
                _ => 1_700_000_000_000_000_000 + self.below(1 << 40) as i64,
            }
        }

        fn symbol(&mut self) -> String {
            const PARTS: [&str; 12] = [
                "",
                "X:",
                "ALPHA",
                "SYMBOLA24DEC50000CE",
                "a\"b",
                "back\\slash",
                "\n\t\r",
                "\u{8}\u{c}\u{1}\u{7f}",
                "caf\u{e9}",
                "\u{20ac}",
                "\u{1f600}",
                " ~",
            ];
            (0..self.below(4)).map(|_| PARTS[self.below(PARTS.len() as u64) as usize]).collect()
        }

        fn levels(&mut self) -> Vec<(f64, f64)> {
            (0..self.below(6)).map(|_| (self.float(), self.float())).collect()
        }
    }

    fn record(g: &mut Gen, kind: &str) -> Record {
        Record {
            seq: g.next() >> g.below(64),
            kind: kind.to_string(),
            ts_event: g.int(),
            ts_arrival: g.int(),
            source: source::FEED.to_string(),
            client_order_id: None,
            payload: Value::Null,
        }
    }

    /// The direct bytes of `payload` inside a record, and those of the record
    /// built the generic way, are the same: and so are their links.
    fn same<P: Payload>(g: &mut Gen, kind: &str, payload: &P) {
        let mut direct = String::new();
        payload.write(&mut direct);
        let value = payload.value();
        assert_eq!(direct.as_bytes(), value.to_bytes().as_slice());
        let mut r = record(g, kind);
        let mut fast = String::new();
        r.write_canonical(&mut fast, |out| payload.write(out));
        r.payload = value;
        let generic = r.to_value().to_bytes();
        assert_eq!(fast.as_bytes(), generic.as_slice());
        assert_eq!(r.canonical(), generic);
        let prev = crate::journal::chain_start(&g.next().to_le_bytes());
        assert_eq!(
            crate::journal::chain_step(&prev, fast.as_bytes()),
            crate::journal::chain_step(&prev, &generic)
        );
    }

    #[test]
    fn the_direct_bytes_of_every_market_payload_are_the_generic_bytes() {
        let mut g = Gen(0x5eed_0f1a_2b3c);
        for _ in 0..20_000 {
            let symbol = g.symbol();
            let tick = Tick {
                symbol: &symbol,
                ltp: g.float(),
                bid: g.float(),
                ask: g.float(),
                buy_qty_delta: g.float(),
                sell_qty_delta: g.float(),
                ltq: g.float(),
                bid_qty: g.float(),
                ask_qty: g.float(),
                oi: g.float(),
            };
            same(&mut g, kind::TICK, &tick);
            let bar = Bar {
                symbol: &symbol,
                timestamp: g.int(),
                open: g.float(),
                high: g.float(),
                low: g.float(),
                close: g.float(),
                volume: g.float(),
                warmup: g.below(2) == 0,
            };
            same(&mut g, kind::BAR, &bar);
            let (bids, asks) = (g.levels(), g.levels());
            same(&mut g, kind::DEPTH, &Depth { symbol: &symbol, bids: &bids, asks: &asks });
            let time = Time { timestamp: g.int() };
            same(&mut g, kind::TIME, &time);
        }
    }

    #[test]
    fn a_file_written_the_direct_way_is_the_file_written_the_generic_way() {
        let dir = std::env::temp_dir();
        let path = |name: &str| {
            let p = dir.join(format!("rbt-journal-{}-market-{name}", std::process::id()));
            let _ = std::fs::remove_file(&p);
            p
        };
        let (fast_path, generic_path) = (path("fast"), path("generic"));
        let header = Value::map([("run", Value::str("market"))]);
        let open = |p: &std::path::Path| {
            Box::new(FileSink::open(p, Fsync::Orders, &header, 3, None).unwrap())
        };
        // The direct path is taken only by a journal that keeps no values.
        let mut fast = Journal::new(header.clone(), 3, None);
        fast.add_byte_sink(open(&fast_path));
        let mut generic = Journal::new(header.clone(), 3, None);
        generic.add_sink(open(&generic_path));
        let mut g = Gen(42);
        for i in 0..2_000i64 {
            let symbol = g.symbol();
            let (ts, arr) = (g.int(), g.int());
            let tick = Tick {
                symbol: &symbol,
                ltp: g.float(),
                bid: g.float(),
                ask: g.float(),
                buy_qty_delta: g.float(),
                sell_qty_delta: g.float(),
                ltq: g.float(),
                bid_qty: g.float(),
                ask_qty: g.float(),
                oi: g.float(),
            };
            let bids = g.levels();
            let depth = Depth { symbol: &symbol, bids: &bids, asks: &[] };
            for j in [&mut fast, &mut generic] {
                j.append_payload(kind::TICK, ts, arr, source::FEED, &tick).unwrap();
                j.append_payload(kind::DEPTH, ts, arr, source::FEED, &depth).unwrap();
                j.append_payload(kind::TIME, ts, ts, source::FEED, &Time { timestamp: i }).unwrap();
                // A forced record between market data, as an order would be.
                j.append(kind::COMMAND, ts, arr, source::STRATEGY, Some("s0-1"), Value::Int(i))
                    .unwrap();
            }
        }
        assert_eq!(fast.chain(), generic.chain());
        assert_eq!(fast.count(), generic.count());
        assert_eq!(fast.last_kind(), generic.last_kind());
        let (a, b) = (std::fs::read(&fast_path).unwrap(), std::fs::read(&generic_path).unwrap());
        assert_eq!(a, b);
        let back = parse_file(&a).unwrap();
        assert_eq!(back.records.len(), 8_000);
        assert_eq!(back.records.last().unwrap().chain, fast.chain());
        let _ = std::fs::remove_file(&fast_path);
        let _ = std::fs::remove_file(&generic_path);
    }

    #[test]
    fn a_journal_that_keeps_records_holds_their_values() {
        let mut j = Journal::new(Value::Null, 0, None);
        j.keep_in_memory(None);
        let bids = [(100.5, 2.0)];
        let depth = Depth { symbol: "X", bids: &bids, asks: &[] };
        j.append_payload(kind::DEPTH, 1, 1, source::FEED, &depth).unwrap();
        let kept = j.since(0).unwrap();
        assert_eq!(kept[0].record.payload, depth.value());
        assert_eq!(kept[0].canonical, kept[0].record.canonical());
    }
}
