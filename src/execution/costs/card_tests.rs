use super::*;

const DOC: &str =
    r#"{"name":"t-2026-09","effective_from":"2026-09-01T00:00:00+05:30","rates":%R%}"#;

fn doc(rates: &str) -> String {
    DOC.replace("%R%", rates)
}

fn one(row: &str) -> String {
    doc(&format!("[{row}]"))
}

const ROW: &str = r#"{"exchange":"NSE","segment":"INTRADAY","underlying":null,
    "brokerage":{"percent":"0.03","cap":"20"},
    "transaction_tax":{"kind":"STT","percent":"0.025","sides":["SELL"]},
    "exchange_turnover_percent":"0.00307","sebi_percent":"0.0001",
    "stamp_duty_percent":"0.003","gst_percent":"18"}"#;

#[test]
fn rfc3339_offsets_and_fractions() {
    let ist = parse_rfc3339_ns("2026-09-01T00:00:00+05:30").unwrap();
    let utc = parse_rfc3339_ns("2026-08-31T18:30:00Z").unwrap();
    assert_eq!(ist, utc);
    assert_eq!(parse_rfc3339_ns("2026-08-31T18:30:00.5Z").unwrap() - utc, 500_000_000);
    assert_eq!(parse_rfc3339_ns("1970-01-01T00:00:00Z").unwrap(), 0);
    assert_eq!(parse_rfc3339_ns("2000-03-01T00:00:00Z").unwrap(), 951_868_800_000_000_000);
    for bad in
        ["2026-09-01", "2026-09-01T00:00:00", "2026-09-01T25:00:00Z", "x", "2026-02-30T00:00:00Z"]
    {
        assert!(parse_rfc3339_ns(bad).is_err(), "{bad}");
    }
}

#[test]
fn a_valid_row_maps_to_contract_note_rates() {
    let card = RateCard::from_json(&one(ROW)).unwrap();
    assert_eq!(card.rows.len(), 1);
    let r = &card.rows[0];
    assert_eq!((r.venue.as_str(), r.segment), ("NSE", FeeSegment::Intraday));
    assert_eq!(r.rates.rounding, Rounding::ContractNote);
    assert_eq!(r.rates.brokerage_per, BrokeragePer::Order);
    // The text is read two places down: the double nearest 0.0003.
    assert_eq!(r.rates.brokerage_rate, 0.0003);
    assert_eq!(r.rates.brokerage_cap, Some(20.0));
    assert_eq!((r.rates.tax_buy_rate, r.rates.tax_sell_rate), (0.0, 0.00025));
    assert_eq!(r.rates.gst_rate, 0.18);
    assert_eq!(r.effective_from, card.effective_ns);
}

#[test]
fn the_checks_refuse_in_order() {
    let cases = [
        (r#"{"rates":[]}"#.to_string(), "no name"),
        (r#"{"name":"x","rates":[1]}"#.to_string(), "effective_from is required"),
        (doc("[]"), "non-empty list"),
        (doc("[1]"), "is not an object"),
        (one(&ROW.replace("INTRADAY", "SWAP")), "unknown segment"),
        (one(&ROW.replace(r#""exchange":"NSE","#, "")), "exchange is required"),
        (
            one(&ROW.replace(r#""percent":"0.03","cap":"20""#, r#""cap":"20""#)),
            "per_order, or percent",
        ),
        (one(&ROW.replace(r#""kind":"STT""#, r#""kind":"VAT""#)), "STT, CTT or NONE"),
        (one(&ROW.replace(r#"["SELL"]"#, "[]")), "needs at least one side"),
        (one(&ROW.replace(r#""18""#, r#""-1""#)), "non-negative"),
        (one(&ROW.replace(r#""18""#, r#""1e2""#)), "not a decimal string"),
        (one(&ROW.replace(r#""underlying":null"#, r#""event":"EXERCISE""#)), "TRADE or SETTLEMENT"),
    ];
    for (text, want) in cases {
        let err = RateCard::from_json(&text).unwrap_err().to_string();
        assert!(err.contains(want), "{want}: {err}");
    }
}

#[test]
fn a_card_round_trips_through_its_document() {
    let dated = FeeSchedule::exchange_presets();
    // The presets take effect on several dates, which one card cannot carry.
    assert!(dated
        .to_card_json("2026-09-01T00:00:00+05:30")
        .unwrap_err()
        .to_string()
        .contains("different dates"));
    let schedule = dated.in_force_at(parse_rfc3339_ns("2026-09-01T00:00:00+05:30").unwrap());
    let text = schedule.to_card_json("2026-09-01T00:00:00+05:30").unwrap();
    let back = RateCard::from_json(&text).unwrap();
    assert_eq!(back.rows.len(), schedule.rows.len());
    for (a, b) in schedule.rows.iter().zip(&back.rows) {
        // Written as percent text and read back, every rate is the same double.
        assert_eq!(a.venue, b.venue);
        assert_eq!(a.segment, b.segment);
        assert_eq!(a.rates, b.rates, "{}", a.label());
    }
}

#[test]
fn percent_text_moves_the_point_by_text() {
    for (x, want) in [
        (0.0003, "0.03"),
        (0.00025, "0.025"),
        (0.0000307, "0.00307"),
        (0.18, "18"),
        (0.0, "0"),
        (0.000001, "0.0001"),
        (1.0, "100"),
    ] {
        assert_eq!(percent_text(x), want);
    }
}

#[test]
fn rows_pick_the_underlying_then_the_date() {
    let rate = |p: &str| ROW.replace(r#""0.00307""#, &format!(r#""{p}""#));
    let agri = rate("0.5").replace(r#""underlying":null"#, r#""underlying":"GOLD""#);
    let card = RateCard::from_json(&doc(&format!("[{},{agri}]", rate("0.1")))).unwrap();
    let s = card.fee_schedule();
    let at = card.effective_ns;
    let general = s.row_for("NSE", FeeSegment::Intraday, Some("X"), at, false).unwrap();
    assert_eq!(general.rates.exchange_rate, 0.001);
    let own = s.row_for("NSE", FeeSegment::Intraday, Some("GOLD"), at, false).unwrap();
    assert_eq!(own.rates.exchange_rate, 0.005);
    let gap = s.row_for("MCX", FeeSegment::Futures, None, at, false).unwrap_err().to_string();
    assert!(gap.contains("no rate for MCX FUTURES"), "{gap}");
    // A fill before the card took effect is priced on its earliest row.
    assert!(s.row_for("NSE", FeeSegment::Intraday, None, at - 1, false).is_ok());
    assert_eq!(card.backdated_fills(&[at - 1, at, at + 1]), 1);
}
