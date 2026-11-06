# Speed against TA-Lib

TA-Lib 0.6.4 on macOS-26.5.2-arm64-arm-64bit, 10 cores. Each measurement is the median of several calls after a warm-up, then the fastest of 2 process run(s) of each library. Ratio is raptorbt / TA-Lib; below 1.0 is faster. Series rows come from the run with a rayon pool of 1 thread(s); a single-series evaluation does not use the pool, so they are one-thread whatever it is.

## Single series, single thread

| Function | Worst ratio | Where | Verdict |
|---|---|---|---|
| ACOS | 1.21 | n=10,000,000 p=None | PASS 1.5x |
| AD | 1.47 | n=10,000,000 p=None | PASS 1.5x |
| ADD | 1.27 | n=10,000,000 p=None | PASS 1.5x |
| ADOSC | 1.09 | n=1,000,000 p=14 | PASS 1.1x |
| ADX | 0.80 | n=1,000,000 p=14 | PASS 1.1x |
| ADXR | 0.76 | n=10,000,000 p=14 | PASS 1.1x |
| APO | 3.01 | n=1,000,000 p=200 | FAIL |
| AROON | 3.71 | n=10,000,000 p=200 | FAIL |
| AROONOSC | 3.71 | n=10,000,000 p=200 | FAIL |
| ASIN | 1.04 | n=1,000,000 p=None | PASS 1.1x |
| ATAN | 1.11 | n=1,000,000 p=None | PASS 1.5x |
| ATR | 0.35 | n=1,000,000 p=14 | PASS 1.1x |
| AVGPRICE | 1.02 | n=1,000,000 p=None | PASS 1.1x |
| BBANDS | 2.30 | n=1,000,000 p=14 | FAIL |
| BETA | 1.84 | n=1,000,000 p=200 | FAIL |
| BOP | 2.06 | n=1,000,000 p=None | FAIL |
| CCI | 2.43 | n=1,000,000 p=14 | FAIL |
| CDL2CROWS | 1.59 | n=1,000,000 p=None | FAIL |
| CDL3BLACKCROWS | 1.88 | n=10,000,000 p=None | FAIL |
| CDL3INSIDE | 1.69 | n=10,000,000 p=None | FAIL |
| CDL3LINESTRIKE | 1.86 | n=10,000,000 p=None | FAIL |
| CDL3OUTSIDE | 1.67 | n=1,000,000 p=None | FAIL |
| CDL3STARSINSOUTH | 2.01 | n=1,000,000 p=None | FAIL |
| CDL3WHITESOLDIERS | 2.40 | n=1,000,000 p=None | FAIL |
| CDLABANDONEDBABY | 1.71 | n=1,000,000 p=None | FAIL |
| CDLADVANCEBLOCK | 1.95 | n=1,000,000 p=None | FAIL |
| CDLBELTHOLD | 1.60 | n=1,000,000 p=None | FAIL |
| CDLBREAKAWAY | 1.43 | n=10,000,000 p=None | PASS 1.5x |
| CDLCLOSINGMARUBOZU | 1.60 | n=1,000,000 p=None | FAIL |
| CDLCONCEALBABYSWALL | 2.13 | n=10,000,000 p=None | FAIL |
| CDLCOUNTERATTACK | 1.65 | n=10,000,000 p=None | FAIL |
| CDLDARKCLOUDCOVER | 1.58 | n=10,000,000 p=None | FAIL |
| CDLDOJI | 6.67 | n=1,000,000 p=None | FAIL |
| CDLDOJISTAR | 2.03 | n=1,000,000 p=None | FAIL |
| CDLDRAGONFLYDOJI | 1.76 | n=1,000,000 p=None | FAIL |
| CDLENGULFING | 1.71 | n=1,000,000 p=None | FAIL |
| CDLEVENINGDOJISTAR | 2.52 | n=1,000,000 p=None | FAIL |
| CDLEVENINGSTAR | 2.26 | n=1,000,000 p=None | FAIL |
| CDLGAPSIDESIDEWHITE | 1.85 | n=10,000,000 p=None | FAIL |
| CDLGRAVESTONEDOJI | 1.78 | n=1,000,000 p=None | FAIL |
| CDLHAMMER | 1.68 | n=1,000,000 p=None | FAIL |
| CDLHANGINGMAN | 1.63 | n=1,000,000 p=None | FAIL |
| CDLHARAMI | 1.66 | n=1,000,000 p=None | FAIL |
| CDLHARAMICROSS | 1.62 | n=10,000,000 p=None | FAIL |
| CDLHIGHWAVE | 1.43 | n=1,000,000 p=None | PASS 1.5x |
| CDLHIKKAKE | 2.02 | n=1,000,000 p=None | FAIL |
| CDLHIKKAKEMOD | 3.38 | n=10,000,000 p=None | FAIL |
| CDLHOMINGPIGEON | 1.86 | n=10,000,000 p=None | FAIL |
| CDLIDENTICAL3CROWS | 2.21 | n=1,000,000 p=None | FAIL |
| CDLINNECK | 1.71 | n=1,000,000 p=None | FAIL |
| CDLINVERTEDHAMMER | 1.58 | n=10,000,000 p=None | FAIL |
| CDLKICKING | 1.76 | n=1,000,000 p=None | FAIL |
| CDLKICKINGBYLENGTH | 1.79 | n=1,000,000 p=None | FAIL |
| CDLLADDERBOTTOM | 1.98 | n=1,000,000 p=None | FAIL |
| CDLLONGLEGGEDDOJI | 2.03 | n=1,000,000 p=None | FAIL |
| CDLLONGLINE | 1.43 | n=1,000,000 p=None | PASS 1.5x |
| CDLMARUBOZU | 1.62 | n=10,000,000 p=None | FAIL |
| CDLMATCHINGLOW | 1.75 | n=10,000,000 p=None | FAIL |
| CDLMATHOLD | 1.93 | n=10,000,000 p=None | FAIL |
| CDLMORNINGDOJISTAR | 2.47 | n=1,000,000 p=None | FAIL |
| CDLMORNINGSTAR | 2.26 | n=1,000,000 p=None | FAIL |
| CDLONNECK | 1.64 | n=1,000,000 p=None | FAIL |
| CDLPIERCING | 1.54 | n=10,000,000 p=None | FAIL |
| CDLRICKSHAWMAN | 1.78 | n=1,000,000 p=None | FAIL |
| CDLRISEFALL3METHODS | 1.93 | n=1,000,000 p=None | FAIL |
| CDLSEPARATINGLINES | 1.62 | n=10,000,000 p=None | FAIL |
| CDLSHOOTINGSTAR | 1.59 | n=1,000,000 p=None | FAIL |
| CDLSHORTLINE | 1.38 | n=10,000,000 p=None | PASS 1.5x |
| CDLSPINNINGTOP | 1.78 | n=1,000,000 p=None | FAIL |
| CDLSTALLEDPATTERN | 2.09 | n=1,000,000 p=None | FAIL |
| CDLSTICKSANDWICH | 1.99 | n=1,000,000 p=None | FAIL |
| CDLTAKURI | 1.83 | n=1,000,000 p=None | FAIL |
| CDLTASUKIGAP | 2.42 | n=10,000,000 p=None | FAIL |
| CDLTHRUSTING | 1.64 | n=1,000,000 p=None | FAIL |
| CDLTRISTAR | 1.83 | n=1,000,000 p=None | FAIL |
| CDLUNIQUE3RIVER | 2.29 | n=10,000,000 p=None | FAIL |
| CDLUPSIDEGAP2CROWS | 1.66 | n=1,000,000 p=None | FAIL |
| CDLXSIDEGAP3METHODS | 1.39 | n=10,000,000 p=None | PASS 1.5x |
| CEIL | 0.98 | n=10,000,000 p=None | PASS 1.1x |
| CMO | 1.06 | n=10,000,000 p=200 | PASS 1.1x |
| CORREL | 4.63 | n=1,000,000 p=14 | FAIL |
| COS | 1.05 | n=10,000,000 p=None | PASS 1.1x |
| COSH | 1.06 | n=1,000,000 p=None | PASS 1.1x |
| DEMA | 0.79 | n=10,000,000 p=200 | PASS 1.1x |
| DIV | 1.23 | n=10,000,000 p=None | PASS 1.5x |
| DX | 0.68 | n=10,000,000 p=14 | PASS 1.1x |
| EMA | 1.28 | n=1,000,000 p=200 | PASS 1.5x |
| EXP | 1.03 | n=10,000,000 p=None | PASS 1.1x |
| FLOOR | 0.98 | n=10,000,000 p=None | PASS 1.1x |
| HT_DCPERIOD | 1.32 | n=10,000,000 p=None | PASS 1.5x |
| HT_DCPHASE | 1.24 | n=1,000,000 p=None | PASS 1.5x |
| HT_PHASOR | 1.61 | n=10,000,000 p=None | FAIL |
| HT_SINE | 1.14 | n=10,000,000 p=None | PASS 1.5x |
| HT_TRENDLINE | 1.45 | n=10,000,000 p=None | PASS 1.5x |
| HT_TRENDMODE | 1.09 | n=10,000,000 p=None | PASS 1.1x |
| KAMA | 3.12 | n=10,000,000 p=14 | FAIL |
| LINEARREG | 1.81 | n=1,000,000 p=14 | FAIL |
| LINEARREG_ANGLE | 1.15 | n=1,000,000 p=14 | PASS 1.5x |
| LINEARREG_INTERCEPT | 1.93 | n=1,000,000 p=14 | FAIL |
| LINEARREG_SLOPE | 2.08 | n=1,000,000 p=14 | FAIL |
| LN | 1.12 | n=1,000,000 p=None | PASS 1.5x |
| LOG10 | 1.06 | n=1,000,000 p=None | PASS 1.1x |
| MA | 2.95 | n=10,000,000 p=14 | FAIL |
| MACD | 1.81 | n=1,000,000 p=200 | FAIL |
| MACDEXT | 3.12 | n=1,000,000 p=14 | FAIL |
| MACDFIX | 1.78 | n=10,000,000 p=14 | FAIL |
| MAMA | 1.25 | n=10,000,000 p=None | PASS 1.5x |
| MAVP | 15.92 | n=10,000,000 p=14 | FAIL |
| MAX | 2.15 | n=1,000,000 p=200 | FAIL |
| MAXINDEX | 7.34 | n=10,000,000 p=200 | FAIL |
| MEDPRICE | 1.04 | n=1,000,000 p=None | PASS 1.1x |
| MFI | 17.41 | n=10,000,000 p=200 | FAIL |
| MIDPOINT | 0.27 | n=10,000,000 p=14 | PASS 1.1x |
| MIDPRICE | 2.41 | n=1,000,000 p=14 | FAIL |
| MIN | 1.55 | n=10,000,000 p=14 | FAIL |
| MININDEX | 5.87 | n=1,000,000 p=200 | FAIL |
| MINMAX | 2.20 | n=1,000,000 p=14 | FAIL |
| MINMAXINDEX | 7.21 | n=10,000,000 p=200 | FAIL |
| MINUS_DI | 1.11 | n=1,000,000 p=200 | PASS 1.5x |
| MINUS_DM | 1.04 | n=10,000,000 p=200 | PASS 1.1x |
| MOM | 4.18 | n=10,000,000 p=14 | FAIL |
| MULT | 1.25 | n=10,000,000 p=None | PASS 1.5x |
| NATR | 0.97 | n=1,000,000 p=14 | PASS 1.1x |
| OBV | 1.13 | n=10,000,000 p=None | PASS 1.5x |
| PLUS_DI | 1.12 | n=10,000,000 p=14 | PASS 1.5x |
| PLUS_DM | 1.05 | n=1,000,000 p=200 | PASS 1.1x |
| PPO | 2.63 | n=1,000,000 p=200 | FAIL |
| ROC | 2.63 | n=10,000,000 p=200 | FAIL |
| ROCP | 2.26 | n=10,000,000 p=200 | FAIL |
| ROCR | 2.25 | n=1,000,000 p=14 | FAIL |
| ROCR100 | 2.46 | n=10,000,000 p=14 | FAIL |
| RSI | 0.45 | n=1,000,000 p=14 | PASS 1.1x |
| SAR | 1.46 | n=10,000,000 p=None | PASS 1.5x |
| SAREXT | 1.41 | n=10,000,000 p=None | PASS 1.5x |
| SIN | 1.05 | n=10,000,000 p=None | PASS 1.1x |
| SINH | 1.06 | n=10,000,000 p=None | PASS 1.1x |
| SMA | 1.32 | n=10,000,000 p=14 | PASS 1.5x |
| SQRT | 0.66 | n=1,000,000 p=None | PASS 1.1x |
| STDDEV | 3.54 | n=1,000,000 p=14 | FAIL |
| STOCH | 3.04 | n=10,000,000 p=200 | FAIL |
| STOCHF | 3.14 | n=10,000,000 p=200 | FAIL |
| STOCHRSI | 2.34 | n=1,000,000 p=200 | FAIL |
| SUB | 1.22 | n=10,000,000 p=None | PASS 1.5x |
| SUM | 1.87 | n=1,000,000 p=200 | FAIL |
| T3 | 2.55 | n=10,000,000 p=14 | FAIL |
| TAN | 1.12 | n=1,000,000 p=None | PASS 1.5x |
| TANH | 1.07 | n=1,000,000 p=None | PASS 1.1x |
| TEMA | 1.04 | n=1,000,000 p=200 | PASS 1.1x |
| TRANGE | 3.00 | n=1,000,000 p=None | FAIL |
| TRIMA | 2.59 | n=1,000,000 p=200 | FAIL |
| TRIX | 1.60 | n=10,000,000 p=200 | FAIL |
| TSF | 1.91 | n=1,000,000 p=14 | FAIL |
| TYPPRICE | 1.00 | n=10,000,000 p=None | PASS 1.1x |
| ULTOSC | 3.76 | n=10,000,000 p=200 | FAIL |
| VAR | 6.22 | n=1,000,000 p=14 | FAIL |
| WCLPRICE | 1.12 | n=10,000,000 p=None | PASS 1.5x |
| WILLR | 2.34 | n=10,000,000 p=200 | FAIL |
| WMA | 6.70 | n=1,000,000 p=14 | FAIL |

28/158 functions within 1.1x (18%, need 90%); all within 1.5x: False. Speed bar: FAIL

Not registered or not comparable here: 1 (see JSON).

## Cases

| Case | Bars | Python threads | Rayon pool | TA-Lib s | raptorbt s | Ratio |
|---|---|---|---|---|---|---|
| rolling_max_200 | 1,000,000 | 1 | 1 | 0.0046 | 0.0110 | 2.37 |
| shared_nodes | 1,000,000 | 1 | 1 | 0.0606 | 0.0178 | 0.29 |
| instruments_500 | 1,000,000 | 1 | 1 | 0.0010 | 0.0019 | 1.85 |
| instruments_500 | 1,000,000 | 10 | 1 | 0.0024 | 0.0043 | 1.81 |
| rolling_max_200 | 10,000,000 | 1 | 1 | 0.4508 | 0.0679 | 0.15 |
| shared_nodes | 10,000,000 | 1 | 1 | 0.6511 | 0.2575 | 0.40 |
| instruments_500 | 10,000,000 | 1 | 1 | 0.0101 | 0.0149 | 1.47 |
| instruments_500 | 10,000,000 | 10 | 1 | 0.0115 | 0.0068 | 0.58 |
| rolling_max_200 | 1,000,000 | 1 | 10 | 0.0046 | 0.0116 | 2.51 |
| shared_nodes | 1,000,000 | 1 | 10 | 0.0606 | 0.0189 | 0.31 |
| instruments_500 | 1,000,000 | 1 | 10 | 0.0010 | 0.0020 | 1.92 |
| instruments_500 | 1,000,000 | 10 | 10 | 0.0024 | 0.0041 | 1.72 |
| rolling_max_200 | 10,000,000 | 1 | 10 | 0.4508 | 0.0661 | 0.15 |
| shared_nodes | 10,000,000 | 1 | 10 | 0.6511 | 0.2582 | 0.40 |
| instruments_500 | 10,000,000 | 1 | 10 | 0.0101 | 0.0157 | 1.55 |
| instruments_500 | 10,000,000 | 10 | 10 | 0.0115 | 0.0065 | 0.57 |

## Sweep

100 bindings of `sma(close, $n) > sma(close, $m)`: the joint plan against each binding evaluated on its own. No TA-Lib counterpart.

| Bars | Bindings | Rayon pool | Joint s | Independent s | Joint / independent |
|---|---|---|---|---|---|
| 1,000,000 | 100 | 1 | 0.0856 | 0.3207 | 0.27 |
| 1,000,000 | 100 | 10 | 0.0484 | 0.3206 | 0.15 |

The shared-nodes case is one raptorbt program summing 20 terms that share `sma` and `ema`, against TA-Lib recomputing both per term. The instruments case is `sma(close, 20)` over 500 instruments; the rows with more than one Python thread call it from that many threads on both sides (each call is one series on one thread, and does not use the rayon pool). The rows with one Python thread are the ones to hold against the single-series bar.
