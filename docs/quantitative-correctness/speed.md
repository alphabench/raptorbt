# Speed against TA-Lib

The figures below come from one run on a busy machine (arm64), reproduced by `benches/python/talib_compare.py` (the steps are in its header). They are not a quiet-machine benchmark. x86 has not been measured; there `mul_add` without FMA target features may be slower. The single-series bar (within 1.1x of TA-Lib for 90% of functions, 1.5x for all) is not met: 35 of 158 functions are within 1.1x and 56 within 1.5x. MAVP is 15.5x by design (one core per period); the rest of the slow ones are up to 6.4x, from branch misprediction (candlesticks, STOCH, AROON), accuracy bookkeeping at short windows (WMA, VAR, STDDEV, CORREL) and TA-Lib's rescan rule (MAXINDEX). Accuracy and zero `unsafe` were kept by decision over speed.

TA-Lib 0.6.4 on macOS-27.2.1-arm64-arm-64bit, 10 cores. Each measurement is the median of several calls after a warm-up, then the fastest of 2 process run(s) of each library. Ratio is raptorbt / TA-Lib; below 1.0 is faster. Series rows come from the run with a rayon pool of 1 thread(s); a single-series evaluation does not use the pool, so they are one-thread whatever it is.

## Single series, single thread

| Function | Worst ratio | Where | Verdict |
|---|---|---|---|
| ACOS | 1.13 | n=10,000,000 p=None | PASS 1.5x |
| AD | 1.43 | n=10,000,000 p=None | PASS 1.5x |
| ADD | 1.00 | n=10,000,000 p=None | PASS 1.1x |
| ADOSC | 1.07 | n=10,000,000 p=200 | PASS 1.1x |
| ADX | 0.82 | n=1,000,000 p=14 | PASS 1.1x |
| ADXR | 0.77 | n=10,000,000 p=14 | PASS 1.1x |
| APO | 3.18 | n=1,000,000 p=14 | FAIL |
| AROON | 3.72 | n=10,000,000 p=200 | FAIL |
| AROONOSC | 3.70 | n=10,000,000 p=200 | FAIL |
| ASIN | 1.01 | n=1,000,000 p=None | PASS 1.1x |
| ATAN | 1.11 | n=10,000,000 p=None | PASS 1.5x |
| ATR | 0.36 | n=1,000,000 p=200 | PASS 1.1x |
| AVGPRICE | 0.97 | n=1,000,000 p=None | PASS 1.1x |
| BBANDS | 2.34 | n=1,000,000 p=14 | FAIL |
| BETA | 1.92 | n=10,000,000 p=200 | FAIL |
| BOP | 3.04 | n=1,000,000 p=None | FAIL |
| CCI | 2.74 | n=1,000,000 p=14 | FAIL |
| CDL2CROWS | 1.56 | n=1,000,000 p=None | FAIL |
| CDL3BLACKCROWS | 1.79 | n=1,000,000 p=None | FAIL |
| CDL3INSIDE | 1.70 | n=10,000,000 p=None | FAIL |
| CDL3LINESTRIKE | 1.79 | n=10,000,000 p=None | FAIL |
| CDL3OUTSIDE | 1.59 | n=10,000,000 p=None | FAIL |
| CDL3STARSINSOUTH | 1.92 | n=1,000,000 p=None | FAIL |
| CDL3WHITESOLDIERS | 2.03 | n=10,000,000 p=None | FAIL |
| CDLABANDONEDBABY | 1.65 | n=10,000,000 p=None | FAIL |
| CDLADVANCEBLOCK | 1.79 | n=10,000,000 p=None | FAIL |
| CDLBELTHOLD | 1.55 | n=1,000,000 p=None | FAIL |
| CDLBREAKAWAY | 1.47 | n=10,000,000 p=None | PASS 1.5x |
| CDLCLOSINGMARUBOZU | 1.57 | n=1,000,000 p=None | FAIL |
| CDLCONCEALBABYSWALL | 1.94 | n=10,000,000 p=None | FAIL |
| CDLCOUNTERATTACK | 1.60 | n=10,000,000 p=None | FAIL |
| CDLDARKCLOUDCOVER | 1.54 | n=1,000,000 p=None | FAIL |
| CDLDOJI | 5.94 | n=1,000,000 p=None | FAIL |
| CDLDOJISTAR | 1.76 | n=1,000,000 p=None | FAIL |
| CDLDRAGONFLYDOJI | 1.58 | n=1,000,000 p=None | FAIL |
| CDLENGULFING | 1.68 | n=1,000,000 p=None | FAIL |
| CDLEVENINGDOJISTAR | 2.40 | n=10,000,000 p=None | FAIL |
| CDLEVENINGSTAR | 2.22 | n=10,000,000 p=None | FAIL |
| CDLGAPSIDESIDEWHITE | 1.76 | n=10,000,000 p=None | FAIL |
| CDLGRAVESTONEDOJI | 1.61 | n=1,000,000 p=None | FAIL |
| CDLHAMMER | 1.60 | n=1,000,000 p=None | FAIL |
| CDLHANGINGMAN | 1.59 | n=1,000,000 p=None | FAIL |
| CDLHARAMI | 1.66 | n=1,000,000 p=None | FAIL |
| CDLHARAMICROSS | 1.61 | n=1,000,000 p=None | FAIL |
| CDLHIGHWAVE | 1.46 | n=1,000,000 p=None | PASS 1.5x |
| CDLHIKKAKE | 1.96 | n=10,000,000 p=None | FAIL |
| CDLHIKKAKEMOD | 3.02 | n=1,000,000 p=None | FAIL |
| CDLHOMINGPIGEON | 1.78 | n=10,000,000 p=None | FAIL |
| CDLIDENTICAL3CROWS | 2.02 | n=10,000,000 p=None | FAIL |
| CDLINNECK | 1.63 | n=10,000,000 p=None | FAIL |
| CDLINVERTEDHAMMER | 1.57 | n=10,000,000 p=None | FAIL |
| CDLKICKING | 1.70 | n=1,000,000 p=None | FAIL |
| CDLKICKINGBYLENGTH | 1.71 | n=10,000,000 p=None | FAIL |
| CDLLADDERBOTTOM | 1.86 | n=10,000,000 p=None | FAIL |
| CDLLONGLEGGEDDOJI | 1.81 | n=10,000,000 p=None | FAIL |
| CDLLONGLINE | 1.43 | n=10,000,000 p=None | PASS 1.5x |
| CDLMARUBOZU | 1.58 | n=1,000,000 p=None | FAIL |
| CDLMATCHINGLOW | 1.68 | n=10,000,000 p=None | FAIL |
| CDLMATHOLD | 1.87 | n=1,000,000 p=None | FAIL |
| CDLMORNINGDOJISTAR | 2.39 | n=10,000,000 p=None | FAIL |
| CDLMORNINGSTAR | 2.21 | n=1,000,000 p=None | FAIL |
| CDLONNECK | 1.60 | n=1,000,000 p=None | FAIL |
| CDLPIERCING | 1.52 | n=1,000,000 p=None | FAIL |
| CDLRICKSHAWMAN | 1.68 | n=1,000,000 p=None | FAIL |
| CDLRISEFALL3METHODS | 2.06 | n=1,000,000 p=None | FAIL |
| CDLSEPARATINGLINES | 1.52 | n=1,000,000 p=None | FAIL |
| CDLSHOOTINGSTAR | 1.59 | n=1,000,000 p=None | FAIL |
| CDLSHORTLINE | 1.34 | n=1,000,000 p=None | PASS 1.5x |
| CDLSPINNINGTOP | 1.67 | n=1,000,000 p=None | FAIL |
| CDLSTALLEDPATTERN | 1.99 | n=10,000,000 p=None | FAIL |
| CDLSTICKSANDWICH | 1.88 | n=10,000,000 p=None | FAIL |
| CDLTAKURI | 1.73 | n=1,000,000 p=None | FAIL |
| CDLTASUKIGAP | 2.44 | n=10,000,000 p=None | FAIL |
| CDLTHRUSTING | 1.58 | n=10,000,000 p=None | FAIL |
| CDLTRISTAR | 1.78 | n=10,000,000 p=None | FAIL |
| CDLUNIQUE3RIVER | 2.31 | n=10,000,000 p=None | FAIL |
| CDLUPSIDEGAP2CROWS | 1.61 | n=10,000,000 p=None | FAIL |
| CDLXSIDEGAP3METHODS | 1.43 | n=10,000,000 p=None | PASS 1.5x |
| CEIL | 0.82 | n=10,000,000 p=None | PASS 1.1x |
| CMO | 1.04 | n=1,000,000 p=200 | PASS 1.1x |
| CORREL | 4.54 | n=1,000,000 p=14 | FAIL |
| COS | 1.06 | n=10,000,000 p=None | PASS 1.1x |
| COSH | 1.05 | n=10,000,000 p=None | PASS 1.1x |
| DEMA | 0.77 | n=10,000,000 p=200 | PASS 1.1x |
| DIV | 0.97 | n=1,000,000 p=None | PASS 1.1x |
| DX | 0.69 | n=10,000,000 p=14 | PASS 1.1x |
| EMA | 1.21 | n=10,000,000 p=200 | PASS 1.5x |
| EXP | 1.05 | n=10,000,000 p=None | PASS 1.1x |
| FLOOR | 0.79 | n=10,000,000 p=None | PASS 1.1x |
| HT_DCPERIOD | 1.29 | n=10,000,000 p=None | PASS 1.5x |
| HT_DCPHASE | 1.25 | n=1,000,000 p=None | PASS 1.5x |
| HT_PHASOR | 1.59 | n=10,000,000 p=None | FAIL |
| HT_SINE | 1.14 | n=10,000,000 p=None | PASS 1.5x |
| HT_TRENDLINE | 1.43 | n=10,000,000 p=None | PASS 1.5x |
| HT_TRENDMODE | 1.10 | n=10,000,000 p=None | PASS 1.1x |
| KAMA | 3.19 | n=1,000,000 p=14 | FAIL |
| LINEARREG | 1.93 | n=1,000,000 p=14 | FAIL |
| LINEARREG_ANGLE | 1.10 | n=1,000,000 p=14 | PASS 1.5x |
| LINEARREG_INTERCEPT | 1.91 | n=1,000,000 p=14 | FAIL |
| LINEARREG_SLOPE | 1.81 | n=1,000,000 p=14 | FAIL |
| LN | 1.11 | n=10,000,000 p=None | PASS 1.5x |
| LOG10 | 1.06 | n=10,000,000 p=None | PASS 1.1x |
| MA | 2.87 | n=10,000,000 p=14 | FAIL |
| MACD | 1.78 | n=1,000,000 p=14 | FAIL |
| MACDEXT | 2.91 | n=1,000,000 p=14 | FAIL |
| MACDFIX | 1.71 | n=10,000,000 p=200 | FAIL |
| MAMA | 1.23 | n=10,000,000 p=None | PASS 1.5x |
| MAVP | 15.47 | n=10,000,000 p=14 | FAIL |
| MAX | 2.09 | n=1,000,000 p=200 | FAIL |
| MAXINDEX | 2.40 | n=10,000,000 p=14 | FAIL |
| MEDPRICE | 1.00 | n=1,000,000 p=None | PASS 1.1x |
| MFI | 0.72 | n=10,000,000 p=200 | PASS 1.1x |
| MIDPOINT | 0.27 | n=10,000,000 p=14 | PASS 1.1x |
| MIDPRICE | 2.36 | n=1,000,000 p=14 | FAIL |
| MIN | 1.54 | n=10,000,000 p=14 | FAIL |
| MININDEX | 2.36 | n=10,000,000 p=14 | FAIL |
| MINMAX | 2.14 | n=1,000,000 p=14 | FAIL |
| MINMAXINDEX | 2.17 | n=10,000,000 p=14 | FAIL |
| MINUS_DI | 1.10 | n=1,000,000 p=200 | PASS 1.1x |
| MINUS_DM | 1.04 | n=1,000,000 p=200 | PASS 1.1x |
| MOM | 1.59 | n=10,000,000 p=14 | FAIL |
| MULT | 1.22 | n=10,000,000 p=None | PASS 1.5x |
| NATR | 0.95 | n=1,000,000 p=14 | PASS 1.1x |
| OBV | 1.05 | n=10,000,000 p=None | PASS 1.1x |
| PLUS_DI | 1.09 | n=1,000,000 p=200 | PASS 1.1x |
| PLUS_DM | 1.04 | n=10,000,000 p=200 | PASS 1.1x |
| PPO | 2.81 | n=1,000,000 p=200 | FAIL |
| ROC | 2.11 | n=10,000,000 p=200 | FAIL |
| ROCP | 2.07 | n=10,000,000 p=14 | FAIL |
| ROCR | 2.03 | n=10,000,000 p=200 | FAIL |
| ROCR100 | 2.16 | n=10,000,000 p=200 | FAIL |
| RSI | 0.44 | n=1,000,000 p=200 | PASS 1.1x |
| SAR | 1.50 | n=10,000,000 p=None | FAIL |
| SAREXT | 1.47 | n=10,000,000 p=None | PASS 1.5x |
| SIN | 1.06 | n=10,000,000 p=None | PASS 1.1x |
| SINH | 1.06 | n=10,000,000 p=None | PASS 1.1x |
| SMA | 1.39 | n=1,000,000 p=200 | PASS 1.5x |
| SQRT | 0.65 | n=10,000,000 p=None | PASS 1.1x |
| STDDEV | 3.62 | n=1,000,000 p=14 | FAIL |
| STOCH | 3.00 | n=10,000,000 p=200 | FAIL |
| STOCHF | 3.05 | n=10,000,000 p=200 | FAIL |
| STOCHRSI | 3.01 | n=1,000,000 p=200 | FAIL |
| SUB | 1.21 | n=10,000,000 p=None | PASS 1.5x |
| SUM | 1.88 | n=1,000,000 p=200 | FAIL |
| T3 | 2.58 | n=1,000,000 p=200 | FAIL |
| TAN | 1.11 | n=10,000,000 p=None | PASS 1.5x |
| TANH | 1.07 | n=10,000,000 p=None | PASS 1.1x |
| TEMA | 0.98 | n=1,000,000 p=200 | PASS 1.1x |
| TRANGE | 1.98 | n=10,000,000 p=None | FAIL |
| TRIMA | 2.80 | n=10,000,000 p=200 | FAIL |
| TRIX | 1.56 | n=10,000,000 p=200 | FAIL |
| TSF | 1.91 | n=1,000,000 p=14 | FAIL |
| TYPPRICE | 0.86 | n=1,000,000 p=None | PASS 1.1x |
| ULTOSC | 3.65 | n=10,000,000 p=200 | FAIL |
| VAR | 6.18 | n=1,000,000 p=14 | FAIL |
| WCLPRICE | 1.00 | n=1,000,000 p=None | PASS 1.1x |
| WILLR | 2.26 | n=10,000,000 p=200 | FAIL |
| WMA | 6.40 | n=1,000,000 p=14 | FAIL |

35/158 functions within 1.1x (22%, need 90%); all within 1.5x: False. Speed bar: FAIL

Not registered or not comparable here: 1 (see JSON).

## Cases

| Case | Bars | Python threads | Rayon pool | TA-Lib s | raptorbt s | Ratio |
|---|---|---|---|---|---|---|
| rolling_max_200 | 1,000,000 | 1 | 1 | 0.0090 | 0.0209 | 2.33 |
| shared_nodes | 1,000,000 | 1 | 1 | 0.1315 | 0.0345 | 0.26 |
| instruments_500 | 1,000,000 | 1 | 1 | 0.0024 | 0.0038 | 1.60 |
| instruments_500 | 1,000,000 | 10 | 1 | 0.0051 | 0.0082 | 1.60 |
| rolling_max_200 | 10,000,000 | 1 | 1 | 0.8612 | 0.1229 | 0.14 |
| shared_nodes | 10,000,000 | 1 | 1 | 1.2522 | 0.3807 | 0.30 |
| instruments_500 | 10,000,000 | 1 | 1 | 0.0203 | 0.0275 | 1.35 |
| instruments_500 | 10,000,000 | 10 | 1 | 0.0232 | 0.0105 | 0.45 |
| rolling_max_200 | 1,000,000 | 1 | 10 | 0.0090 | 0.0208 | 2.32 |
| shared_nodes | 1,000,000 | 1 | 10 | 0.1315 | 0.0346 | 0.26 |
| instruments_500 | 1,000,000 | 1 | 10 | 0.0024 | 0.0038 | 1.59 |
| instruments_500 | 1,000,000 | 10 | 10 | 0.0051 | 0.0086 | 1.70 |
| rolling_max_200 | 10,000,000 | 1 | 10 | 0.8612 | 0.1240 | 0.14 |
| shared_nodes | 10,000,000 | 1 | 10 | 1.2522 | 0.3690 | 0.29 |
| instruments_500 | 10,000,000 | 1 | 10 | 0.0203 | 0.0270 | 1.33 |
| instruments_500 | 10,000,000 | 10 | 10 | 0.0232 | 0.0105 | 0.46 |

## Sweep

100 bindings of `sma(close, $n) > sma(close, $m)`: the joint plan against each binding evaluated on its own. No TA-Lib counterpart.

| Bars | Bindings | Rayon pool | Joint s | Independent s | Joint / independent |
|---|---|---|---|---|---|
| 1,000,000 | 100 | 1 | 0.1586 | 0.6436 | 0.25 |
| 1,000,000 | 100 | 10 | 0.0654 | 0.6474 | 0.10 |

The shared-nodes case is one raptorbt program summing 20 terms that share `sma` and `ema`, against TA-Lib recomputing both per term. The instruments case is `sma(close, 20)` over 500 instruments; the rows with more than one Python thread call it from that many threads on both sides (each call is one series on one thread, and does not use the rayon pool). The rows with one Python thread are the ones to hold against the single-series bar.
