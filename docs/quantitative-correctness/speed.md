# Speed against TA-Lib

TA-Lib 0.6.4 on macOS-26.5.2-arm64-arm-64bit, 10 cores. Median of several runs after a warm-up. Ratio is raptorbt / TA-Lib; below 1.0 is faster. Single-threaded raptorbt run: 1 thread(s).

## Single series, single thread

| Function | Worst ratio | Where | Verdict |
|---|---|---|---|
| ACOS | 1.31 | n=1,000,000 p=None | PASS 1.5x |
| AD | 1.53 | n=10,000,000 p=None | FAIL |
| ADD | 5.99 | n=1,000,000 p=None | FAIL |
| ADOSC | 1.09 | n=10,000,000 p=14 | PASS 1.1x |
| ADX | 1.06 | n=10,000,000 p=200 | PASS 1.1x |
| ADXR | 1.02 | n=10,000,000 p=14 | PASS 1.1x |
| APO | 2.97 | n=1,000,000 p=200 | FAIL |
| AROON | 7.23 | n=10,000,000 p=14 | FAIL |
| AROONOSC | 7.19 | n=10,000,000 p=14 | FAIL |
| ASIN | 1.27 | n=1,000,000 p=None | PASS 1.5x |
| ATAN | 1.19 | n=10,000,000 p=None | PASS 1.5x |
| ATR | 0.35 | n=10,000,000 p=14 | PASS 1.1x |
| AVGPRICE | 2.29 | n=1,000,000 p=None | FAIL |
| BBANDS | 19.46 | n=10,000,000 p=200 | FAIL |
| BETA | 276.16 | n=1,000,000 p=200 | FAIL |
| BOP | 2.11 | n=1,000,000 p=None | FAIL |
| CCI | 2.52 | n=10,000,000 p=14 | FAIL |
| CDL2CROWS | 3.10 | n=10,000,000 p=None | FAIL |
| CDL3BLACKCROWS | 3.89 | n=1,000,000 p=None | FAIL |
| CDL3INSIDE | 3.35 | n=10,000,000 p=None | FAIL |
| CDL3LINESTRIKE | 3.01 | n=10,000,000 p=None | FAIL |
| CDL3OUTSIDE | 2.19 | n=1,000,000 p=None | FAIL |
| CDL3STARSINSOUTH | 3.70 | n=1,000,000 p=None | FAIL |
| CDL3WHITESOLDIERS | 3.59 | n=1,000,000 p=None | FAIL |
| CDLABANDONEDBABY | 2.99 | n=10,000,000 p=None | FAIL |
| CDLADVANCEBLOCK | 3.29 | n=10,000,000 p=None | FAIL |
| CDLBELTHOLD | 2.60 | n=1,000,000 p=None | FAIL |
| CDLBREAKAWAY | 2.74 | n=10,000,000 p=None | FAIL |
| CDLCLOSINGMARUBOZU | 2.62 | n=1,000,000 p=None | FAIL |
| CDLCONCEALBABYSWALL | 4.66 | n=1,000,000 p=None | FAIL |
| CDLCOUNTERATTACK | 2.80 | n=10,000,000 p=None | FAIL |
| CDLDARKCLOUDCOVER | 2.93 | n=10,000,000 p=None | FAIL |
| CDLDOJI | 18.58 | n=10,000,000 p=None | FAIL |
| CDLDOJISTAR | 3.05 | n=1,000,000 p=None | FAIL |
| CDLDRAGONFLYDOJI | 3.12 | n=1,000,000 p=None | FAIL |
| CDLENGULFING | 2.23 | n=10,000,000 p=None | FAIL |
| CDLEVENINGDOJISTAR | 4.91 | n=10,000,000 p=None | FAIL |
| CDLEVENINGSTAR | 4.52 | n=10,000,000 p=None | FAIL |
| CDLGAPSIDESIDEWHITE | 2.85 | n=10,000,000 p=None | FAIL |
| CDLGRAVESTONEDOJI | 3.16 | n=1,000,000 p=None | FAIL |
| CDLHAMMER | 2.84 | n=1,000,000 p=None | FAIL |
| CDLHANGINGMAN | 2.87 | n=1,000,000 p=None | FAIL |
| CDLHARAMI | 3.25 | n=1,000,000 p=None | FAIL |
| CDLHARAMICROSS | 3.16 | n=1,000,000 p=None | FAIL |
| CDLHIGHWAVE | 2.47 | n=1,000,000 p=None | FAIL |
| CDLHIKKAKE | 3.19 | n=1,000,000 p=None | FAIL |
| CDLHIKKAKEMOD | 7.78 | n=1,000,000 p=None | FAIL |
| CDLHOMINGPIGEON | 3.61 | n=10,000,000 p=None | FAIL |
| CDLIDENTICAL3CROWS | 3.85 | n=1,000,000 p=None | FAIL |
| CDLINNECK | 3.00 | n=10,000,000 p=None | FAIL |
| CDLINVERTEDHAMMER | 2.74 | n=1,000,000 p=None | FAIL |
| CDLKICKING | 2.83 | n=10,000,000 p=None | FAIL |
| CDLKICKINGBYLENGTH | 2.83 | n=1,000,000 p=None | FAIL |
| CDLLADDERBOTTOM | 5.53 | n=10,000,000 p=None | FAIL |
| CDLLONGLEGGEDDOJI | 3.52 | n=1,000,000 p=None | FAIL |
| CDLLONGLINE | 2.20 | n=1,000,000 p=None | FAIL |
| CDLMARUBOZU | 2.71 | n=1,000,000 p=None | FAIL |
| CDLMATCHINGLOW | 2.68 | n=10,000,000 p=None | FAIL |
| CDLMATHOLD | 3.55 | n=1,000,000 p=None | FAIL |
| CDLMORNINGDOJISTAR | 4.90 | n=10,000,000 p=None | FAIL |
| CDLMORNINGSTAR | 4.51 | n=10,000,000 p=None | FAIL |
| CDLONNECK | 2.97 | n=10,000,000 p=None | FAIL |
| CDLPIERCING | 2.75 | n=1,000,000 p=None | FAIL |
| CDLRICKSHAWMAN | 3.15 | n=1,000,000 p=None | FAIL |
| CDLRISEFALL3METHODS | 3.54 | n=10,000,000 p=None | FAIL |
| CDLSEPARATINGLINES | 2.79 | n=10,000,000 p=None | FAIL |
| CDLSHOOTINGSTAR | 2.75 | n=1,000,000 p=None | FAIL |
| CDLSHORTLINE | 2.01 | n=1,000,000 p=None | FAIL |
| CDLSPINNINGTOP | 2.81 | n=1,000,000 p=None | FAIL |
| CDLSTALLEDPATTERN | 3.92 | n=10,000,000 p=None | FAIL |
| CDLSTICKSANDWICH | 3.74 | n=10,000,000 p=None | FAIL |
| CDLTAKURI | 3.24 | n=1,000,000 p=None | FAIL |
| CDLTASUKIGAP | 5.10 | n=10,000,000 p=None | FAIL |
| CDLTHRUSTING | 2.93 | n=10,000,000 p=None | FAIL |
| CDLTRISTAR | 3.80 | n=1,000,000 p=None | FAIL |
| CDLUNIQUE3RIVER | 4.85 | n=10,000,000 p=None | FAIL |
| CDLUPSIDEGAP2CROWS | 2.94 | n=10,000,000 p=None | FAIL |
| CDLXSIDEGAP3METHODS | 1.76 | n=10,000,000 p=None | FAIL |
| CEIL | 3.45 | n=10,000,000 p=None | FAIL |
| CMO | 1.03 | n=1,000,000 p=200 | PASS 1.1x |
| CORREL | 308.92 | n=1,000,000 p=200 | FAIL |
| COS | 1.08 | n=1,000,000 p=None | PASS 1.1x |
| COSH | 1.34 | n=10,000,000 p=None | PASS 1.5x |
| DEMA | 0.83 | n=10,000,000 p=14 | PASS 1.1x |
| DIV | 5.93 | n=1,000,000 p=None | FAIL |
| DX | 1.17 | n=10,000,000 p=200 | PASS 1.5x |
| EMA | 1.23 | n=10,000,000 p=200 | PASS 1.5x |
| EXP | 1.38 | n=10,000,000 p=None | PASS 1.5x |
| FLOOR | 3.31 | n=10,000,000 p=None | FAIL |
| HT_DCPERIOD | 1.28 | n=10,000,000 p=None | PASS 1.5x |
| HT_DCPHASE | 1.24 | n=10,000,000 p=None | PASS 1.5x |
| HT_PHASOR | 1.59 | n=10,000,000 p=None | FAIL |
| HT_SINE | 1.14 | n=10,000,000 p=None | PASS 1.5x |
| HT_TRENDLINE | 6.35 | n=10,000,000 p=None | FAIL |
| HT_TRENDMODE | 1.09 | n=10,000,000 p=None | PASS 1.1x |
| KAMA | 3.24 | n=1,000,000 p=14 | FAIL |
| LINEARREG | 2.76 | n=1,000,000 p=14 | FAIL |
| LINEARREG_ANGLE | 2.15 | n=1,000,000 p=200 | FAIL |
| LINEARREG_INTERCEPT | 2.96 | n=1,000,000 p=14 | FAIL |
| LINEARREG_SLOPE | 3.11 | n=1,000,000 p=14 | FAIL |
| LN | 1.33 | n=10,000,000 p=None | PASS 1.5x |
| LOG10 | 1.30 | n=10,000,000 p=None | PASS 1.5x |
| MA | 4.75 | n=1,000,000 p=14 | FAIL |
| MACD | 1.82 | n=10,000,000 p=14 | FAIL |
| MACDEXT | 2.94 | n=10,000,000 p=14 | FAIL |
| MACDFIX | 1.76 | n=10,000,000 p=14 | FAIL |
| MAMA | 1.24 | n=10,000,000 p=None | PASS 1.5x |
| MAVP | 15.20 | n=10,000,000 p=14 | FAIL |
| MAX | 5.51 | n=1,000,000 p=14 | FAIL |
| MAXINDEX | 7.34 | n=10,000,000 p=200 | FAIL |
| MEDPRICE | 5.92 | n=1,000,000 p=None | FAIL |
| MFI | 48.21 | n=10,000,000 p=200 | FAIL |
| MIDPOINT | 0.49 | n=10,000,000 p=14 | PASS 1.1x |
| MIDPRICE | 5.76 | n=1,000,000 p=14 | FAIL |
| MIN | 4.51 | n=10,000,000 p=14 | FAIL |
| MININDEX | 5.72 | n=1,000,000 p=200 | FAIL |
| MINMAX | 4.78 | n=1,000,000 p=14 | FAIL |
| MINMAXINDEX | 7.31 | n=10,000,000 p=200 | FAIL |
| MINUS_DI | 1.80 | n=10,000,000 p=200 | FAIL |
| MINUS_DM | 1.02 | n=1,000,000 p=200 | PASS 1.1x |
| MOM | 4.35 | n=10,000,000 p=14 | FAIL |
| MULT | 4.51 | n=10,000,000 p=None | FAIL |
| NATR | 0.96 | n=10,000,000 p=200 | PASS 1.1x |
| OBV | 1.08 | n=10,000,000 p=None | PASS 1.1x |
| PLUS_DI | 1.86 | n=1,000,000 p=14 | FAIL |
| PLUS_DM | 1.01 | n=1,000,000 p=200 | PASS 1.1x |
| PPO | 2.68 | n=1,000,000 p=200 | FAIL |
| ROC | 3.41 | n=1,000,000 p=14 | FAIL |
| ROCP | 2.29 | n=10,000,000 p=14 | FAIL |
| ROCR | 2.36 | n=1,000,000 p=200 | FAIL |
| ROCR100 | 2.65 | n=10,000,000 p=14 | FAIL |
| RSI | 0.38 | n=10,000,000 p=14 | PASS 1.1x |
| SAR | 1.46 | n=10,000,000 p=None | PASS 1.5x |
| SAREXT | 1.41 | n=10,000,000 p=None | PASS 1.5x |
| SIN | 1.10 | n=10,000,000 p=None | PASS 1.5x |
| SINH | 1.52 | n=1,000,000 p=None | FAIL |
| SMA | 1.54 | n=10,000,000 p=200 | FAIL |
| SQRT | 1.66 | n=10,000,000 p=None | FAIL |
| STDDEV | 53.34 | n=1,000,000 p=200 | FAIL |
| STOCH | 4.70 | n=10,000,000 p=14 | FAIL |
| STOCHF | 5.17 | n=10,000,000 p=14 | FAIL |
| STOCHRSI | 2.47 | n=1,000,000 p=14 | FAIL |
| SUB | 5.66 | n=1,000,000 p=None | FAIL |
| SUM | 11.70 | n=10,000,000 p=200 | FAIL |
| T3 | 2.57 | n=1,000,000 p=200 | FAIL |
| TAN | 1.17 | n=10,000,000 p=None | PASS 1.5x |
| TANH | 1.23 | n=10,000,000 p=None | PASS 1.5x |
| TEMA | 0.95 | n=1,000,000 p=200 | PASS 1.1x |
| TRANGE | 3.23 | n=10,000,000 p=None | FAIL |
| TRIMA | 2.73 | n=1,000,000 p=14 | FAIL |
| TRIX | 1.56 | n=10,000,000 p=14 | FAIL |
| TSF | 2.81 | n=1,000,000 p=14 | FAIL |
| TYPPRICE | 2.28 | n=1,000,000 p=None | FAIL |
| ULTOSC | 3.60 | n=10,000,000 p=200 | FAIL |
| VAR | 91.23 | n=1,000,000 p=200 | FAIL |
| WCLPRICE | 2.79 | n=10,000,000 p=None | FAIL |
| WILLR | 5.61 | n=10,000,000 p=14 | FAIL |
| WMA | 96.53 | n=1,000,000 p=200 | FAIL |

15/158 functions within 1.1x (9%, need 90%); all within 1.5x: False. Speed bar: FAIL

Not registered or not comparable here: 1 (see JSON).

## Cases

| Case | Bars | Mode | TA-Lib s | raptorbt s | Ratio |
|---|---|---|---|---|---|
| rolling_max_200 | 1,000,000 | single (1 rayon threads) | 0.0089 | 0.0225 | 2.52 |
| shared_nodes | 1,000,000 | single (1 rayon threads) | 0.1256 | 0.0380 | 0.30 |
| instruments_500 | 1,000,000 | single (1 rayon threads) | 0.0020 | 0.0038 | 1.89 |
| instruments_500 | 1,000,000 | parallel (1 rayon threads) | 0.0049 | 0.0096 | 1.94 |
| rolling_max_200 | 10,000,000 | single (1 rayon threads) | 0.8624 | 0.1267 | 0.15 |
| shared_nodes | 10,000,000 | single (1 rayon threads) | 1.2519 | 0.3858 | 0.31 |
| instruments_500 | 10,000,000 | single (1 rayon threads) | 0.0198 | 0.0314 | 1.59 |
| instruments_500 | 10,000,000 | parallel (1 rayon threads) | 0.0228 | 0.0109 | 0.48 |
| rolling_max_200 | 1,000,000 | parallel (10 rayon threads) | 0.0089 | 0.0208 | 2.33 |
| shared_nodes | 1,000,000 | parallel (10 rayon threads) | 0.1256 | 0.0426 | 0.34 |
| instruments_500 | 1,000,000 | single (10 rayon threads) | 0.0020 | 0.0039 | 1.91 |
| instruments_500 | 1,000,000 | parallel (10 rayon threads) | 0.0049 | 0.0081 | 1.64 |
| rolling_max_200 | 10,000,000 | parallel (10 rayon threads) | 0.8624 | 0.1253 | 0.15 |
| shared_nodes | 10,000,000 | parallel (10 rayon threads) | 1.2519 | 0.3959 | 0.32 |
| instruments_500 | 10,000,000 | single (10 rayon threads) | 0.0198 | 0.0314 | 1.59 |
| instruments_500 | 10,000,000 | parallel (10 rayon threads) | 0.0228 | 0.0125 | 0.55 |

## Sweep

100 bindings of `sma(close, $n) > sma(close, $m)`: the joint plan against each binding evaluated on its own. No TA-Lib counterpart.

| Bars | Bindings | rayon threads | Joint s | Independent s | Joint / independent |
|---|---|---|---|---|---|
| 1,000,000 | 100 | 1 | 0.1734 | 0.7204 | 0.24 |
| 1,000,000 | 100 | 10 | 0.0747 | 0.7218 | 0.10 |

The shared-nodes case is one raptorbt program summing 20 terms that share `sma` and `ema`, against TA-Lib recomputing both per term. The instruments case is `sma(close, 20)` over 500 instruments; the parallel rows use a thread pool on both sides.
