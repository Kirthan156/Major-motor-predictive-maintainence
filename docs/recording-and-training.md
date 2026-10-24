# Recording, training and evaluation

Details behind steps 4 and 5 of the main [README](../README.md).

## Recording plan (the part that decides your marks)

- **Session ids must be unique.** Evaluation holds out whole sessions, so one session = one physical setup.
- **Take the fault off and put it back between repeats** of the same label/severity. Two back-to-back
  recordings of the same setup are one session for evaluation purposes.
- Minimum to get meaningful numbers: per fault, 3 severities x 2 repeats x 3-5 min; healthy: at least
  6 sessions spread over different days (start of day, after the motor warms up, after remounting).
- Write down what each severity physically is (grams of imbalance, shim thickness, % of grille blocked)
  in `--notes`. That table goes straight into the report.
- Let the motor run ~2 min before starting a session so you are not recording the start-up transient.

## What train.py reports

```
cd pm
python train.py                                                    # uses data/raw
python synthetic.py && python train.py --data ../data/synthetic    # pipeline smoke test only
```

Prints and saves (in `reports/`): leave-one-session-out macro-F1 and confusion matrix for random forest and
XGBoost, the inflated random-split score for comparison, recall per fault severity, a sensor ablation
(vibration vs +electrical vs all), and anomaly ROC-AUC / false-alarm rate / detection per severity.

## Firmware v2: what changed from v1 and why

| Change | Why |
|---|---|
| Accelerometer: 1024 raw samples at ~1 kHz per message (was 1 sample per 2 s) | One sample every 2 s cannot show vibration. A fan at ~1300 RPM vibrates at ~22 Hz (1x) and ~65 Hz (blade pass); you need hundreds of samples per second to see either. |
| Accel range +/-4 g, MPU low-pass 184 Hz | Headroom for a loose blade; anti-aliasing for 1 kHz sampling. |
| RPM from time between pulses | Counting pulses per second at 1 pulse/rev only gives 60 RPM steps. |
| Sound in dBFS with DC offset removed | The INMP441 has a DC offset; RMS of the raw value was mostly that offset (the "stuck at 32000" reading). |
| Power factor and mains frequency added | Cheap extra electrical features. |
| Message is ~15-20 KB and streamed with `beginPublish` | Raw samples are needed for FFT features. No PubSubClient buffer change needed. |
| Wi-Fi password moved to `secrets.h` | Keeps it out of anything you share. |

The OLED freezes for ~1 s every 2 s while the burst is captured. That is expected.

## PZEM current: what to check before buying anything

A 60 W motor at 230 V draws about 0.26-0.3 A. With 18 turns through the ring the PZEM should see
~5 A, which is well inside its range, so reading 0.03 A means the ring is not seeing the current,
not that the current is too small. Check, in order:

1. Only the **live** wire passes through the ring (if both live and neutral pass through, they cancel).
2. All 18 turns go through in the **same direction** (wind it like a coil, never back through).
3. The ring's two leads are firmly in the PZEM's CT terminals.
4. Test with a 100 W incandescent bulb (~0.43 A): expect ~7.8 A raw with 18 turns. Set
   `CT_TURNS = raw / true`.

If it still reads near zero, buy the PZEM-004T **10A** (shunt, in series with live). It is cheap and removes
the ring entirely. Mains wiring: power off and unplugged while rewiring.
