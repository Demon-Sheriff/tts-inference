
### `metrics for benchmarking`

### `Plot 1: chunk_timeline_analysis.png (4 panels)`

Top-Left: Chunk Arrival Timeline
Shows when each chunk (Y-axis: Chunk ID 1-12) arrives over time (X-axis: 1.5-5.5 seconds)
Each colored line = one iteration (15 total)
The lines are tightly clustered, showing consistent streaming behavior
Iteration 1 (purple line at far right ~5s) was the cold start, much slower than others

Top-Right: Chunk Gap Analysis (Stall Detection)
Scatter plot of inter-chunk gaps over time
Red dashed line at 200ms threshold, dark red at 500ms
Most gaps cluster around 200-300ms range
A few outliers above 300ms (potential micro-stalls)
No gaps exceeded 500ms.


Bottom-Left: Max Chunk Gap per Iteration (Drift Detection)
Orange bars = worst gap per iteration
Blue line = mean gap per iteration
Gaps range from ~240-360ms
No clear upward drift over 15 iterations (no thermal throttling detected)


Bottom-Right: RTF and TTFA Stability Over Time
Green line = RTF (Real-Time Factor) - stable around 1.3-1.4x
Purple line = TTFA - spike at iteration 1 (cold start ~4800ms), then stable ~1700ms
Shows the system is stable after warmup

<img width="2100" height="1500" alt="image" src="https://github.com/user-attachments/assets/271ae152-7576-49df-9b88-9a621700e09c" />


### `chunk timeline analysis`
<img width="2100" height="1500" alt="image" src="https://github.com/user-attachments/assets/0705152c-541c-4cbc-a985-23d230ea1238" />
