# Novel object location analysis

Turns DeepLabCut pose tracking into a discrimination index and a set of exploration and
locomotion measures for the novel object location (NOL) task — one per-animal table
per cohort.

Edit arena size, pixels-per-centimeter, and frame rate based on your setup prior to running code. 

## What it measures

Per animal:

- **Distortion correction**: a projective homography fit from the four arena corners
  (A, B, C, D) maps pixel coordinates to centimetres in the arena frame.
- **Centroid**: the mean of Neck, Back1, Back2, and Back3, ignoring any point DeepLabCut
  flagged low-likelihood.
- **Investigation**: counted when the nose is within a small radius of an object edge
  *and* the head points at it (the angle between the neck→nose and nose→object vectors).
- **Discrimination index**: novel-object time minus familiar-object time, over total
  object time. Object B is the novel location, object A the familiar one.
- **Thigmotaxis**: wall time over wall-plus-centre time, with object time pulled out of
  the denominator so wall preference isn't confounded by investigation.
- **Locomotion**: summed frame-to-frame centroid displacement, short tracking gaps
  interpolated, frames below a speed threshold flagged immobile.

There are also a handful of alternative learning readouts (time-binned DI, cumulative DI,
a bout-count DI, an occupancy DI, and which object got approached first), because the
single end-of-session number can hide the shape of the learning.

## Input Data

DeepLabCut exports, `.csv` or `.h5`, with:

- body points Nose, Neck, Back1, Back2, Back3, Tailbase
- object points ObjA_S (familiar) and ObjB_Novel (novel)
- corners A, B, C, D

If both a raw and a filtered export exist for a recording, it uses the filtered one.

## Running it

```
pip install -r requirements.txt
```

`generate_synthetic_data.py` writes DeepLabCut-shaped tables (a few animals wandering the
arena with scripted visits to the two locations), so the pipeline runs with no real
recordings on hand:

```
python3 generate_synthetic_data.py --output-dir synthetic_data --n-animals 8 --duration-sec 300 --frame-rate 30 --seed 7
python3 nol_analysis.py --input-dir synthetic_data --output-dir outputs --frame-rate 30 --time-window-sec 0,300
```

On a real recording I pass my rig frame rate and the scoring window I use for the task —
usually starting a couple of minutes in and running to the end. Every threshold is a flag;
`--help` lists them all with defaults.

Redraw the summary figure from a saved table without re-running the analysis:

```
python3 make_nol_figures.py --input outputs --output outputs/summary.png
```

## What comes out

- a per-animal table: DI, the exploration measures, the alternative metrics
- a per-bout table, one row per investigation bout
- a summary JSON with the run parameters and the cohort numbers
- a summary figure: DI per animal, against wall time and distance

## Also here

`plot_b1a_splenocytes.py` is unrelated to the NOL pipeline — it plots a flow cytometry
readout against the SCI cohort split (uninjured, and vehicle vs NM72 at 6 h and 24 h).
Two columns of the FlowJo B-1 table are carried:

- `igm_pos` — the `Live / CD45+ / B cells / IgM+ IgDlo / CD43+ B220lo` gate, labelled
  B-1a, as a percentage of B cells
- `igm_neg` — the `IgM-` column of that same table

The per-animal values and the cohort assignments are transcribed into the script, so it
needs no input files:

```
python3 plot_b1a_splenocytes.py --measure both --output outputs/b1_both.png --csv outputs/b1_splenocytes.csv
```

`--measure` takes `igm_pos`, `igm_neg`, or `both` (a two-panel figure, each panel on its
own y-scale). `--stats` annotates the vehicle-vs-NM72 Welch t-tests, `--dark` renders on
a dark surface.

Animal 489 is excluded, and 501, 503 and 504 are on the cohort sheet but absent from the
FlowJo export. On startup the script re-derives the mean and SD of each column over every
acquired sample and compares them against that table's own footer, so a mistyped value
shows up as a warning rather than as a quietly wrong figure.

## FlowJo T cell / myeloid panel

`plot_flow_panel.py` reads a FlowJo workspace directly. The workspace caches an event
count on every gate node, so the whole hierarchy comes out without the `.fcs` files:

```
python3 plot_flow_panel.py --wsp "260916 SCI T Cell (Ly6G fixed).wsp"
python3 plot_flow_panel.py          # rebuild from the cached counts in data/
```

Counts and per-sample acquisition metadata are cached under `data/`, so the figures
rebuild without the workspace. Two things worth knowing about the format:

- FlowJo hangs boolean-NOT gates off `NotNode` rather than `Population`. The whole
  myeloid branch here sits under a `CD3+-` NotNode, so a parser that only walks
  `Population` silently drops it.
- A sample can appear twice; the entry with more live leukocyte events is kept.

Exclusions are judged on the live leukocyte gate via `--min-live` (default 1000).
Frequencies computed from fewer than 20 numerator events are blanked rather than
plotted, and any panel that loses more than a third of its animals that way is drawn
with a red title.

It also writes two QC figures, which are the point as much as the population plots are:

- `flow_qc_live.png` — live leukocyte yield per animal against the exclusion threshold.
- `flow_qc_runorder.png` — CD4+CD8 as a share of CD3+, CD3+ of live, and CD4+ of CD3+
  against the order the tubes were acquired in. CD4 and CD8 should account for most
  CD3+ events in spleen; where they do not, the CD3 gate is holding something else.
  The script splits that measure at its largest gap, widens the split to the contiguous
  acquisition window it covers, and warns on the figure if any cohort group falls
  entirely inside that window — such a group cannot be compared with one outside it.

## Files

## Files

- `nol_analysis.py`: the analysis module and CLI entry point
- `make_nol_figures.py`: draws the summary figure from a per-animal table
- `generate_synthetic_data.py`: writes the synthetic DeepLabCut tables
- `plot_b1a_splenocytes.py`: the B-1 cohort figures described above
- `plot_flow_panel.py`: the T cell / myeloid figures and their QC
- `sci_cohorts.py`: the cohort split, palette and group-scatter panel both flow scripts share

MIT licensed.
