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

## Files

- `nol_analysis.py`: the analysis module and CLI entry point
- `make_nol_figures.py`: draws the summary figure from a per-animal table
- `generate_synthetic_data.py`: writes the synthetic DeepLabCut tables

MIT licensed.
