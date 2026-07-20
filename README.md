# NOL analysis

This is my analysis code for the novel object location (NOL) task. It reads pose tracking from DeepLabCut and computes a discrimination index along with a set of exploration and locomotion measures for each animal. I wrote it to turn raw tracked coordinates into one clean table per cohort that I can check and plot.

The arena size, the pixel to centimetre scale, and the frame rate in the code are set to my own recording setup, so replace them with yours before you read anything into the output.

## What it computes

For each animal I do the following.

- I correct camera distortion with a projective homography fit from the four arena corners A, B, C, and D, so pixel coordinates become centimetres in the arena frame.
- I take the body centroid as the average of Neck, Back1, Back2, and Back3, ignoring any body point that DeepLabCut marked with low likelihood.
- I count an animal as investigating an object when its nose is within a small radius of the object edge and its head is pointed at the object. The head angle is the angle between the neck-to-nose vector and the nose-to-object vector.
- I compute the discrimination index as the novel-object time minus the familiar-object time, divided by the total object time. Object B is the novel location and object A is the familiar one.
- I measure thigmotaxis as wall time divided by wall time plus centre time, with object time removed from that denominator so wall preference is not confounded by object investigation.
- I measure locomotion as the summed frame-to-frame displacement of the centroid, with short tracking gaps filled by interpolation, and I flag immobile frames that fall below a speed threshold.
- I also compute a few alternative learning readouts, such as time-binned DI, cumulative DI, a bout-count DI, an occupancy DI, and which object was approached first.

I apply exclusion rules before I trust a discrimination index. An animal needs a minimum total object-interaction time and a minimum number of separate bouts at each object. I also flag low explorers and any animal whose arena corners were not detected.

At the cohort level I report the mean discrimination index and a one-sample test against chance, which is zero. I do not hard-code any group comparison. The tool treats every file it finds under the input folder as one cohort.

## Inputs

The analysis reads DeepLabCut exports in .csv or .h5 form. It expects these tracked points.

- body points Nose, Neck, Back1, Back2, Back3, and Tailbase
- object points ObjA_S for the familiar location and ObjB_Novel for the novel location
- corner points A, B, C, and D

When both a raw and a filtered export exist for the same recording, the filtered one is used.

## How to run

First install the requirements.

```
pip install -r requirements.txt
```

The repository ships with a synthetic data generator so you can run the whole pipeline without any real recordings. It writes fake DeepLabCut tables for a handful of animals, each wandering the arena and making scripted visits to the two object locations.

```
python3 generate_synthetic_data.py --output-dir synthetic_data --n-animals 8 --duration-sec 300 --frame-rate 30 --seed 7
```

Then run the analysis over that folder. I pass a frame rate and a full time window that match the synthetic session.

```
python3 nol_analysis.py --input-dir synthetic_data --output-dir outputs --frame-rate 30 --time-window-sec 0,300
```

For a real recording I run it with my rig frame rate and the scoring window I use for the task, for example a window that starts a couple of minutes into the trial and runs to the end. Every threshold is a command line flag, so run the script with --help to see the full list and the defaults.

You can also redraw the summary figure on its own from a saved table.

```
python3 make_nol_figures.py --input outputs --output outputs/summary.png
```

## Outputs

Each run writes a few files into the output folder.

- a per-animal table with the discrimination index, the exploration measures, and the alternative metrics
- a per-bout table with one row per investigation bout
- a summary json with the run parameters and the cohort-level numbers
- a summary figure with the discrimination index per animal and its relationship to wall time and distance

## Files

- `nol_analysis.py` is the core module and the command line entry point for the analysis.
- `make_nol_figures.py` draws the summary figure from a per-animal table.
- `generate_synthetic_data.py` writes the synthetic DeepLabCut tables.

## Note on the synthetic data

The synthetic data exists only so the code runs end to end on a fresh machine. It is drawn from random number generators. It is not real behavior and it is not meant to reflect any real result.

## License

MIT. See LICENSE.
