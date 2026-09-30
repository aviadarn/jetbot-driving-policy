# The 2023–24 build, and what its code actually did

The original car was built with my kid: a JetBot (Jetson Nano, IMX219 camera, Adafruit
MotorHAT) wired into a kids' ride-on car, driving an indoor corridor of orange cones.
Code, dataset notes and photos are at
[github.com/daisyKim12/autonomous_driving](https://github.com/daisyKim12/autonomous_driving).
The steering images and trained weights did not survive; this repository rebuilds the
system in simulation and on the same Nano, and re-runs its data story as an experiment.

This page records what the 2024 notebooks do, read line by line, including the bugs.
Everything marked *measured* is reproducible with the command next to it.

## The system

| Part | 2024 implementation | Source |
|---|---|---|
| Camera | IMX219-160, `jetbot.Camera` at 224×224, ~21 fps | `rf_live_detection.ipynb` |
| Labels | click where the car should steer; filename `xy_<px>_<py>_<uuid>.jpg` | `rf_data_collection.ipynb` |
| Policy | torchvision ResNet18, ImageNet weights, `fc = Linear(512, 2)` | `rf_train_model.ipynb` |
| Training | Adam lr 1e-3, MSE, batch 16, **70 epochs**, 90/10 random split, best test loss | `rf_train_model.ipynb` |
| Augmentation | ColorJitter(0.3, 0.3, 0.3, 0.3); horizontal flip (see below) | `rf_train_model.ipynb` |
| Controller | `atan2` on the predicted point, counter-steer, deadband, gain 1.4 | `rf_live_detection.ipynb` cell 13 |
| Actuation | `left_motor` = throttle, `right_motor` = steering motor. Not differential drive. | same |
| Safety stop | YOLOv5s, any box → throttle 0 | same |

The controller is ported verbatim in [`jdp/controller.py`](../jdp/controller.py) and tested
against a transcription of the notebook cell on 500 random inputs
(`tests/test_core.py::test_controller_matches_notebook_on_random_sequence`).

## The three dataset rounds (from the 2024 README)

| Round | Total | straight | left | right |
|---|---|---|---|---|
| 1 | 1800 | 800 | 600 | 400 |
| 2 | 3000 | 1000 | 1000 | 1000 |
| 3 | 4500 | 2500 | 1000 | 1000 |

The README records the counts, not the reasons. Round 2 balanced the classes; round 3
added only straights. The round-3 model is the one that drove the corridor.

## Bugs found reading the code

**1. The label normalisation is off-centre.** `get_x` returns `(px − 50) / 50` for a click on
a 224-pixel image, so the centre of the frame is x = +1.24, not 0. The formula is from an
older version of NVIDIA's JetBot road-following notebook, where the range was 0–100;
NVIDIA's current notebook uses `(px − width/2) / (width/2)`. The same symptom, a JetBot
"very biased towards right side", was reported upstream as
[NVIDIA-AI-IOT/jetbot#401](https://github.com/NVIDIA-AI-IOT/jetbot/issues/401).

**2. The flip ignores its own switch, and on these labels it is wrong.** `XYDataset` is built
with `random_hflips=False`, but `__getitem__` flips every sample with probability 0.5
anyway and sets `x = -x`. With centred labels that is correct. With `(px − 50)/50` labels the
mirrored click is `2.46 − x`, so every flipped sample's x is wrong by 2.46
(`tests/test_core.py::test_2024_flip_is_wrong_by_2p46`).

The two bugs partly cancel. The MSE-optimal output under that flip is the average of the
true and wrong labels, which is `(px − 111.5)/50`: centred again, but 2.24× steeper than
intended. y is never flipped, so it keeps `(py − 50)/50`.

**3. With those y labels the controller runs bang-bang.** The controller computes
`y = (0.5 − y_label)/2`. Any click below pixel row 75 makes that negative, so `atan2(x, y)`
lands beyond ±90° and the 1.4 gain saturates the steering motor. Most floor targets are
below row 75.

*Measured* with perfect labels and no learning at all: the expert's own clicks, encoded
three ways, driven through the verbatim controller on 20 held-out corridors in both
directions (`python scripts/eval.py --policy expert --label-mode <mode>`):

| Labels | Complete | Steering saturated | Weave (crossings/m) | Failures |
|---|---|---|---|---|
| centred `(px−112)/112` | 40/40 | 0.1% | 0.026 | — |
| 2024 `(px−50)/50` | 9/40 | 95% | 0 | 31 off the **right** side |
| 2024 after the flip, i.e. what a perfect 2024 net learns | 40/40 | 94% | 0.147 | — |

So the flip bug is what made the 2024 car drivable at all. The steering it produced was
on/off almost all the time, with 5.8× the weaving of the centred version. The learned
bug-faithful arm in the main experiment tests whether a trained network behaves this way.

**4. The detector saw BGR.** `jetbot.Camera` delivers BGR. The steering net was trained on
BGR too (`image.numpy()[::-1]`), so it was consistent. YOLOv5 was trained on RGB and fed the
same BGR frame, so red stop signs reached it as blue.

**5. The detector trained on crops.** The 2024 stop-sign set is a Roboflow export of German
Traffic Sign Recognition Benchmark crops. Median image 49×49 px, and every box covers at
least 92% of its image. A detector trained on that has never seen a sign inside a scene.
Its 0.995 mAP50 came from a validation split of the same kind of crops.

**6. The detector ran at 640 on a 224 frame, in FP32.** The 224×224 camera frame was stretched
to 640×640 before YOLO, about 8× the pixels with no added information. `DetectMultiBackend`
defaults to `fp16=False`, so it ran in FP32, in series with the policy on every frame.

## What the notebooks got right

The steering net's input path was consistent between training and the car, BGR and
normalisation included. The split between a learned waypoint and a hand-written
controller made failures attributable. And the round-3 decision, to make training look
like driving rather than like a balanced classification set, is the right instinct. The
experiment in this repository tests how much of the round-to-round change came from that
decision and how much from the label bug.
