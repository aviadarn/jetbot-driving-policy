# Sim-to-real contract

What the simulator matches on purpose, what it guesses, and how each guess gets replaced
by a measurement on the car. Everything lives in [`jdp/config.py`](../jdp/config.py).

## Matched to the car by construction

| | Simulator | Car (2024 code) |
|---|---|---|
| Policy input | 224×224, BGR, /255 | `jetbot.Camera` 224×224 BGR |
| Normalisation | ImageNet mean/std applied to BGR channels, inside the model | same quirk, in `preprocess()` |
| Policy | ResNet18, `fc = Linear(512, 2)` | same |
| Controller | `jdp/controller.py`, verbatim port | `rf_live_detection.ipynb` cell 13 |
| Outputs | throttle channel + steering channel, ±1 | `left_motor` / `right_motor` |
| Label | pixel the car should steer toward | click in the same frame |

The exported ONNX takes the camera's BGR/255 tensor directly, so the Nano loop does no
normalisation of its own.

## Guessed, and how to measure it

| Parameter | Value | Where it came from | How the car replaces it |
|---|---|---|---|
| Lens field of view | 110° diagonal, equidistant | fitted by eye to the 2024 frames. The lens is sold as 160°, but rendering at 160° looks far wider than the 2024 frames | checkerboard calibration on the IMX219 |
| Camera height / tilt | 0.40 m, 26° down | horizon position in the 2024 frames | tape measure + the checkerboard pose |
| Steering actuator | first-order lag, τ = 0.2 s, ±0.45 rad | unknown: DC motor or servo | step test: command ±1, film the wheels at 60 fps |
| Speed per throttle | 2.0 m/s per unit (0.34 → 0.68 m/s) | typical ride-on | time the car over 5 m at throttle 0.34 |
| Wheelbase | 0.65 m | typical ride-on | tape measure |
| Corridor width | cones at ±0.9 m | 2024 frames | the course itself |
| Loop latency | 1 frame (50 ms) by default | the Nano loop measures 23 ms from frame arrival to command | `nano/loop.py`, and a GPIO LED in view for glass-to-command |

## Known gaps that stay gaps

- **Scene content.** Terrazzo, cones and box clutter are procedural. The real hallway had
  windows, people and reflections. Colour, gamma and noise are randomised per episode to
  cover some of this, not all of it.
- **Motion blur and rolling shutter.** Not simulated.
- **Kinematic car.** No tyre slip, no contact physics. Failure is geometric: the car body
  reaching the cone line, or turning back.
