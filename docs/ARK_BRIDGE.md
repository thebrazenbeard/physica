# ARK dry-run motion bridge (experimental)

Source inspiration: [vera-ark](https://github.com/thebrazenbeard/vera-ark),
whose `VERA_ARK_ACTION_INTENTION_V1` provides bounded, dry-run actions.
This is an independently implemented optional adapter in Physica; it adds
**no runtime dependency** on vera-ark, PS Remote Play, or any driver.

`physica.ark_bridge.predict_ark_move` accepts a full V1 intention-shaped
mapping, explicitly provided 2D world position, a stick-axis-to-world-velocity
matrix, an optional swept-disc radius, and static AABB obstacles. It returns
a short-horizon endpoint and conservative potential overlap identifiers.

**Critical qualification:** a joystick axis does not encode movement in world
metres. The caller-supplied matrix is a hypothesis unless justified through
measured observations and camera orientation. A straight segment is not a
realistic ARK locomotion model; walls, momentum, traversal modes, lag and camera
rotation are not modeled. The swept AABB returns candidate obstacles, never
narrow-phase contact or safe navigation. Intent metadata is checked for shape,
basic bounds and dry-run, but this adapter is not an authenticity verifier and
does not replace validation in vera-ark itself.

The adapter never transmits inputs. Next research step: collect *synthetic or
authorized dry-run observations* to estimate speed/camera transforms and compare
prediction residuals on held-out trajectories before increasing model fidelity.
Do not describe results as calibrated until independent motion evidence exists.
