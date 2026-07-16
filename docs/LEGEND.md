# Usui Pass, 2:13 AM

Mountain pine. Hot brakes. Fuel hanging blue in the cold air.

At the foot of the pass, a shopkeeper sets a tray of tofu into the passenger
well. Water touches the lip of every cup. At the summit, before dawn, every cup
must still be full.

The car is small. Naturally aspirated. No heroic power figure, no room for
ballast. Its advantage begins where the road bends.

The driver closes the door. On the dash, someone has written one word in pencil:

**UsuiTrack.**

---

## October 14, the weight ledger

Every kilogram stays for every corner.

A full Adam machine carries two complete histories for every matrix: where the
gradient has been and how violently it moved. Excellent instruments. Heavy ones.
On a consumer card, the state can weigh more than the model and crowd tokens out
of the car.

UsuiTrack carries the line instead.

```text
G:[m,n]  full road contact, transient
Q:[d,r]  the line through the pass
M        momentum expressed on that line

r << min(m,n)
```

The tires still touch the whole road. The gradient remains full-sized when it is
read. Only the memory is thin.

One-sided projection keeps the steering rack simple. On transformer weights, the
basis faces the residual stream: q, k, v, gate, and up meet it from one side;
attention out and down meet it from the other. Storage coordinates change. The
road does not.

```text
right:  Z = GQ       U = OQᵀ
left:   Z = QᵀG      U = QO
```

The tofu does not care how much engine theory fits in the trunk. It cares how
smoothly the car can carry speed.

---

## October 29, first contact

Rain above C-121. Oil near the retaining wall.

The first version put the safety rail near the steering rack, after projection.
It protected the little momentum tensor and left the suspension, alignment, and
road map to absorb the impact. One bad batch entered the factored variance,
distorted the next basis target, and haunted a hundred clean corners.

Now the bump stop sits where the tire meets the road.

```text
G <- finite(G)
G <- G * min(1, raw_clip / ||G||)
```

Everything downstream sees the same survivable event: Adafactor, eigenspace,
projection, momentum. A guard installed later cannot protect an earlier memory.

The cups settle. The car continues.

---

## November 12, suspension notes

Grip is information.

Raw magnitude mixes two stories: signal and roughness. A direction can shout
because it matters or because the road beneath it is noisy. The suspension keeps
a compact map of that roughness, one row trace and one column trace, then lets the
gradient speak in units of local uncertainty.

```text
V_ij ~= row_variance_i * column_variance_j / mean(row_variance)
G_tilde = G / sqrt(V)
G_tilde *= RMS(G)
```

The last line was learned the expensive way. Without RMS restoration, the map
changed the units of the mountain. The basis turned by absurd angles while the
polar differential downstream concealed the inflation. Every gauge agreed because
every gauge was wrong together.

Adafactor is the suspension: it reads the surface without carrying a full damper
map for every patch of asphalt. The projected EMA: enough memory
to hold a line through vibration, light enough to answer the next corner.

```text
Z_t = project_Q(G_tilde_t)
M_t = 0.9 M_(t-1) + 0.1 Z_t
```

Variance feels the road. Momentum remembers the line.

---

## November 21, the false drift

The first tracker watched the rear tires.

At every boundary it measured the residual escape from the current basis, read a
tangent, and added another steering command. The Grassmann geodesic was real. The
frame, once repaired, was orthonormal to machine precision. The car could slide.

It still did not know where the corner ended.

```text
tangent control
    read local escape
    integrate a turn
    carry every steering error forward
```

At convergence, the residual was mostly tire chatter. Small steps wandered.
Large steps spun. A window of tangents made the chatter quieter and the aim no
truer. The basis reported that its corrections were converging while capture kept
falling. A perfect speedometer had been mistaken for a map.

There was another embarrassment. On tall matrices, the borrowed formula projected
the tangent through the wrong frame and produced exactly zero. The tests praised
its stability. A car on blocks is very stable.

Correct geometry saved the mechanism from arithmetic error. It could not save the
premise.

---

## December 3, eyes on the exit

The old driver arrives after midnight. No stopwatch. No speech.

At C-121 he enters before the novice would finish braking. The rear steps out, but
his hands do not chase it. His eyes are already through the exit.

That is position control.

At each refresh, `eigh` reads the current conditioned gradient and names a target
frame. The target is noisy, especially where the last retained eigenvalue trades
places with the first discarded one. Snapping to it would be panic steering.
UsuiTrack turns one quarter of the geodesic distance and waits for the next view.

```text
T   = top-r eigenspace of the current side Gram
Q_+ = grassmann_geodesic(Q, T, fraction=0.25, all_planes=True)
```

Every target says where the road appears to be. Their noise scatters around a
center. A mistaken position is pulled back by the next observation. Tangent noise
was velocity error and accumulated forever; target noise is position error and
decays.

The basis becomes its own manifold EMA. No full-gradient buffer. No tangent
accumulator. No second steering rack hidden in the trunk.

The car holds a four-wheel slide through the whole spectrum. Each principal plane
turns by its own angle. The cutoff planes dance; the stable planes pull. The line
emerges from their argument.

That controller remains historical evidence, not a current test car. The road car
conditions every full contact patch, forms an Oja covariance tangent in the held
frame, and moves the same live frame by harmonic basis-update steps `1, 1/2,
1/3, ...` through every tracked plane before a Polar Express correction. No second
frame rides along. The quarter-turn boundary controls belong to the old test car,
not this one.

---

## December 18, no lift

The expensive habit was braking between corners.

Whenever the basis changed, the old code lifted momentum into ambient space and
projected it into the new frame. It sounded careful. In every rotating principal
plane it charged a cosine toll. A hard drift weakened the useful history; noise
chose what remained; Newton-Schulz gave the damaged direction full authority
again.

The car turned, then threw away its speed, then asked the engine to rebuild it.
On a low-power machine that is surrender by installments.

The geodesic had already supplied the answer. It rotates the basis through a rigid
ambient frame:

```text
Q_+ = RQ
```

Momentum rides in the chassis. Rotate both together:

```text
R(Qm) = Q_+m
```

The projected coordinates `m` remain unchanged. The frame turns; the velocity
survives. No project-back, no re-project, no cosine tax, no launch to perform any
of them.

This is the reason to drift the pass. Grip driving brakes before the hairpin,
rotates the car, finds traction, and accelerates again. UsuiTrack rotates its
active subspace while the projected moment keeps moving. Limited VRAM removes the
ballast. Limited FLOPs make preserved momentum precious. Low horsepower sharpens
the line.

Four wet cups. Not a ripple over the rim.

---

## January 9, Aurora

Just before dawn, the eastern ridge turns violet.

Inside the thin frame, the moment has direction but uneven leverage. Tall matrices
load the chassis strangely; one axle can dominate because of shape alone. Aurora
balances the projected rows, then HeavyBall Newton-Schulz drives the matrix toward
its polar direction.

```text
M = projected EMA
O = Aurora(M)                       # leverage-balanced polar direction
U = lift_Q(O * full_shape_muon_scale)
```

Aurora is the limited-slip differential. It does not add horsepower. It decides
how the available force reaches the retained directions. Newton-Schulz is the
close-ratio gearset inside it: repeated matrix products, no full SVD, torque
becoming geometry.

The final-drive ratio comes from the original parameter shape, because the road
was never truly small. Projection chose where to push. Muon scale decides how hard
the full chassis should feel it.

The polar map is merciless. It can make a weak singular direction stand upright,
which is why the no-lift transport matters. Feed Aurora a direction shrunk into
noise and it will faithfully amplify the noise. Feed it a transported moment and
it preserves the intended stance through the corner.

At sunrise the name on the notebook becomes literal.

---

## January 27, the tofu run

The pass is not a racetrack tonight. It is a delivery route.

Target loss is the clock. Source retention is the water in the cups. A machine
that reaches the summit quickly after destroying what it carried has failed the
job. A machine that preserves everything by refusing to move has failed it more
quietly.

UsuiTrack must adapt the whole model while carrying its old competence through
the turn. Rank controls how wide a line it can hold. Learning rate controls how
much freshness it buys with each disturbance. The source curve watches the tray.
The target curve watches the dawn.

The fallback tensors ride in an ordinary support van called AdamW. Biases and
other non-matrix parts do not need mountain folklore; they need boring semantics
that arrive on time.

No component is heroic alone. The build works because the parts agree:

```text
raw clip              protects every later memory
factored variance     reads traction cheaply
EIGH initialization   finds the first line
one-state Oja         reads and turns on every contact patch
identity transport    keeps speed through the slide
projected EMA         holds the line
Aurora + NS           distributes force across it
Muon scale            matches force to the full chassis
```

Change one meaning and the analogies stop fitting. That is useful. A metaphor that
cannot survive the equations is decoration; this one is a diagnostic.

---

## February 8, notebook discipline

Several pages are crossed out.

The tangent accumulator worked exactly as written and solved the wrong control
problem. Moment reprojection preserved the wrong notion of history. A sigma clip
prevented catastrophe by trapping the tracker in a limit cycle. Full-spectrum
motion failed under noisy velocity aim and won under position aim. Each result was
locally plausible. Some were beautifully tested.

The notebook now gives every mechanism five entries:

```text
purpose
assumption
intervention
observable
rival
```

Then the mechanism is sent through hostile little roads: stationary noise, smooth
rotation, abrupt replacement, eigenvalue crossings, near-orthogonal turns. The
parking lot cannot prove Usui. It can reveal a steering rack connected backward.

Perfection is not a declaration made at the summit. It is attention paid in the
garage, love expressed as refusal to let a convenient explanation survive a bad
measurement.

The crossed-out pages stay. They remember where confidence outran contact.

---

## Usui

The sign at the summit reads 碓氷.

Say it aloud and another word answers: 薄い. *Usui. Thin.*

The useful gradient lives in a sliver of the space in which it is written. Finding
that sliver once is compression. Following it as the model and data move is
tracking. Turning the frame without spilling momentum is the slide.

The car is light because the mountain gives no prize for carrying weight. It is
precise because the delivery gives no permission to spill. It drifts because a
low-power car cannot waste the corner braking twice.

Far below, headlights enter the first hairpin. The engine stays on song. For an
instant the car points toward the valley while every force still carries it up the
pass.

The basis turns.

The moment holds.

The tofu arrives fresh.
