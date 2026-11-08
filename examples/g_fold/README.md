# G-FOLD

This example solves the guided fuel-optimal landing diversion (G-FOLD) problem from [1] for a range of time horizons and keeps the one with the minimum fuel use. The coordinate system is +X up, +Y east, +Z north.

Equation (55) of [2] gives a conservative time horizon search region using the fuel mass and thrust limits. The sections below derive a region from the dynamics, including the planet rotation terms, which is tighter for this example.

## Dynamics

Let q be the position, v be the velocity, u be the thrust acceleration, σ be its magnitude limit, z = ln(m) be the log mass, g be the gravitational acceleration, and ω be the planet's angular velocity, all expressed in the rotating planet frame. Let α be the fuel consumption rate, m_wet be the initial mass, m_dry be the dry mass, ρ₁ and ρ₂ be the thrust magnitude limits, θ be the thrust pointing limit, and v_max be the velocity limit. Let S(ω) be the cross product matrix of ω, so S(ω)a = ω×a.

Equations (1) and (2) of [1] give the dynamics ẋ = A(ω)x + B(g + T_c/m) of the state x = (q, v), where T_c is the thrust and m is the mass. The velocity rows are v̇ = g + T_c/m − 2S(ω)v − S(ω)²q. Substituting u = T_c/m and writing S(ω) as a cross product gives (1), and equation (33) of [1] gives the log mass dynamics (2).

```
  v̇ = g + u − 2ω×v − ω×(ω×q)                                          (1)
  ż = −ασ                                                             (2)
```

Equation (34) of [1] gives the thrust magnitude limit |u| ≤ σ and the thrust pointing limit uₓ ≥ cos(θ)σ, equation (35) of [1] gives the thrust limits ρ₁e⁻ᶻ ≤ σ ≤ ρ₂e⁻ᶻ, and equation (12) of [1] gives the velocity limit |v| ≤ v_max. θ lies within [0°, 180°].

## Time horizon upper bound

Gravity adds g t_f of velocity over the flight, which the thrust must cancel on top of the change from v₀ to v_f. The velocity change the thrust must provide therefore grows with t_f, while the fuel caps the velocity change it can provide. t_max is where the two meet, since any longer flight would need more fuel than the lander carries.

### Fuel limit

Integrating the mass dynamics (2) from 0 to t_f gives

```
  z(t_f) = z(0) − α∫σ dt
  ∫σ dt = (z(0) − z(t_f))/α
```

The initial mass gives z(0) = ln(m_wet), and the final mass constraint gives z(t_f) ≥ ln(m_dry), so

```
  ∫σ dt ≤ ln(m_wet/m_dry)/α = Δv_max                                  (3)
```

where Δv_max is the largest velocity change the fuel can provide.

### Inertial frame dynamics

The planet rotation terms are fictitious accelerations from the rotating planet frame, so we'll integrate the dynamics in an inertial frame instead. Let R(t) rotate the planet frame at time t into the inertial frame that coincides with the planet frame at t = 0, so R(0) = I. The planet rotates at ω, so R satisfies

```
  Ṙ = RS(ω)                                                           (4)
```

This is a linear differential equation with the initial condition R(0) = I, so its solution is

```
  R(t) = exp(S(ω)t)
```

Let v_I be the velocity relative to the inertial frame, expressed in the planet frame.

```
  v_I = v + ω×q                                                       (5)
```

Differentiate it and substitute q̇ = v.

```
  v̇_I = v̇ + ω×v
```

Substitute v̇ = g + u − 2ω×v − ω×(ω×q) from (1).

```
  v̇_I = g + u − ω×v − ω×(ω×q)
```

Factor ω× out of the last two terms and substitute (5).

```
  v̇_I = g + u − ω×v_I
```

Rv_I is the inertial velocity expressed in the inertial frame. Differentiate it with the product rule.

```
  d/dt(Rv_I) = Ṙv_I + Rv̇_I
```

Substitute Ṙ = RS(ω) from (4) and factor out R.

```
  d/dt(Rv_I) = R(S(ω)v_I + v̇_I)
```

Substitute S(ω)v_I = ω×v_I and v̇_I = g + u − ω×v_I. The planet rotation terms cancel.

```
  d/dt(Rv_I) = R(g + u)
```

Integrate from 0 to t_f.

```
  R(t_f)v_I(t_f) − R(0)v_I(0) = ∫R(g + u) dt
```

Split the integral and pull the constant g out of the first one.

```
  R(t_f)v_I(t_f) − R(0)v_I(0) = (∫R dt)g + ∫Ru dt
```

Let R_int(t) be the integrated rotation.

```
  R_int(t) = ∫₀ᵗ R(τ) dτ                                              (6)
```

Let v_I,0 = v₀ + ω×q₀ and v_I,f = v_f + ω×q_f be (5) at 0 and t_f. Substitute them, R(0) = I, and R_int(t_f) for the integral.

```
  R(t_f)v_I,f − v_I,0 = R_int(t_f)g + ∫Ru dt                          (7)
```

### Closed form rotation

Evaluating R(t) and R_int(t) inside an optimization problem needs closed forms of them. R(t) = exp(S(ω)t) is the rotation by the angle |ω|t about the planet's rotation axis ω̂ = ω/|ω|, so Rodrigues's rotation formula gives

```
  R(t) = I + sin(|ω|t)S(ω̂) + (1 − cos(|ω|t))S(ω̂)²
```

Substitute S(ω̂) = S(ω)/|ω|.

```
  R(t) = I + (sin(|ω|t)/|ω|)S(ω) + ((1 − cos(|ω|t))/|ω|²)S(ω)²
```

Let

```
  k₁(t) = sin(|ω|t)/|ω|
  k₂(t) = (1 − cos(|ω|t))/|ω|²
```

Substitute them.

```
  R(t) = I + k₁(t)S(ω) + k₂(t)S(ω)²                                   (8)
```

Substitute (8) into (6).

```
  R_int(t) = ∫₀ᵗ (I + k₁(τ)S(ω) + k₂(τ)S(ω)²) dτ
```

Integrate term by term.

```
  R_int(t) = tI + (∫₀ᵗ k₁(τ) dτ)S(ω) + (∫₀ᵗ k₂(τ) dτ)S(ω)²
```

Integrate k₁ and k₂.

```
  ∫₀ᵗ k₁(τ) dτ = (1 − cos(|ω|t))/|ω|² = k₂(t)
  ∫₀ᵗ k₂(τ) dτ = (t − sin(|ω|t)/|ω|)/|ω|² = (t − k₁(t))/|ω|²
```

Let

```
  k₃(t) = (t − k₁(t))/|ω|²
```

Substitute the integrals.

```
  R_int(t) = tI + k₂(t)S(ω) + k₃(t)S(ω)²                              (9)
```

k₁, k₂, and k₃ divide by |ω|, so these closed forms require ω ≠ 0. Without planet rotation, (4) and (6) give R(t) = I and R_int(t) = tI directly.

### Bound

The ∫Ru dt term in (7) is the velocity change the thrust provides in the inertial frame. Let

```
  Δv_u(t) = R(t)v_I,f − v_I,0 − R_int(t)g                             (10)
```

Then (7) becomes

```
  Δv_u(t_f) = ∫Ru dt
```

The magnitude of an integral is at most the integral of the magnitude, rotations preserve magnitude, |u| ≤ σ, and (3) bounds ∫σ dt, so

```
  |Δv_u(t_f)| = |∫Ru dt| ≤ ∫|Ru| dt = ∫|u| dt ≤ ∫σ dt ≤ Δv_max
```

Both sides are nonnegative, so squaring them preserves the inequality.

```
  |Δv_u(t_f)|² ≤ Δv_max²                                              (11)
```

### Feasible set

Sleipnir finds a local maximum, which is guaranteed to be the global maximum when the t_f that satisfy (11) form a single interval. Without planet rotation, (10) gives Δv_u(t) = v_f − v₀ − tg, so |Δv_u(t)|² is a convex quadratic and they do. With planet rotation, we'll show that they do by bounding them within a bracket, then showing |Δv_u(t)|² is convex within it.

#### Bracket

R(t) and its inverse R(t)ᵀ are rotations about ω̂, so they leave ω̂ fixed. Transpose R(t)ᵀω̂ = ω̂.

```
  ω̂ᵀR(t) = ω̂ᵀ                                                         (12)
```

Project (6) onto ω̂ and substitute (12).

```
  ω̂ᵀR_int(t) = ∫₀ᵗ ω̂ᵀ dτ = tω̂ᵀ                                        (13)
```

Project (10) onto ω̂.

```
  ω̂ᵀΔv_u(t) = ω̂ᵀR(t)v_I,f − ω̂ᵀv_I,0 − ω̂ᵀR_int(t)g
```

Substitute (12) and (13).

```
  ω̂ᵀΔv_u(t) = ω̂ᵀ(v_I,f − v_I,0) − tω̂ᵀg                                (14)
```

The projection of a vector onto a unit vector is at most the vector's magnitude, so

```
  |Δv_u(t)| ≥ |ω̂ᵀΔv_u(t)|
```

Substitute (14).

```
  |Δv_u(t)| ≥ |ω̂ᵀ(v_I,f − v_I,0) − tω̂ᵀg|
```

Apply the reverse triangle inequality |a − b| ≥ |b| − |a|.

```
  |Δv_u(t)| ≥ |t||ω̂ᵀg| − |ω̂ᵀ(v_I,f − v_I,0)|
```

When ω̂ᵀg ≠ 0, the right-hand side exceeds Δv_max for all |t| > t_b where

```
  t_b = (Δv_max + |ω̂ᵀ(v_I,f − v_I,0)|)/|ω̂ᵀg|                          (15)
```

so (11) fails there, and every t_f that satisfies (11) lies in [−t_b, t_b]. When ω̂ᵀg = 0 (the rotation axis is perpendicular to gravity), this argument gives no bracket.

#### Convexity

Let h(t) = |Δv_u(t)|². Differentiating h(t) = Δv_u(t)ᵀΔv_u(t) twice with the product rule gives

```
  h''(t) = 2|Δv_u'(t)|² + 2Δv_u(t)ᵀΔv_u''(t)                          (16)
```

Differentiate (10).

```
  Δv_u'(t) = Ṙ(t)v_I,f − (d/dt R_int(t))g
```

Substitute Ṙ = RS(ω) from (4) and d/dt R_int(t) = R(t), which follows from (6).

```
  Δv_u'(t) = R(t)S(ω)v_I,f − R(t)g
```

Factor out R(t).

```
  Δv_u'(t) = R(t)(S(ω)v_I,f − g)
```

Let c = S(ω)v_I,f − g, which is constant.

```
  Δv_u'(t) = R(t)c                                                    (17)
```

Differentiate again and substitute Ṙ = RS(ω) from (4).

```
  Δv_u''(t) = R(t)S(ω)c                                               (18)
```

Rotations preserve magnitude, so (17) and (18) give

```
  |Δv_u'(t)| = |c|
  |Δv_u''(t)| = |S(ω)c|
```

Substitute S(ω)c = ω×c, whose magnitude is at most |ω||c|.

```
  |Δv_u''(t)| ≤ |ω||c|                                                (19)
```

The Cauchy–Schwarz inequality bounds the second term of (16) by

```
  Δv_u(t)ᵀΔv_u''(t) ≥ −|Δv_u(t)||Δv_u''(t)|
```

Substitute it and |Δv_u'(t)| = |c| into (16).

```
  h''(t) ≥ 2|c|² − 2|Δv_u(t)||Δv_u''(t)|
```

Substitute (19).

```
  h''(t) ≥ 2|c|² − 2|ω||c||Δv_u(t)|
```

Factor out 2|c|.

```
  h''(t) ≥ 2|c|(|c| − |ω||Δv_u(t)|)                                   (20)
```

Δv_u changes at the rate |Δv_u'(t)| = |c|, so it moves at most |c||t| from Δv_u(0).

```
  |Δv_u(t)| ≤ |Δv_u(0)| + |c||t|
```

Substitute |t| ≤ t_b from the bracket.

```
  |Δv_u(t)| ≤ |Δv_u(0)| + |c|t_b
```

Substitute it into (20).

```
  h''(t) ≥ 2|c|(|c| − |ω|(|Δv_u(0)| + |c|t_b))
```

When c ≠ 0, dividing by 2|c|² shows the right-hand side is positive when

```
  |ω|(|Δv_u(0)|/|c| + t_b) < 1                                        (21)
```

where substituting R(0) = I and R_int(0) = 0 into (10) gives Δv_u(0) = v_I,f − v_I,0.

When c ≠ 0 and (21) holds, h is strictly convex on [−t_b, t_b]. The sublevel sets of a convex function are intervals, so the t in [−t_b, t_b] that satisfy h(t) ≤ Δv_max² form a single interval. Every t_f that satisfies (11) lies in [−t_b, t_b], so when any t_f satisfies (11), they form that single interval, and its upper end t_max is the only local maximum.

This example has t_b ≈ 250 s and |ω|(|Δv_u(0)|/|c| + t_b) ≈ 0.019.

### Solving

t_max is the largest t_f that satisfies (11), so we find it by maximizing t_f subject to (11) with Sleipnir, using (8) and (9) for R(t_f) and R_int(t_f) in (10).

```
   max t_f
   t_f
  s.t. |Δv_u(t_f)|² ≤ Δv_max²
```

The feasible set is a single interval, so the local maximum Sleipnir finds is t_max.

## Time horizon lower bound

Since m ≥ m_dry, the thrust limits bound the thrust acceleration by |u| ≤ σ ≤ ρ₂/m_dry, so uₓ ≤ ρ₂/m_dry. When θ ≤ 90°, cos(θ) ≥ 0, so the thrust pointing limit gives uₓ ≥ cos(θ)σ ≥ 0, and thrust can't push downward. When θ > 90°, cos(θ) < 0, so it gives uₓ ≥ cos(θ)σ ≥ cos(θ)ρ₂/m_dry. Both cases give

```
  min(0, cos(θ))ρ₂/m_dry ≤ uₓ ≤ ρ₂/m_dry                              (22)
```

Let eₓ = [1  0  0]ᵀ be the up direction. The vertical component of (1) is

```
  v̇ₓ = gₓ + uₓ − 2eₓᵀ(ω×v) − eₓᵀ(ω×(ω×q))                             (23)
```

### Planet rotation terms

The scalar triple product identity aᵀ(b×c) = cᵀ(a×b) and the velocity limit bound the planet rotation terms by

```
  |eₓᵀ(ω×v)| = |vᵀ(eₓ×ω)| ≤ |eₓ×ω|v_max
  |eₓᵀ(ω×(ω×q))| = |(ω×q)ᵀ(eₓ×ω)| ≤ |eₓ×ω||ω||q|
```

The velocity limit also keeps the lander within v_max t of q₀ and within v_max(t_f − t) of q_f at time t, so

```
  |q| ≤ |q₀| + v_max t
  |q| ≤ |q_f| + v_max (t_f − t)
```

Averaging the two bounds cancels the v_max t terms, and substituting t_f ≤ t_max gives

```
  |q| ≤ (|q₀| + |q_f| + v_max t_max)/2                                (24)
```

Let q_bound be the right-hand side. The planet rotation terms therefore add at most

```
  a_ω = |eₓ×ω|(2v_max + |ω|q_bound)                                   (25)
```

of vertical acceleration in either direction. This example's gravity points straight down, so gₓ = −|g|, and the vertical acceleration lies within [−a_fall, a_brake] where

```
  a_fall = |g| − min(0, cos(θ))ρ₂/m_dry + a_ω                         (26)
  a_brake = ρ₂/m_dry − |g| + a_ω                                      (27)
```

### Minimum time descent

The minimum time to null the vertical position and velocity is a fall at a_fall up to some peak downward speed s_peak, followed by braking at a_brake. This requires a_brake > 0, so the thrust can overcome gravity and the planet rotation terms. Let Δh be the initial height above the target and s₀ be the initial downward speed.

Under a constant acceleration a, changing speed from s_a to s_b takes time (s_b − s_a)/a and distance (s_b² − s_a²)/(2a). During the fall, the lander accelerates downward at a_fall from s₀ to s_peak, which takes

```
  t₁ = (s_peak − s₀)/a_fall
  d₁ = (s_peak² − s₀²)/(2a_fall)
```

During the braking, the lander slows down from s_peak to 0 at a_brake, which takes

```
  t₂ = s_peak/a_brake
  d₂ = s_peak²/(2a_brake)
```

The two phases must cover the height Δh, so

```
  d₁ + d₂ = Δh
  (s_peak² − s₀²)/(2a_fall) + s_peak²/(2a_brake) = Δh
```

Multiply by 2, collect the s_peak² terms, and solve for s_peak.

```
  s_peak²(1/a_fall + 1/a_brake) = 2Δh + s₀²/a_fall
  s_peak = √((2Δh + s₀²/a_fall)/(1/a_fall + 1/a_brake))               (28)
```

The fall phase requires s_peak ≥ s₀, which holds when the lander starts moving upward (s₀ ≤ 0) or can brake to a stop within the height (s₀²/(2a_brake) ≤ Δh). Otherwise, it passes below the target height, which the glide slope constraint forbids.

The minimum time is the sum of the phase durations.

```
  t_min = t₁ + t₂
  t_min = (s_peak − s₀)/a_fall + s_peak/a_brake                       (29)
```

## Number of control intervals

Following equation (57) of [2], we convert the time horizon bounds into a search region for the number of control intervals N with timestep Δt.

```
  ⌈t_min/Δt⌉ ≤ N ≤ ⌊t_max/Δt⌋
```

## References

[1] Açıkmeşe et al., "Lossless Convexification of Nonconvex Control Bound and Pointing Constraints of the Soft Landing Optimal Control Problem", 2013. http://www.larsblackmore.com/iee_tcst13.pdf

[2] Açıkmeşe et al., "Convex Programming Approach to Powered Descent Guidance for Mars Landing", 2007. https://sci-hub.st/10.2514/1.27553
