# Presentation Script for an AI Slide Maker

Paste everything below the line into an AI presentation tool (Gamma, Canva Magic Design, Beautiful.ai, PowerPoint Copilot or similar). Plot images are public links to this repository; if a tool cannot import a link, download the PNG from the `results/` folder and insert it by hand.

---

## Hard rules for the slide maker (follow exactly)

1. **Make EXACTLY 10 slides.** Not 9, not 11. Do not add a title, agenda, section, summary or Q&A slide beyond the 10 below. Do not split or merge slides.
2. **Keep text short.** At most 3 bullets per slide and at most 12 words per bullet. Do not add extra sentences, introductions or filler text.
3. **Use only the numbers and claims given here.** Do not invent statistics, quotes or results.
4. **Every slide must have a visual element** (the image, diagram, icon set or big-number callout listed for it). Visuals should take at least half of each slide.
5. **Style:** 16:9, white or dark navy background, one teal accent colour, simple sans-serif font, large readable text. Put the speaker notes in each slide's notes section, not on the slide.

Context: Interstate Quantum Hackathon 2026, beginner "Rising-star" category, QAOA track. Team **Quantum 5**. Talk length: about 7 minutes.

---

## Slide 1: Title

- **Title:** Charging EVs Overnight with a Quantum Algorithm
- **Subtitle:** Team Quantum 5 · Interstate Quantum Hackathon 2026 · Rising-star
- **Names (small):** Sakxam Shrestha · Riya Prasai · Mariam · Samyog Karki · Tesfaye Kefene
- **Visual:** full-width illustration of an electric car plugged in at night, with a faint quantum-circuit line pattern.

**Notes (Sakxam):** We're Quantum 5. Two weeks ago none of us had built a quantum circuit. We used a quantum algorithm to schedule EV charging and ran it on real IBM quantum computers.

## Slide 2: The problem

- **Title:** 3 vehicles · 4 hours · 1 best plan
- **Bullets:**
  - Goals: low bill, no demand spike, less battery wear
  - 12 yes/no choices = 4,096 combinations
  - Only 144 combinations follow the rules
- **Visual:** a 3 × 4 grid (rows: Bus A, Bus B, Van; columns: 10 pm, 11 pm, 12 am, 1 am). Price tag above each column: 32¢, 14¢, 9¢, 21¢. Shade a few cells to show one charging plan. Car, lightning and price-tag icons beside the grid.

**Notes (Riya):** A depot charges three vehicles overnight. Electricity prices change every hour. Two vehicles need two hours, the van needs one. We picked EV charging because more cars plug in every night, and the hackathon theme was energy and grids.

## Slide 3: How QAOA works

- **Title:** A quantum + classical loop
- **Bullets:**
  - Quantum circuit proposes charging plans
  - Classical optimiser tunes the circuit's angles
  - Repeat until cheap plans appear most often
- **Visual:** circular loop diagram with 4 icon nodes: "Charging problem" (car) → "Quantum circuit" (circuit icon) → "Measure plans" (bar chart) → "Optimiser" (gear) → back to "Quantum circuit".

**Notes (Samyog):** QAOA is a hybrid method. The quantum part proposes plans, a normal computer adjusts the settings, and we repeat. Because the problem is small, we can check every answer against the exact best plan.

## Slide 4: Our first wall and the fix

- **Title:** Build the rules into the circuit
- **Bullets:**
  - Starter "X mixer" mostly produced rule-breaking plans
  - Penalties helped but were hard to tune
  - XY mixer only moves hours within a vehicle
- **Visual:** two side-by-side mini grids. Left, labelled "X mixer ✗": a vehicle row with no shaded cells and a red X. Right, labelled "XY mixer ✓": an arrow sliding one shaded cell from 10 pm to 12 am in the same row, green check.

**Notes (Mariam):** The starter notebook's mixer can flip any qubit, so most answers broke a rule, like a vehicle that never charged. The fix, from Hadfield 2019 and Wang 2020: a mixer that can only move a charging hour inside the same vehicle, so the rules can never break.

## Slide 5: Result 1, simulation

- **Title:** 100% valid plans
- **Big-number callouts (three tiles):** **100%** valid (XY mixer) · **10.7%** finds the best plan · **4×** better than random
- **Visual:** image https://raw.githubusercontent.com/SakxamShrestha/quant5-fisk-IQ-hackathon/main/results/01_mixer_penalty/summary_bars.png

**Notes (Mariam):** On a perfect simulator, the XY mixer gave valid plans every time and found the single best plan 10.7% of the time. The best penalty setting reached only 78% valid and 1.8% best plan.

## Slide 6: We explored further

- **Title:** Bigger problems, different optimisers
- **Bullets:**
  - Scaled from 12 to 25 qubits
  - Reused angles: 1–68 steps instead of ~1,000
  - Powell optimiser handled noise best
- **Visual:** image https://raw.githubusercontent.com/SakxamShrestha/quant5-fisk-IQ-hackathon/main/results/03_scaling/approx_ratio_vs_qubits.png

**Notes (Tesfaye):** The hackathon suggested exploring size, optimisers and mixers, so we did all three. Bigger problems are harder, but reusing angles from the small problem saved hundreds of optimiser steps.

## Slide 7: Result 2, real quantum computers

- **Title:** Two real IBM quantum computers
- **Big-number callouts (three tiles):** **2** IBM chips (ibm_fez, ibm_kingston) · **41 s** quantum time · **2.5–3×** better than random
- **Visual:** image https://raw.githubusercontent.com/SakxamShrestha/quant5-fisk-IQ-hackathon/main/results/07_hardware_extra/chips_fez_vs_kingston.png

**Notes (Tesfaye):** Real chips are noisy: our circuit has 357 two-qubit gates, and only 15–35% of answers stayed valid. But among valid answers, the best plan came up 6.9% on fez and 8.4% on kingston, against 2.8% by chance.

## Slide 8: Result 3, undoing noise

- **Title:** Error mitigation helps, partly
- **Bullets:**
  - Zero-noise extrapolation: add noise, extrapolate to zero
  - Valid answers: 16% → 40% (deep), 28% → 67% (shallow)
  - Still short of the noise-free 100%
- **Visual:** image https://raw.githubusercontent.com/SakxamShrestha/quant5-fisk-IQ-hackathon/main/results/07_hardware_extra/zne_vs_raw_summary.png

**Notes (Sakxam):** We ran the circuit with extra noise on purpose, then worked backwards to estimate the noise-free answer. It more than doubled valid answers. Shallow circuits survive noise better; deep circuits find better answers.

## Slide 9: What we learned

- **Title:** What we learned
- **Bullets:**
  - Circuit design matters as much as the algorithm
  - Real chips are noisier than simulators, but signal survives
  - Next: real grid prices and classical comparisons
- **Visual:** three large icons in a row with one-word labels: a puzzle piece "Design", a wave "Noise", a road sign "Next".

**Notes (Riya):** Our biggest lesson as beginners: keeping answers valid is the hard part. We were also careful to compare everything with the exact answer and random guessing. A laptop solves this size instantly; our goal was to measure the quantum method honestly.

## Slide 10: Thank you

- **Title:** Thank you
- **Text:** Code, notebooks, IBM data and full report:
  github.com/SakxamShrestha/quant5-fisk-IQ-hackathon
- **Visual:** large QR code linking to https://github.com/SakxamShrestha/quant5-fisk-IQ-hackathon, with the team name "Quantum 5" beside it.

**Notes (all):** Thank you. Everything is in our public repository. We're happy to take questions.

---

## Likely judge questions (for the team, not for the slides)

- **Why not just use a normal computer?** At this size a laptop is faster. We kept the problem small so we could check every quantum answer.
- **XY mixer in one sentence?** It moves a charging hour between time slots of the same vehicle, so rules never break.
- **Why was the real chip worse than the simulator?** 357 two-qubit gates each add error; the simulator underestimated this by about 2×.
- **Where does "2.8% by chance" come from?** 4 of the 144 valid plans tie for best: 4/144 = 2.8%.
- **Did you use AI tools?** Yes, for help writing and testing code and drafting text. The team ran and checked every result.
