# Presentation Script for an AI Slide Maker

Paste everything below the line into an AI presentation tool (Gamma, Canva Magic Design, Beautiful.ai, PowerPoint Copilot or similar). Each slide lists its title, the text on the slide, the visual to use and the speaker notes. Plot images are public links to this repository, so most tools can import them directly; if a tool cannot, download the PNGs from the `results/` folder and insert them by hand.

---

## Instructions for the slide maker

Create a 12-slide, 16:9 presentation for a university quantum computing hackathon (Interstate Quantum Hackathon 2026, beginner "Rising-star" category, QAOA track). The team is **Quantum 5**. The audience is judges and students; keep slides clean and readable, with at most 4 short bullets per slide and large numbers where a slide has a key result. Use a calm, modern style: dark navy or white background, one teal accent colour, a simple sans-serif font. Do not invent numbers or claims; use only the text given. Put the speaker notes in each slide's notes section. Total talk length: about 8 minutes.

---

## Slide 1: Title

**Title:** Charging Electric Vehicles Overnight with a Quantum Algorithm
**Subtitle:** QAOA on simulators and two real IBM quantum computers
**Text:** Team Quantum 5 · Sakxam Shrestha, Riya Prasai, Mariam, Samyog Karki, Tesfaye Kefene · Interstate Quantum Hackathon 2026 · Rising-star category
**Visual:** a simple illustration of an electric vehicle plugged into a charger at night, with a faint circuit pattern.

**Speaker notes (Sakxam):** Hi, we are Quantum 5. Two weeks ago none of us had built a quantum circuit. Today we'll show how we used a quantum optimisation algorithm to schedule electric-vehicle charging, and what happened when we ran it on real IBM quantum computers.

## Slide 2: The problem

**Title:** Three vehicles, four hours, one good plan
**Bullets:**
- A depot charges 3 electric vehicles between 10 pm and 2 am
- Electricity costs 32, 14, 9 and 21 cents/kWh across the four hours
- Two vehicles need 2 hours of charge; the van needs 1
- Goals: low bill, no demand spike, fewer battery start/stop cycles

**Visual:** a 3-row × 4-column grid (vehicles × hours), with the price above each hour column and a few cells shaded to show one charging plan.

**Speaker notes (Riya):** Each vehicle can charge or not in each hour, so there are 12 yes-or-no choices. We want the plan that keeps the bill low, avoids everyone charging at once, and is gentle on the batteries. With 12 choices there are 4,096 combinations, but only 144 of them give every vehicle the hours it needs.

## Slide 3: Why we chose this

**Title:** Why EV charging?
**Bullets:**
- More EVs plug in every night, and charging all at once raises bills and grid strain
- The hackathon theme included energy and electric grids
- The starter notebook solved Max-Cut; we wanted a problem with real rules

**Visual:** an icon row: car, lightning bolt, price tag.

**Speaker notes (Riya):** The starter notebook solves a textbook graph problem. We wanted to see what changes when the quantum algorithm has to follow real rules, like "this van charges for exactly one hour". That turned out to be the most interesting part of the project.

## Slide 4: How QAOA works

**Title:** How the quantum algorithm searches for a plan
**Bullets:**
- Turn the cost of a plan into an energy on 12 qubits
- A quantum circuit mixes all plans, then favours low-cost ones
- A classical optimiser tunes the circuit's angles
- Measure many times; good plans should come up most often

**Visual:** a left-to-right loop diagram: "Charging problem → Cost function → Quantum circuit → Measure plans → Classical optimiser → back to Quantum circuit".

**Speaker notes (Samyog):** QAOA is a hybrid method. The quantum circuit proposes plans, and a normal computer adjusts the circuit's settings so that cheap plans appear more often. We check every answer against the exact best plan, which we can find by brute force because the problem is small.

## Slide 5: Our first wall

**Title:** The starter mixer kept breaking the rules
**Bullets:**
- The starter "X mixer" can flip any qubit, so most answers were invalid
- Adding a penalty for broken rules helped, but was hard to tune
- Best penalty: 78% valid answers, but the best plan only 1.8% of the time

**Visual:** a short before/after: a charging grid where one vehicle has 0 hours (marked with a red X) labelled "invalid plan".

**Speaker notes (Samyog):** Our first attempt used the mixer from the starter notebook. Almost every answer broke a rule, for example a vehicle that never charged. A penalty helped but still wasted many shots, and the best plan appeared no more often than if we just picked a valid plan at random.

## Slide 6: The fix: the XY mixer

**Title:** Building the rules into the circuit
**Bullets:**
- The XY mixer moves a charging hour within the same vehicle
- Each vehicle's number of charging hours never changes
- So every answer is a valid plan
- We learned this from Hadfield et al. (2019) and Wang et al. (2020)

**Visual:** an arrow showing one shaded cell sliding from hour 1 to hour 3 within a single vehicle's row.

**Speaker notes (Mariam):** The idea that changed our results: instead of punishing broken rules, use a mixer that cannot break them. The XY mixer swaps a charging hour from one slot to another inside the same vehicle. We start it from an equal mix of all valid plans, called a Dicke state.

## Slide 7: Result 1: simulation

**Title:** 100% valid plans in simulation
**Big number:** 100% valid · best plan found 10.7% of the time
**Bullets:**
- Penalty method: 78% valid, best plan 1.8%
- XY mixer: 100% valid, best plan 10.7%, almost 4× a random valid plan

**Visual:** image https://raw.githubusercontent.com/SakxamShrestha/quant5-fisk-IQ-hackathon/main/results/01_mixer_penalty/summary_bars.png

**Speaker notes (Mariam):** Here is the comparison on a perfect simulator. The left chart is the share of valid answers; the right chart is how often the single best plan appears. The XY mixer is fully valid and finds the best plan about 4 times more often than random guessing among valid plans.

## Slide 8: We explored further

**Title:** Bigger problems and different optimisers
**Bullets:**
- Scaled from 12 to 25 qubits; quality drops as problems grow
- Angles learned on 12 qubits transfer to bigger problems and save hundreds of optimiser steps
- With realistic measurement noise, the Powell optimiser held up best
- Also mapped every best trade-off plan between cost, peak and battery wear

**Visual:** image https://raw.githubusercontent.com/SakxamShrestha/quant5-fisk-IQ-hackathon/main/results/03_scaling/approx_ratio_vs_qubits.png

**Speaker notes (Tesfaye):** The hackathon suggested exploring system size, optimisers and mixers, so we did all three. As the problem grows the answers get worse, but we found a shortcut: reusing the angles from the small problem gets a bigger problem to a good answer in 1 to 68 steps instead of around 1,000.

## Slide 9: Result 2: real quantum computers

**Title:** Running on two real IBM quantum computers
**Big number:** 2.5–3× better than random, on both chips
**Bullets:**
- Ran on ibm_fez and ibm_kingston (156 qubits each), 41 seconds of free quantum time in total
- Noise cut valid answers to 15–35%
- After discarding invalid answers, the best plan appeared 6.9% (fez) and 8.4% (kingston) of the time, against 2.8% by chance

**Visual:** image https://raw.githubusercontent.com/SakxamShrestha/quant5-fisk-IQ-hackathon/main/results/07_hardware_extra/chips_fez_vs_kingston.png

**Speaker notes (Tesfaye):** This was the most exciting part. Real chips are noisy: our main circuit has 357 two-qubit gates, and noise turned most answers invalid. But when we kept only the valid answers, the circuit still picked the cheapest plan 2.5 to 3 times more often than chance, on two different chips. The noisy simulator had predicted about twice as many valid answers as the real chip gave us.

## Slide 10: Result 3: error mitigation

**Title:** Can we undo the noise?
**Bullets:**
- Zero-noise extrapolation: run with extra noise on purpose, then extrapolate back to zero
- Valid answers rose from 16% to 40% (deeper circuit) and from 28% to 67% (shallower circuit)
- It helps, but does not reach the noise-free 100%

**Visual:** image https://raw.githubusercontent.com/SakxamShrestha/quant5-fisk-IQ-hackathon/main/results/07_hardware_extra/zne_vs_raw_summary.png

**Speaker notes (Sakxam):** We tried an error-mitigation method called zero-noise extrapolation. It more than doubled the share of valid answers, but the deeper circuit still fell well short of the ideal. The lesson: shallow circuits survive noise better, while deeper circuits find better answers, so on today's hardware you have to balance the two.

## Slide 11: What we learned

**Title:** What we learned
**Bullets:**
- Keeping answers valid is the hard part; build the rules into the circuit
- Real hardware is noisier than its simulator, but the signal survives
- Every result needs a baseline: we compared everything with the exact answer and random guessing
- Next: real grid prices, classical comparisons, and training on the noisy chip itself

**Visual:** three simple icons with one-word labels: "Rules", "Noise", "Baselines".

**Speaker notes (Riya):** As beginners, our biggest takeaway is that the design of the circuit matters as much as the algorithm. We also learned to be honest about limits: our problem is small enough for a laptop to solve instantly, so the goal was to measure how well the quantum method does, not to beat the laptop.

## Slide 12: Thank you

**Title:** Thank you
**Text:** Team Quantum 5 · Code, notebooks and full report: github.com/SakxamShrestha/quant5-fisk-IQ-hackathon
**Visual:** a QR code linking to https://github.com/SakxamShrestha/quant5-fisk-IQ-hackathon

**Speaker notes (all):** Thank you. Everything we showed is in our public repository: seven executed notebooks, the raw data from both IBM chips, and our full report. We're happy to take questions.

---

## Likely judge questions and short answers

- **Why not just use a normal computer?** For this size a laptop is faster. We used a small problem on purpose so we could check every quantum answer against the exact best plan.
- **What is the XY mixer, in one sentence?** A circuit step that moves a charging hour between time slots of the same vehicle, so the rules are never broken.
- **Why did the real chip do worse than the simulator?** Our main circuit has 357 two-qubit gates, and each adds a little error. The simulator's noise model underestimated this by about 2×.
- **What does "2.8% by chance" mean?** There are 144 valid plans and 4 of them tie for best, so a random valid plan is a best plan 4/144 = 2.8% of the time.
- **Did you use AI tools?** Yes, AI coding assistants helped write and test code and draft text. The team ran, checked and discussed every result.
