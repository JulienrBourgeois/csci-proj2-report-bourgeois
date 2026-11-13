---
title: Project 2: Flow Matching
---

**Name:** Julien Bourgeois

I trained a small network that moves 2D points. It starts from Gaussian noise and learns a velocity that carries those points onto a shape. The shape is a cloud of coordinates, not a picture. Part 1 uses a thick ring, so I can see the idea before the shape gets fiddly. Part 2 uses a letter J that I drew in code, with the same particles tracked and colored the whole way. Part 3 asks the question in Flow Straight and Fast: if I retrain on the paths my own model already produced, do those paths get straighter, and can I then use fewer network evaluations?

The shorter PDF is [report.pdf](report.pdf). The numbers below are the same run. The animation is in Part 2.

Time 0 is noise and time 1 is data. A training pair is `x_t = (1 - t) x_0 + t x_1`, and the velocity I regress is `u = x_1 - x_0`. The loss on a batch is the mean of `||v(x_t, t) - u||^2`. When I generate, the network only sees the current point and the time. It never sees the endpoint. I move the points with explicit Euler: start at a fresh Gaussian draw, then add `dt * v(z, t)` with `t = k / K` and `dt = 1 / K`. One step is one network evaluation per particle, so P particles and K steps cost `P * K` evaluations.

## Setup

I ran this on cpu. CUDA was not available. MPS was available, and I left it off so the timing would be a plain CPU number. The banner printed torch 2.14.0, numpy 1.23.5, and matplotlib 3.7.2. The network takes `(x, y, t)`, then three hidden layers of width 128 with SiLU, and it outputs a 2D velocity. The script printed **33,794** parameters. Adam, learning rate 0.001, batch size 256. Part 1 is seed 0 for 12,000 steps. Part 2 is seed 7 for 20,000 steps. Part 3 repeats the letter on seeds 7, 21, 42. Three seeds is enough to see whether the pattern repeats. It is not a big sweep.

For the letter, I score clouds with one histogram and I do not change it. It is 32 by 32 bins on the square [-3.5, 3.5], so each bin is 0.21875 wide. One extra bin catches every point that falls outside that square, and those points stay in the score. Each comparison uses 8,000 generated points and 8,000 points from one frozen letter sample, seed 20262. The held-out Gaussian is seed 20261. Those starts are not the sources I used to train reflow. The score is the L1 distance between the two normalized histograms. I am comparing two piles of points, not matching point 17 to point 17. A second letter draw, seed 20263, scores L1 **0.1220** against that frozen reference. Two honest samples of the same letter already disagree by that much, because the bins are coarse. I treat 0.1220 as the floor. Every model below is scored against the same reference.

## Part 1: Gaussian to a ring

### Task 1: Straight training paths

The source is a standard 2D Gaussian. The target is a thick ring, inner radius 1.20, outer radius 2.10. I draw the angle uniformly on a full turn. I draw the squared radius uniformly between the squared edges, so the points fill the band evenly instead of piling up on the inner circle. Nothing in the target sits outside radius 2.10. A training pair is one Gaussian draw and one ring draw, matched only because they came out of the sampler in the same slot. There is no nearest-neighbor matching. `t` is uniform on (0, 1).

I wanted to see the training paths before I trusted the derivative. The first path in the figure starts at (-0.5108, 1.0283) and ends at (-1.4724, 0.3994). The velocity `u = x_1 - x_0` is (-0.9616, -0.6289). I split that segment into four equal times. The four displacements agree. The largest absolute difference among them is 1.192e-07, which is float noise. Along the segment, `dx/dt = x_1 - x_0`, and `t` has already dropped out. So the velocity on a training segment is constant, and the blue marks are equally spaced. Looking at the plot, the gray lines are chords. They cross the hole in the middle of the axes because the pairing is random. A point in the blob is tied to whatever ring point happened to share its index, not to a nearby point on the ring.

![Twelve training pairs. Orange is the Gaussian source, green is the ring, and the blue marks split each straight segment into equal times.](figures/part1_straight_paths.png)


### Task 2: Training

Training took 16.052 seconds for 12,000 steps. The first batch loss is 4.7361. The mean of the last 500 batch losses is 3.4847. On 20,000 fresh pairs, the mean of `||x_1 - x_0||^2` is 4.9096. A network that always said zero would sit near that number. The trained mean, 3.4847, is lower, and it is still far from zero. I do not think that leftover is a bug. Each pair is one random match. Two pairs can pass near the same `(x, t)` and ask for two different velocities, because their endpoints are different. The network only sees `(x, t)`, so the best it can do is an average. The squared error of that average stays positive. The pale line in the figure is one noisy batch. The blue line is a 100-batch mean, and it flattens in that range instead of diving to the axis.

![Part 1 batch loss. The pale line is one batch. The blue line is a 100-batch mean.](figures/part1_loss.png)


### Task 3: Generation without endpoints

Generation starts from 4,000 fresh Gaussian points, seed 11. I integrate with 100 Euler steps. I do not hand the network a ring point to aim at. That integration, including the stored trajectory and leaving out the plots, took 0.3146 seconds and 400,000 evaluations. The fraction of points inside the annulus is 1.0000 for a target sample, 0.3790 for the Gaussian source, and 0.9455 for the generated cloud at t = 1. On this 32 by 32 grid, with 4,000 points on each side, the histogram L1 against a target draw is 0.2960. That is a different sample size from the 8,000-point letter score later, so I do not put the two L1 numbers on the same scale.

The cloud figure is the thing I actually wanted to see. At t = 0 it is a round blob. By t = 0.50 a hole has opened in the middle. By t = 0.75 the mass is in a band, and at t = 1 the band looks like the gray target ring next to it. A few points are still inside the hole or outside the band. The annulus fraction, 0.9455 against 1 for the target, is that leftover.

![Generated cloud at t = 0, 0.25, 0.50, 0.75, and 1, next to an independent ring sample. 100 Euler steps.](figures/part1_clouds.png)


The quiver is the learned field on a 17 by 17 grid from -2.8 to 2.8. t = 0.0 has median speed 2.3539 and 80th percentile speed 2.9513. t = 0.5 has median speed 1.0534 and 80th percentile speed 1.6509. t = 1.0 has median speed 1.1671 and 80th percentile speed 1.8226. I used one arrow scale for all three panels, so a longer arrow is a faster velocity. At t = 0 the arrows point out from the origin and they are long, which matches the median speed of about 2.4. At t = 0.5 they are shorter and more even. At t = 1 they point outward again, a bit faster than at the middle and slower than at the start. The field is pushing the blob out into a ring, and then it does not shut off completely.

![Learned velocity on a grid at t = 0, 0.5, and 1. One arrow scale for all three panels.](figures/part1_quiver.png)


The learned paths use the first 2,000 of those generated particles and the same 100 segments. The mean polyline length is 1.0185. The mean start-to-end distance is 0.5214. The mean excess length is 0.4972. On the plot, most of these paths are short radial moves from inside the ring out onto the band. They are not the long crossing chords from the training figure. The excess of 0.4972 says the average path is about twice its straight-line distance, so they are not straight either. Training was shown the endpoint and asked for `x_1 - x_0`. Generation never receives that endpoint, and the field is an average over many random pairs. Straight training paths do not imply straight learned trajectories. The checkpoint is `checkpoints/part1.pt`.

![Learned Euler trajectories. Gray is a ring sample. Orange is t = 0 and red is t = 1.](figures/part1_learned_paths.png)


## Part 2: An original letter J

I wanted a target I could recognize in one glance, and I did not want to download a logo. The letter is a thick J that I built from two pieces of geometry. The stem is the box from x = 0.35 to x = 1.10 and from y = -0.25 to y = 1.85. The hook is an annular sector centered at (-0.15, -0.25), inner radius 0.50, outer radius 1.25. The angle covers the lower half of that annulus and then 0.55 radians of curl up the left side. There is no image file. I sample uniformly in a box just outside that shape and I keep the points that land inside it. A sample of 20,000 points has x in [-1.400, 1.100] and y in [-1.499, 1.850]. The farthest point in that sample is radius 2.143, and the mean radius is 1.133. This is a new set of weights. Same architecture as the ring.

Training took 29.534 seconds for 20,000 steps, seed 7. The first batch loss is 3.2345. The mean of the last 500 batches is 2.2997. The mean of `||x_1 - x_0||^2` on 20,000 fresh pairs is 3.4172. The trained loss sits under that zero-velocity level and well above zero, for the same reason as the ring: random pairs that pass near one `(x, t)` do not share one velocity. The checkpoint is `checkpoints/baseline_seed7.pt`.

The target scatter is the sampler by itself, before any network. The stem is the vertical bar on the right and the hook is the thick curve under it. The inside of the hook is empty, which matters later, because a bad sample fills that hole.

![Twenty thousand samples from the letter J sampler. No network is involved.](figures/part2_target.png)


![Part 2 batch loss for the letter model, seed 7.](figures/part2_loss.png)


### Task 2: Tracked particles

The animation follows 1,800 particles from one Gaussian draw, seed 70, for 40 Euler steps. Color is the angle of the source coordinate, through an HSV colormap. I compute the color once and I leave it on that particle. The network never sees the color, and it does not predict a color. If the colors mixed, I would be looking at a new sample at each frame. They do not mix.

At t = 0 the cloud is a Gaussian painted like a color wheel: red on the left, yellow and green on the bottom, cyan and blue on the right, purple on top. At t = 0.4 it is still that wheel, pulled in a little. The letter shows up at the end. In the t = 1 frame the hook is the warm colors and the stem is the cool colors. A red particle that started on the left finished in the hook. A blue particle that started on the right finished in the stem. The color is a label for where the particle began, stuck to it while the coordinates moved.

![Same particles, from the colored Gaussian to the letter. Color stays on the particle.](figures/part2_animation.gif)

![Same particles at t = 0. Color is the angle of the source point.](figures/part2_still_t000.png)

![Same particles at t = 0.2. Color is the angle of the source point.](figures/part2_still_t020.png)

![Same particles at t = 0.4. Color is the angle of the source point.](figures/part2_still_t040.png)

![Same particles at t = 0.6. Color is the angle of the source point.](figures/part2_still_t060.png)

![Same particles at t = 0.8. Color is the angle of the source point.](figures/part2_still_t080.png)

![Same particles at t = 1. Color is the angle of the source point.](figures/part2_still_t100.png)


### Task 3: Euler step count

I froze the weights and I froze the initial Gaussian. The same 8,000 held-out points are integrated with 1, 5, 10, 25, and 50 Euler steps. The clock is the Euler loop only. Each time is the median of 3 repeats after one untimed warmup forward. Drawing the picture is outside the timer. One evaluation is one forward pass on one particle at one Euler time, so the count is particles times steps. The largest overflow fraction in this table is 0.0000. The histogram is keeping the points that fall outside the window.

| steps | L1 | overflow | seconds | evaluations |
| --- | --- | --- | --- | --- |
| 1 | 1.9788 | 0.0000 | 0.0049 | 8000 |
| 5 | 0.5697 | 0.0000 | 0.0268 | 40000 |
| 10 | 0.3087 | 0.0000 | 0.0406 | 80000 |
| 25 | 0.2092 | 0.0000 | 0.1012 | 200000 |
| 50 | 0.1983 | 0.0000 | 0.1903 | 400000 |


These times are the Part 2 clock on the seed-7 baseline. Part 3 times that same baseline again, in the same process as reflow and continued training. The histogram distances match. The head-to-head table later uses the Part 3 clock.

The picture is clearer than the table. At 1 step the blue points are a tight clump sitting in the hook of the gray J. One Euler step with this curved field does not have time to build the letter, and the L1 is 1.9788. At 5 steps the stem and the hook are both there, with a spray of points still off the gray shape, and the L1 is 0.5697. At 10 steps the spray thins, L1 0.3087. At 25 and 50 steps I have to look at the titles to tell the panels apart: L1 0.2092 and 0.1983. Two letter draws score 0.1220 on this grid, so the 50-step cloud is still above that floor. From 1 step to 10 steps the L1 falls by 1.6700. From 25 steps to 50 steps it falls by 0.0110. The early steps are fixing the integration. The later steps are shaving a little off a score that has mostly settled. Extra steps cannot repair a velocity that points at the wrong place, and they cannot put back a thin hook if the random pairing averaged that detail away. On the shared 100-step grid, 2,000 particles have mean length 0.9234, mean chord 0.5051, and mean excess 0.4183.

![Seed 7 baseline at 1, 5, 10, 25, and 50 Euler steps. Gray is the frozen letter. Blue is the generated cloud.](figures/part2_step_clouds.png)


![Histogram L1 of the seed 7 baseline against the frozen letter sample.](figures/part2_l1_vs_steps.png)


## Part 3: One round of reflow

### Task 1: Hypothesis, then frozen pairs

I wrote this down in the log before the losses and the tables. The seed 42 log, which I started later, has the same paragraph at the top. I did not pick the hypothesis after seeing the numbers.

> Hypothesis: one round of reflow can straighten the learned transport and keep sample quality with fewer network evaluations. This is not guaranteed. Reflow learns straight paths toward the endpoints the frozen baseline reached, so it can copy the baseline's endpoint errors. A straighter path is not the same thing as a closer match to the target.

The baseline weights stay frozen while I build the pairs. Each seed draws 16,384 Gaussian sources and integrates them with 100 Euler steps. Source i is saved with the endpoint that integration actually reached. I do not shuffle. I do not swap those endpoints for fresh letter samples. If I did, I would be training on the target again and calling it reflow. The wall time for that pass is data generation. It is separate from reflow training and from the inference times below. Reflow then trains a new network, same architecture, on straight interpolations of those saved pairs. Continued training is the control for "maybe it just needed more steps." It starts from the baseline weights and the Adam state and takes the same number of extra steps on fresh letter pairs. Both extra runs use 6,000 steps. The baseline itself was trained for 20,000 steps.

| seed | baseline s | pair gen s | reflow s | continued s |
| --- | --- | --- | --- | --- |
| 7 | 29.534 | 0.704 | 9.550 | 10.338 |
| 21 | 30.236 | 0.885 | 8.358 | 9.251 |
| 42 | 28.939 | 0.801 | 9.022 | 10.091 |


Across the three seeds, pair generation totals 2.390 seconds, reflow training totals 26.931 seconds, and continued training totals 29.680 seconds. Each pair file holds 16,384 aligned pairs and costs 1,638,400 network evaluations. Seed 7's baseline time is the Part 2 training run. Seeds 21 and 42 were trained inside `part_3.py`. Reflow's mean loss over the last 500 batches is 0.0008 on seed 7, 0.0007 on seed 21, and 0.0007 on seed 42. That is tiny next to the letter loss. The continued run, which is still matching random letter pairs, stays near 2.2933, 2.2808, and 2.2979. Those two losses are squared errors on different targets. A small reflow loss means the saved pairs are easy to connect with a straight line. It does not, by itself, mean a closer letter.

![Reflow training loss, seed 7, on the saved baseline endpoints.](figures/part3_reflow_loss_seed7.png)


![Continued-training loss, seed 7, on fresh letter pairs. These 6,000 steps start from the trained baseline.](figures/part3_continued_loss_seed7.png)


### Task 2: Same held-out set, same step counts

Every row below uses the frozen Gaussian of 8,000 points. Straightness uses the first 2,000 of those points and one common grid of 100 Euler steps, dt 0.01. The number I report is the mean, over particles, of polyline length minus the straight start-to-end distance. A straight path scores near zero. Sampling seconds are medians of 3 bare Euler runs. Rendering is left out. The largest overflow fraction in the Part 3 tables is 0.0000.

| seed | model | steps | L1 | seconds | evaluations |
| --- | --- | --- | --- | --- | --- |
| 7 | baseline | 1 | 1.9788 | 0.0052 | 8000 |
| 7 | baseline | 5 | 0.5697 | 0.0241 | 40000 |
| 7 | baseline | 10 | 0.3087 | 0.0490 | 80000 |
| 7 | baseline | 25 | 0.2092 | 0.1042 | 200000 |
| 7 | baseline | 50 | 0.1983 | 0.2151 | 400000 |
| 7 | reflow | 1 | 0.2508 | 0.0041 | 8000 |
| 7 | reflow | 5 | 0.2175 | 0.0229 | 40000 |
| 7 | reflow | 10 | 0.2180 | 0.0425 | 80000 |
| 7 | reflow | 25 | 0.2167 | 0.1056 | 200000 |
| 7 | reflow | 50 | 0.2155 | 0.2272 | 400000 |
| 7 | continued | 1 | 1.9832 | 0.0050 | 8000 |
| 7 | continued | 5 | 0.6148 | 0.0244 | 40000 |
| 7 | continued | 10 | 0.3407 | 0.0446 | 80000 |
| 7 | continued | 25 | 0.2335 | 0.1048 | 200000 |
| 7 | continued | 50 | 0.2142 | 0.2437 | 400000 |
| 21 | baseline | 1 | 1.9860 | 0.0050 | 8000 |
| 21 | baseline | 5 | 0.5425 | 0.0250 | 40000 |
| 21 | baseline | 10 | 0.3027 | 0.0484 | 80000 |
| 21 | baseline | 25 | 0.2182 | 0.1220 | 200000 |
| 21 | baseline | 50 | 0.2100 | 0.2101 | 400000 |
| 21 | reflow | 1 | 0.2650 | 0.0044 | 8000 |
| 21 | reflow | 5 | 0.2195 | 0.0203 | 40000 |
| 21 | reflow | 10 | 0.2172 | 0.0415 | 80000 |
| 21 | reflow | 25 | 0.2200 | 0.1011 | 200000 |
| 21 | reflow | 50 | 0.2180 | 0.2141 | 400000 |
| 21 | continued | 1 | 1.9812 | 0.0045 | 8000 |
| 21 | continued | 5 | 0.6098 | 0.0249 | 40000 |
| 21 | continued | 10 | 0.3090 | 0.0422 | 80000 |
| 21 | continued | 25 | 0.2243 | 0.1034 | 200000 |
| 21 | continued | 50 | 0.2253 | 0.2259 | 400000 |
| 42 | baseline | 1 | 1.9823 | 0.0060 | 8000 |
| 42 | baseline | 5 | 0.6000 | 0.0253 | 40000 |
| 42 | baseline | 10 | 0.3320 | 0.0493 | 80000 |
| 42 | baseline | 25 | 0.2323 | 0.1189 | 200000 |
| 42 | baseline | 50 | 0.2137 | 0.2213 | 400000 |
| 42 | reflow | 1 | 0.2500 | 0.0041 | 8000 |
| 42 | reflow | 5 | 0.2228 | 0.0200 | 40000 |
| 42 | reflow | 10 | 0.2210 | 0.0403 | 80000 |
| 42 | reflow | 25 | 0.2215 | 0.1010 | 200000 |
| 42 | reflow | 50 | 0.2223 | 0.2003 | 400000 |
| 42 | continued | 1 | 1.9708 | 0.0046 | 8000 |
| 42 | continued | 5 | 0.5757 | 0.0227 | 40000 |
| 42 | continued | 10 | 0.3005 | 0.0447 | 80000 |
| 42 | continued | 25 | 0.2070 | 0.0986 | 200000 |
| 42 | continued | 50 | 0.2045 | 0.1962 | 400000 |


| seed | model | length | chord | excess |
| --- | --- | --- | --- | --- |
| 7 | baseline | 0.9234 | 0.5051 | 0.418286 |
| 7 | reflow | 0.5012 | 0.5010 | 0.000142 |
| 7 | continued | 0.9025 | 0.5133 | 0.389190 |
| 21 | baseline | 0.9005 | 0.5039 | 0.396655 |
| 21 | reflow | 0.5032 | 0.5030 | 0.000160 |
| 21 | continued | 0.9123 | 0.5433 | 0.368981 |
| 42 | baseline | 0.9107 | 0.5089 | 0.401753 |
| 42 | reflow | 0.5075 | 0.5073 | 0.000153 |
| 42 | continued | 0.9405 | 0.5154 | 0.425079 |


Each straightness row uses 2,000 particles and 100 segments. The same slice of particles is used for every model, so a difference is the field, not a new random start.

The L1 plot is the whole comparison in one picture. The orange reflow lines sit near 0.25 from the first Euler step and barely move. The blue baseline and green continued lines start near 2, fall hard by 10 steps, and meet the orange lines around 25 to 50 steps. At the right edge the blue line is a hair under the orange line. That is the fine-grid baseline being a little closer, and it is small next to the drop from 2 to 0.2.

![Histogram L1 on the held-out Gaussian. Color is the model. Line style is the seed.](figures/part3_l1_vs_steps.png)


![Mean excess length on the shared 100-step grid. Blue is the baseline, orange is reflow, green is continued training.](figures/part3_straightness.png)


I drew the same twelve held-out particles in each panel, with 100 Euler steps. In the baseline panel the traces bow. Several of them hook, the way the ring paths did. The reflow panel is a set of straight segments between a start dot and an end dot. The continued-training panel bows again. Extra Adam steps on the original letter loss did not straighten anything. The seed 21 and seed 42 path plots are the same three-panel comparison. I am resting the "every seed" claim on the excess column, not on staring at those two plots and calling them identical.

![Twelve held-out particles, seed 7, 100 Euler steps. The same starts are drawn in each panel.](figures/part3_paths_seed7.png)


![Same path comparison for seed 21.](figures/part3_paths_seed21.png)


![Same path comparison for seed 42.](figures/part3_paths_seed42.png)


The cloud grid is the step-count story with the other two models in the frame. Top row, baseline: 1 step is a clump in the hook, 5 steps is a rough J, 50 steps is a clean J. Middle row, reflow: 1 step is already a J, a bit speckled, and 5 through 50 steps look almost the same. Bottom row, continued training: 1 step is a clump again. More letter training did not buy the one-step letter. At 50 steps the reflow J looks a little thicker and more speckled than the baseline J. The L1 agrees with that impression, and the gap is small.

![Seed 7 clouds. Rows are baseline, reflow, and continued training. Columns are 1, 5, 10, 25, and 50 Euler steps.](figures/part3_clouds_seed7.png)


![Histogram counts at 50 Euler steps, seed 7. Each panel title includes the overflow count.](figures/part3_hists_seed7.png)


### Task 3: What the numbers support

Reflow straightens the paths on every seed. Seed 7 excess length falls from 0.418286 for the baseline to 0.000142 for reflow. Seed 21 falls from 0.396655 to 0.000160. Seed 42 falls from 0.401753 to 0.000153. On the reflow rows the polyline length and the chord agree to about three decimals, so on this 100-step grid the path is straight. Continued training stays curved. Its excess stays between 0.3690 and 0.4251.

The fewer-evaluations half shows up as soon as I look at one step. Reflow L1 at 1 Euler step is 0.2508 on seed 7, 0.2650 on seed 21, and 0.2500 on seed 42. The baseline at 1 step is 1.9788, 1.9860, and 1.9823. One reflow step also beats that seed's baseline at 5 steps and at 10 steps. By 5 reflow steps the seed 7 L1 is 0.2175, and from 10 to 50 steps it stays between 0.2155 and 0.2180. The reflow curve is already near its plateau when each particle has been evaluated once.

A straight path and a closer letter are two different claims. At 50 steps the baseline L1 is 0.1983 on seed 7, 0.2100 on seed 21, and 0.2137 on seed 42. Reflow at 50 steps is 0.2155, 0.2180, and 0.2223. On every seed the fine baseline solve is a little closer to the frozen letter than reflow is. The gaps are small next to the two-sample floor of 0.1220, and the sign is the same three times. Reflow learned the endpoints the baseline reached, so it can copy the baseline's endpoint errors. Straightening those pairs does not remove that error.

Continued training is there so I cannot credit the reflow gain to "6,000 more updates." At 1 step its L1 stays near 2: 1.9832, 1.9812, and 1.9708. At 50 steps, seed 7 moves from 0.1983 to 0.2142, seed 21 moves from 0.2100 to 0.2253, and seed 42 moves from 0.2137 to 0.2045. Extra letter training helps the 50-step score on seed 42 and hurts it on seeds 7 and 21. It never produces the 1-step reflow score, and it never straightens the 100-step paths. The low-step gain comes from training on the saved pairs.

So the straightening half holds on all three seeds. The fewer-evaluations half holds when I compare reflow with a baseline that only gets 1, 5, or 10 Euler steps. I do not get to say that reflow matches the letter better than a 25-step or 50-step baseline solve. On this run, reflow gives up a small amount of fine-grid accuracy and gets a path that is already straight at one Euler step. I can see that trade in the cloud grid and in the tables. I would not call it a pure win, and I would not call the straightening a failure.

## Other saved figures

The plots above are the ones I talk through. The same run also saved the other seeds. The cloud grids are the check that seed 7 was not a one-off picture.

![Reflow training loss, seed 21.](figures/part3_reflow_loss_seed21.png)


![Continued-training loss, seed 21.](figures/part3_continued_loss_seed21.png)


![Seed 21 clouds. Rows are baseline, reflow, and continued training.](figures/part3_clouds_seed21.png)


![Histogram counts at 50 Euler steps, seed 21.](figures/part3_hists_seed21.png)


![Reflow training loss, seed 42.](figures/part3_reflow_loss_seed42.png)


![Continued-training loss, seed 42.](figures/part3_continued_loss_seed42.png)


![Seed 42 clouds. Rows are baseline, reflow, and continued training.](figures/part3_clouds_seed42.png)


![Histogram counts at 50 Euler steps, seed 42.](figures/part3_hists_seed42.png)

