
## Multi-Objective Optimization Implementation

As discussed in the [MSeg paper](), we apply a state-of-the-art multi-task learning algorithm, MGDA, [1] to MSeg. Performance on various datasets (representing diverse domains) can be viewed as different tasks in a multi-task learning framework. Although these different tasks may conflict (which would require a trade-off), a common compromise is to optimize a proxy objective that minimizes a weighted linear combination of per-task losses.


The main idea of the Multiple Gradient Descent Algorithm (MGDA is instead of heuristically setting such weights, at each iteration solve a small subproblem to find the pareto optimal weight setting.  In each iteration, a loss function and loss function gradient is evaluated independently for each dataset. A gradient descent direction is obtained as a convex combination of these various loss gradients.

We make a few changes to the original implementation:
1. Since we need as many backward passes as we have tasks, we simply put each task in its own process in the Pytorch [DDP](https://pytorch.org/docs/master/notes/ddp.html) framework.
2. In order to prevent synchronization of DDP processes, we use the `ddp.no_sync()` context, before the `loss.backward()` call.

The implementation is found in the following files:
-	dist_mgda_utils.py: Handles the gathering of gradients across processes and forms convex combination of per-task gradients.
-	min_norm_solvers.py: Computes pareto optimal weights per iteration using Frank-Wolfe optimization.

[1] Ozan  Sener  and  Vladlen  Koltun. [Multi-task  learning  as multi-objective optimization.](https://arxiv.org/abs/1810.04650) In NeurIPS. 2018

