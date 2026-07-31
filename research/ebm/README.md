# Energy Based Models for Verifying Predictions

## Verification != Prediction

An EBM learns a function $E_\theta(x,y)$ that assigns low energy to correct pairs and high energy to incorrect pairs. Prediction becomes search:

$$
\hat{y} = \arg\min_y E_\theta(x, y)
$$

An EBM defines a probability distribution via the Boltzmann distribution:

$$
p_\theta(y \mid x) = \frac{e^{-E_\theta(x,y)}}{Z_\theta(x)}, \qquad Z_\theta(x) = \int e^{-E_\theta(x,y)}\, dy
$$

It's the maximum entropy distribution given a fixed expected energy. It's also the only form that's always non-negative and integrates to 1 for arbitrary E.

To train, we maximize log-likelihood:

$$
\nabla_\theta \log p_\theta(y|x) = -\nabla_\theta E_\theta(x, y) + \mathbb{E}_{y' \sim p_\theta}\big[\nabla_\theta E_\theta(x, y')\big]
$$

An EBM can spend arbitrary compute at inference doing optimization/search, trading test-time compute for accuracy.

## Stochastic Gradient Langevin Dynamics (SGLD)

TODO: Why?

$$
y_{t+1} = y_t - \frac{\epsilon}{2}\nabla_y E_\theta(y_t) + \sqrt{\epsilon}\,\eta_t, \quad \eta_t \sim \mathcal N(0, I)
$$

## Contrastive Divergence Training

Uses MCMC with Langevin dynamics.

$$
\mathcal{L}_{CD} = E_\theta(x^+) - E_\theta(x^-), \quad x^- \sim \text{MCMC}(p_\theta)
$$

EBMs with contrastive divergence loss have a tendency to collapse during training. We obtain zero energy for positive and negative samples.

Only use with a replay buffer + weak L2 reg

## Score Matching

MCMC-free.

Instead of matching probabilities (which needs Z), match the gradient of log-density (or score):

$$
s_\theta(x) = \nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)
$$

Z has no x-dependence, so ∇xlogZ=0. The score is independent of the partition function!

$$
\mathcal{L}_{SM} = \mathbb{E}_{x\sim p_{data}}\left[ \tfrac{1}{2}\|\nabla_x E_\theta(x)\|^2 - \nabla_x^2 E_\theta(x) \right]
$$
Needs to compute the laplacian (doesn't scale).

## Denoising Score Matching

You're teaching the energy gradient to be a denoiser.

$$
\mathcal{L}_{DSM} = \mathbb{E}_{x, \tilde{x}}\left[ \left\| \nabla_{\tilde x} E_\theta(\tilde x) + \frac{\tilde x - x}{\sigma^2} \right\|^2 \right], \quad \tilde x = x + \sigma\epsilon
$$

No collapse because the target is a fixed, nonzero vector field — a constant energy (zero gradient) gives huge loss. No MCMC, one backward pass, scales to images. For best results use multiple noise scales (this is exactly the NCSN / score-based diffusion recipe).

## Noise Contrastive Estimation (NCE)

$$
\mathcal{L}_{NCE} = -\mathbb{E}_{x\sim p_{data}}[\log D(x)] - \mathbb{E}_{x\sim p_n}[\log(1 - D(x))]
$$

Where the discriminator is forced to have EBM structure:

$$
D(x) = \sigma\big(\underbrace{-E_\theta(x) - c}_{\log p_\theta} - \log p_n(x)\big)
$$

At the optimum, c converges to the true logZ, you get a properly normalized EBM for free, no MCMC.

Works only if the noise distribution overlaps the data. In high dimensions, Gaussian noise is trivially separable → learns nothing.

## InfoNCE

Generalizes preference loss to 1 positive vs. K negatives (a softmax over candidates):

$$
\mathcal{L}_{InfoNCE} = -\log \frac{e^{-E(x^+)}}{e^{-E(x^+)} + \sum_{k=1}^K e^{-E(x_k^-)}}
$$

This is just softmax classification where the "classes" are candidate answers.
