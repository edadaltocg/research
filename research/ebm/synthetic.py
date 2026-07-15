import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from torch import Tensor, nn
from tqdm import tqdm

from research.utils.utils import get_default_device, seed_all


def _mlp(in_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, 1),
    )


class VerifierEBM(nn.Module):
    """E(y | t, moon): verify candidate y given continuous query t and DISCRETE moon.

    moon_id is a learned embedding, not a raw float, so the two moons are treated as
    distinct categories and the net cannot smoothly interpolate energy between them.
    """

    def __init__(self, hidden=128, embed_dim=16):
        super().__init__()
        self.moon_embed = nn.Embedding(2, embed_dim)
        self.net = _mlp(4 + embed_dim, hidden)  # cos(t), sin(t), moon_emb, y(2)

    def forward(self, t: Tensor, moon_id: Tensor, y: Tensor) -> Tensor:
        m = self.moon_embed(moon_id.long())
        inp = torch.cat([torch.cos(t).unsqueeze(-1), torch.sin(t).unsqueeze(-1), m, y], -1)
        return self.net(inp).squeeze(-1)


class ManifoldEBM(nn.Module):
    """E(y | moon): score-matching energy over the whole moon manifold (discrete moon)."""

    def __init__(self, hidden=128, embed_dim=16):
        super().__init__()
        self.moon_embed = nn.Embedding(2, embed_dim)
        self.net = _mlp(2 + embed_dim, hidden)  # moon_emb, y(2)

    def forward(self, moon_id: Tensor, y: Tensor) -> Tensor:
        m = self.moon_embed(moon_id.long())
        return self.net(torch.cat([m, y], -1)).squeeze(-1)


class PhaseVerifierEBM(nn.Module):
    """E(phi | x, y): verify whether phase phi explains observation y = sin(x + phi) + noise.

    Phase is circular, so a candidate phi enters as [cos(phi), sin(phi)] rather than a raw
    angle -- phi=0 and phi=2*pi must give identical energy. Because sin is periodic and
    even-symmetric, many phases explain the same (x, y): the EBM represents this multimodal
    set with several low-energy valleys, which a single-output regressor could never do.
    """

    def __init__(self, hidden=128):
        super().__init__()
        self.net = _mlp(4, hidden)  # x, y, cos(phi), sin(phi)

    def forward(self, x: Tensor, y: Tensor, phi: Tensor) -> Tensor:
        inp = torch.stack([x, y, torch.cos(phi), torch.sin(phi)], -1)
        return self.net(inp).squeeze(-1)


def sample_context(n: int):
    query = torch.rand(n) * np.pi  # parameter along the arc
    moon_id = torch.randint(0, 2, (n,))  # which moon
    return query, moon_id


def _gt_point(query: Tensor, moon_id: Tensor):
    """The 'correct answer' g(t): a point on the chosen moon."""
    # moon 0: upper arc ; moon 1: lower shifted arc
    x0 = torch.cos(query)
    y0 = torch.sin(query)
    x1 = 1 - torch.cos(query)
    y1 = 1 - torch.sin(query) - 0.5
    x = torch.where(moon_id == 0, x0, x1)
    y = torch.where(moon_id == 0, y0, y1)
    return torch.stack([x, y], -1)  # (B, 2)


def make_candidates(query: Tensor, moon_id: Tensor):
    """Build candidates of graded quality, like box jitter graded by IoU."""
    gt = _gt_point(query, moon_id)  # perfect answer
    B = len(query)

    cands = torch.stack(
        [
            gt,  # perfect (dist 0)
            gt + 0.05 * torch.randn(B, 2),  # near-perfect (hard neg)
            gt + 0.20 * torch.randn(B, 2),  # medium
            gt + 0.50 * torch.randn(B, 2),  # far
            _gt_point(torch.rand(B) * np.pi, 1 - moon_id),  # WRONG moon (wrong-object)
            (torch.rand(B, 2) * 4 - 2),  # random (background) [-2, 2]
        ],
        dim=1,
    )  # (B, 6, 2)

    # quality = -distance to the true answer  (higher = better). Candidate 0 is the perfect
    # answer; the InfoNCE loss uses that index directly, so no manual wrong-moon penalty is
    # needed -- the softmax over candidates pushes every negative (incl. wrong moon) up smoothly.
    quality = -((cands - gt.unsqueeze(1)) ** 2).sum(-1).sqrt()  # (B, 6)
    return cands, quality


def _energy(model, device, t, m, pts):
    """VerifierEBM energy at points `pts` (N, 2) for context (query t, moon m)."""
    n = len(pts)
    with torch.no_grad():
        return (
            model(torch.full((n,), float(t)).to(device), torch.full((n,), int(m)).to(device), pts.to(device))
            .cpu()
            .numpy()
        )


def _loss_figure(losses, log=False):
    """A simple training-loss curve for the Dash pages. Use log=True only for positive losses."""
    fig = go.Figure(go.Scatter(y=losses, mode="lines", line={"color": "steelblue"}))
    fig.update_layout(
        title="Training loss" + (" (log scale)" if log else ""),
        xaxis_title="step",
        yaxis_title="loss",
        yaxis_type="log" if log else "linear",
        height=280,
        margin={"l": 50, "r": 20, "t": 40, "b": 40},
    )
    return fig


def plot_energy_landscape(model: nn.Module, device, moon_noise=0.05, port=8050, losses=None):
    """Interactive verifier Dash app: 2D energy heatmap (click to probe) + 3D surface.

    Slide the query `t` and toggle the moon to move the target g(t); the energy field
    redraws live. Moon samples are projected onto the 3D surface (z = energy). Click
    any (x, y) on the heatmap to read off a candidate's energy. If `losses` is given, the
    training curve is shown at the bottom of the page.
    """
    from dash import Dash, Input, Output, dcc, html

    x_1d = np.linspace(-2.5, 2.5, 100)
    grid = torch.tensor(np.stack([g.ravel() for g in np.meshgrid(x_1d, x_1d)], 1), dtype=torch.float32)
    q = torch.linspace(0, np.pi, 150)
    moons = [_gt_point(q, torch.full_like(q, k)) + moon_noise * torch.randn(150, 2) for k in (0, 1)]

    def build(t, m, picked):
        E = _energy(model, device, t, m, grid).reshape(100, 100)
        g = _gt_point(torch.tensor([t]), torch.tensor([m]))[0]
        fig = make_subplots(
            1, 2, specs=[[{"type": "xy"}, {"type": "scene"}]], subplot_titles=("Heatmap (click to probe)", "Surface")
        )
        fig.add_trace(go.Heatmap(x=x_1d, y=x_1d, z=E, colorscale="Viridis", showscale=False), 1, 1)
        fig.add_trace(go.Surface(x=x_1d, y=x_1d, z=E, colorscale="Viridis", showscale=False, opacity=0.85), 1, 2)
        for pts, c, name in zip(moons, ("royalblue", "forestgreen"), ("Moon 0", "Moon 1"), strict=True):
            ez = _energy(model, device, t, m, pts)  # project samples onto the surface
            fig.add_trace(
                go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers", marker={"size": 4, "color": c}, name=name), 1, 1
            )
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=ez, mode="markers", marker={"size": 3, "color": c}, showlegend=False
                ),
                1,
                2,
            )
        gz = float(_energy(model, device, t, m, g[None])[0])
        fig.add_trace(
            go.Scatter(
                x=[g[0]],
                y=[g[1]],
                mode="markers",
                marker={"size": 14, "color": "crimson", "symbol": "star"},
                name="Target",
            ),
            1,
            1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=[g[0]],
                y=[g[1]],
                z=[gz],
                mode="markers",
                marker={"size": 8, "color": "crimson", "symbol": "diamond"},
                showlegend=False,
            ),
            1,
            2,
        )
        if picked:
            pe = float(_energy(model, device, t, m, torch.tensor([picked], dtype=torch.float32))[0])
            fig.add_trace(
                go.Scatter(
                    x=[picked[0]],
                    y=[picked[1]],
                    mode="markers",
                    marker={"size": 14, "color": "white", "symbol": "x"},
                    name=f"E={pe:.2f}",
                ),
                1,
                1,
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[picked[0]],
                    y=[picked[1]],
                    z=[pe],
                    mode="markers",
                    marker={"size": 6, "color": "white"},
                    showlegend=False,
                ),
                1,
                2,
            )
        fig.update_layout(title=f"EBM Verifier — t={t:.2f}, moon={m}", height=650, scene={"zaxis_title": "Energy"})
        return fig

    app = Dash(__name__)
    layout = [
        html.Label("query t"),
        dcc.Slider(id="t", min=0, max=float(np.pi), step=0.01, value=0.5, marks={0: "0", round(np.pi, 2): "π"}),
        dcc.RadioItems(
            id="m", options=[{"label": "Moon 0", "value": 0}, {"label": "Moon 1", "value": 1}], value=0, inline=True
        ),
        dcc.Graph(id="fig", figure=build(0.5, 0, None)),
    ]
    if losses:
        layout.append(dcc.Graph(id="loss", figure=_loss_figure(losses)))
    app.layout = html.Div(layout)

    @app.callback(Output("fig", "figure"), Input("t", "value"), Input("m", "value"), Input("fig", "clickData"))
    def _update(t, m, click):
        p = click["points"][0] if click else {}
        return build(t, m, (p["x"], p["y"]) if "x" in p else None)

    print(f"Serving interactive verifier at http://127.0.0.1:{port}")
    app.run(port=port)


def ebm_verifier_example():
    seed_all(42)
    device = get_default_device()
    print(f"{device=}")

    model = VerifierEBM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    steps = 10000
    # Cosine-annealed LR: high early to fit fast, decayed late to stop the loss oscillating
    # around the minimum (the noise is from SGD + fresh random negatives every step).
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)
    batch_size = 256
    n_neg = 128  # importance samples used to estimate the partition function Z
    area = 5.0 * 5.0  # the [-2.5, 2.5]^2 proposal region; q(y) = 1/area (uniform)
    log_area_over_n = float(np.log(area) - np.log(n_neg))
    e_l2 = 1e-2  # L2 on energies keeps the (otherwise unbounded) landscape from spiking
    losses = []

    for step in tqdm(range(steps)):
        query, moon_id = sample_context(batch_size)
        query, moon_id = query.to(device), moon_id.to(device)
        pos = _gt_point(query, moon_id).to(device) + 0.03 * torch.randn(batch_size, 2, device=device)

        e_pos = model(query, moon_id, pos)  # E(y+ | ctx), shape (B,)

        # Negative log-likelihood of the EBM p(y|ctx) = exp(-E(y)) / Z(ctx).
        #   NLL = E(y+) + log Z ,  Z = ∫ exp(-E) dy
        # Estimate log Z by importance sampling from the uniform proposal q(y) = 1/area:
        #   Z ≈ (1/n) Σ exp(-E(y_i)) / q(y_i) = (area/n) Σ exp(-E(y_i))
        #   log Z ≈ logsumexp(-E_neg) + log(area) - log(n)
        # The positive is folded into the normalizer too, so the model can't cheat by driving
        # E(y+) to -inf in a region the finite negatives never sample (self-normalized ML).
        neg = torch.rand(batch_size, n_neg, 2, device=device) * 5 - 2.5
        t = query.repeat_interleave(n_neg)
        m = moon_id.repeat_interleave(n_neg)
        e_neg = model(t, m, neg.reshape(batch_size * n_neg, 2)).reshape(batch_size, n_neg)
        log_z = torch.logsumexp(torch.cat([-e_pos.unsqueeze(1), -e_neg], dim=1), dim=1) + log_area_over_n
        # Regularize ALL energies (not just the positive) to keep the landscape bounded/smooth.
        e_all = torch.cat([e_pos.unsqueeze(1), e_neg], dim=1)
        loss = (e_pos + log_z).mean() + e_l2 * (e_all**2).mean()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        losses.append(loss.item())
        if step % 300 == 0:
            print(f"step {step:4d} | nll {loss.item():.3f} | lr {sched.get_last_lr()[0]:.2e}")

    plot_energy_landscape(model, device, losses=losses)


def plot_score_landscape(model: nn.Module, device, moon_noise=0.05, port=8051, losses=None):
    """Interactive viewer for the per-moon score-matching energy.

    There is no single target here: the energy is carved out along the WHOLE moon manifold,
    so the valley traces the entire arc (low energy = on the data). The radio button toggles
    the ACTIVE conditioning moon; samples from BOTH moons are always shown, so you can watch
    the valley snap under the active moon while the other moon's points sit up on the hills.
    If `losses` is given, the training curve is shown at the bottom of the page.
    """
    from dash import Dash, Input, Output, dcc, html

    x_1d = np.linspace(-2.5, 2.5, 100)
    grid = torch.tensor(np.stack([g.ravel() for g in np.meshgrid(x_1d, x_1d)], 1), dtype=torch.float32)
    q = torch.linspace(0, np.pi, 300)
    moons = [_gt_point(q, torch.full_like(q, k)) + moon_noise * torch.randn(300, 2) for k in (0, 1)]

    def energy(m, pts):
        with torch.no_grad():
            return model(torch.full((len(pts),), int(m)).to(device), pts.to(device)).cpu().numpy()

    def build(m, picked):
        E = energy(m, grid).reshape(100, 100)
        fig = make_subplots(
            1, 2, specs=[[{"type": "xy"}, {"type": "scene"}]], subplot_titles=("Heatmap (click to probe)", "Surface")
        )
        fig.add_trace(go.Heatmap(x=x_1d, y=x_1d, z=E, colorscale="Viridis", showscale=False), 1, 1)
        fig.add_trace(go.Surface(x=x_1d, y=x_1d, z=E, colorscale="Viridis", showscale=False, opacity=0.85), 1, 2)
        # Always show BOTH moons; energy is evaluated under the active conditioning moon m.
        for k, (pts, c) in enumerate(zip(moons, ("royalblue", "forestgreen"), strict=True)):
            name = f"Moon {k}" + (" (active)" if k == m else "")
            fig.add_trace(
                go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers", marker={"size": 4, "color": c}, name=name), 1, 1
            )
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=energy(m, pts),
                    mode="markers",
                    marker={"size": 3, "color": c},
                    showlegend=False,
                ),
                1,
                2,
            )
        if picked:
            pe = float(energy(m, torch.tensor([picked], dtype=torch.float32))[0])
            fig.add_trace(
                go.Scatter(
                    x=[picked[0]],
                    y=[picked[1]],
                    mode="markers",
                    marker={"size": 14, "color": "white", "symbol": "x"},
                    name=f"E={pe:.2f}",
                ),
                1,
                1,
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[picked[0]],
                    y=[picked[1]],
                    z=[pe],
                    mode="markers",
                    marker={"size": 6, "color": "white"},
                    showlegend=False,
                ),
                1,
                2,
            )
        fig.update_layout(
            title=f"Score-matching EBM — moon {m} (valley traces the whole arc)",
            height=650,
            scene={"zaxis_title": "Energy"},
        )
        return fig

    app = Dash(__name__)
    layout = [
        dcc.RadioItems(
            id="m", options=[{"label": "Moon 0", "value": 0}, {"label": "Moon 1", "value": 1}], value=0, inline=True
        ),
        dcc.Graph(id="fig", figure=build(0, None)),
    ]
    if losses:
        layout.append(dcc.Graph(id="loss", figure=_loss_figure(losses, log=True)))
    app.layout = html.Div(layout)

    @app.callback(Output("fig", "figure"), Input("m", "value"), Input("fig", "clickData"))
    def _update(m, click):
        p = click["points"][0] if click else {}
        return build(m, (p["x"], p["y"]) if "x" in p else None)

    print(f"Serving score-matching viewer at http://127.0.0.1:{port}")
    app.run(port=port)


def ebm_score_matching_example():
    """Denoising score matching (DSM): carve energy along the whole moon manifolds.

    We corrupt clean moon points x with Gaussian noise (x~ = x + sigma*eps). For a Gaussian
    kernel the true score of the noised density is (x - x~)/sigma^2, i.e. it points from the
    noisy sample back toward the clean one. We define the model score as -grad_x E(x~) and
    fit it to that target. The result is an energy with a valley (low E) exactly on the data
    manifold and rising energy off it -- unlike the ranking verifier, there is no single
    target point; the entire arc is carved out.
    """
    seed_all(42)
    device = get_default_device()
    print(f"{device=}")

    # Conditioned on the discrete moon only (learned embedding).
    model = ManifoldEBM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 512
    sigma = 0.1
    losses = []

    for step in tqdm(range(10000)):
        _, moon_id = sample_context(batch_size)
        query = torch.rand(batch_size) * np.pi
        x = _gt_point(query, moon_id).to(device)  # clean points on the moons
        moon_id = moon_id.to(device)

        noise = torch.randn_like(x)
        x_tilde = (x + sigma * noise).requires_grad_(True)  # noisy candidate
        e = model(moon_id, x_tilde)
        # model score s_theta = -grad_x E ; DSM target = (x - x_tilde)/sigma^2 = -noise/sigma
        score = -torch.autograd.grad(e.sum(), x_tilde, create_graph=True)[0]
        target = -noise / sigma
        loss = ((score - target) ** 2).sum(-1).mean()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        losses.append(loss.item())
        if step % 500 == 0:
            print(f"step {step:4d} | dsm loss {loss.item():.3f}")

    plot_score_landscape(model, device, losses=losses)


def _true_phases(x: float, y: float):
    """All phi in [0, 2*pi) with sin(x + phi) = y (the ground-truth modes to verify against)."""
    y = float(np.clip(y, -1.0, 1.0))
    a = np.arcsin(y)  # principal solution of sin(theta) = y
    thetas = [a, np.pi - a]  # sin is symmetric about pi/2
    phis = [(theta - x) % (2 * np.pi) for theta in thetas]
    return sorted({round(p, 4) for p in phis})


def plot_phase_landscape(model: nn.Module, device, losses=None, port=8052):
    """Interactive phase verifier: 1D energy E(phi | x, y) over the phase circle.

    Slide x and the observed y; the curve shows the verifier's energy for every candidate
    phase. Dashed lines mark the exact phases where sin(x + phi) = y -- the valleys should sit
    right on them, and there are usually TWO (sin's symmetry) so a regressor would fail here.
    """
    from dash import Dash, Input, Output, dcc, html

    phi = torch.linspace(0, 2 * np.pi, 400)

    def build(x, y):
        with torch.no_grad():
            e = model(torch.full_like(phi, x).to(device), torch.full_like(phi, y).to(device), phi.to(device)).cpu()
        fig = go.Figure(
            go.Scatter(x=phi.numpy(), y=e.numpy(), mode="lines", line={"color": "steelblue"}, name="E(phi)")
        )
        for p in _true_phases(x, y):
            fig.add_vline(x=p, line={"color": "crimson", "dash": "dash"})
        fig.update_layout(
            title=f"Phase verifier — x={x:.2f}, y={y:.2f} (dashed = true phases)",
            xaxis_title="phase phi",
            yaxis_title="energy",
            height=500,
        )
        return fig

    app = Dash(__name__)
    layout = [
        html.Label("x"),
        dcc.Slider(
            id="x", min=-2 * np.pi, max=2 * np.pi, step=0.05, value=0.0, marks=None, tooltip={"placement": "bottom"}
        ),
        html.Label("observed y"),
        dcc.Slider(id="y", min=-1.0, max=1.0, step=0.02, value=0.5, marks=None, tooltip={"placement": "bottom"}),
        dcc.Graph(id="fig", figure=build(0.0, 0.5)),
    ]
    if losses:
        layout.append(dcc.Graph(id="loss", figure=_loss_figure(losses)))
    app.layout = html.Div(layout)

    @app.callback(Output("fig", "figure"), Input("x", "value"), Input("y", "value"))
    def _update(x, y):
        return build(x, y)

    print(f"Serving phase verifier at http://127.0.0.1:{port}")
    app.run(port=port)


def ebm_phase_verifier_example():
    """Verify a phase for a noisy sine: is `phi` consistent with y = sin(x + phi) + noise?

    Data: sample x and a true phase, observe y = sin(x + phi_true) + noise. The verifier
    E(phi | x, y) is trained with self-normalized maximum likelihood over the phase circle so
    that every phase explaining the observation gets low energy. Because sin(theta) = y has two
    solutions per period, the learned energy has TWO valleys -- the whole point of a verifier.
    """
    seed_all(42)
    device = get_default_device()
    print(f"{device=}")

    model = PhaseVerifierEBM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    steps = 10000
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)
    batch_size = 256
    n_neg = 128  # importance samples over phi in [0, 2*pi] to estimate log Z
    noise_std = 0.05
    log_z_const = float(np.log(2 * np.pi) - np.log(n_neg))  # uniform proposal q(phi)=1/(2*pi)
    e_l2 = 1e-2
    losses = []

    for step in tqdm(range(steps)):
        x = (torch.rand(batch_size, device=device) * 4 - 2) * np.pi  # x in [-2pi, 2pi]
        phi_true = torch.rand(batch_size, device=device) * 2 * np.pi
        y = torch.sin(x + phi_true) + noise_std * torch.randn(batch_size, device=device)

        # Positive: the true phase (tiny jitter for a valley of nonzero width).
        phi_pos = phi_true + 0.02 * torch.randn(batch_size, device=device)
        e_pos = model(x, y, phi_pos)

        # Negatives: uniform phases; self-normalized NLL = E(phi+) + log Z.
        phi_neg = torch.rand(batch_size, n_neg, device=device) * 2 * np.pi
        e_neg = model(x.repeat_interleave(n_neg), y.repeat_interleave(n_neg), phi_neg.reshape(-1)).reshape(
            batch_size, n_neg
        )
        log_z = torch.logsumexp(torch.cat([-e_pos.unsqueeze(1), -e_neg], dim=1), dim=1) + log_z_const
        e_all = torch.cat([e_pos.unsqueeze(1), e_neg], dim=1)
        loss = (e_pos + log_z).mean() + e_l2 * (e_all**2).mean()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        losses.append(loss.item())
        if step % 300 == 0:
            print(f"step {step:4d} | nll {loss.item():.3f} | lr {sched.get_last_lr()[0]:.2e}")

    plot_phase_landscape(model, device, losses=losses)


if __name__ == "__main__":
    import fire

    fire.Fire({
        "verifier": ebm_verifier_example,
        "score_matching": ebm_score_matching_example,
        "phase_verifier": ebm_phase_verifier_example,
    })
