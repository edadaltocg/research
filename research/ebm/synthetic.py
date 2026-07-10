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


def _loss_figure(losses):
    """A simple training-loss curve for the Dash pages."""
    fig = go.Figure(go.Scatter(y=losses, mode="lines", line={"color": "steelblue"}))
    fig.update_layout(
        title="Training loss (log scale)",
        xaxis_title="step",
        yaxis_title="loss",
        yaxis_type="log",
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
    n_neg = 128  # negatives spread across the whole plane per context
    temperature = 2.0  # >1 softens the InfoNCE softmax -> wider, less spiky energy valley
    e_l2 = 1e-1  # L2 on energies: kills shift-invariance AND penalizes very deep/sharp spikes
    losses = []

    for step in tqdm(range(steps)):
        query, moon_id = sample_context(batch_size)
        query, moon_id = query.to(device), moon_id.to(device)
        pos = _gt_point(query, moon_id).to(device) + 0.03 * torch.randn(batch_size, 2, device=device)
        # Negatives sampled uniformly over the plane so the softmax denominator covers all of
        # space, not just 6 points near the target. Without this the landscape is unconstrained
        # off the candidates and looks weird; with it, InfoNCE carves a smooth bowl at the target.
        neg = torch.rand(batch_size, n_neg, 2, device=device) * 5 - 2.5
        cands = torch.cat([pos.unsqueeze(1), neg], dim=1)  # (B, 1+n_neg, 2); index 0 = positive
        K = cands.shape[1]

        t = query.repeat_interleave(K)
        m = moon_id.repeat_interleave(K)
        e = model(t, m, cands.reshape(batch_size * K, 2)).reshape(batch_size, K)

        # InfoNCE: -E/temperature are logits, the positive (index 0) must get the most mass.
        # A higher temperature and stronger energy L2 keep the valley smooth instead of a spike.
        logits = -e / temperature
        target = torch.zeros(batch_size, dtype=torch.long, device=device)
        loss = nn.functional.cross_entropy(logits, target) + e_l2 * (e**2).mean()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        losses.append(loss.item())
        if step % 300 == 0:
            print(f"step {step:4d} | loss {loss.item():.3f} | lr {sched.get_last_lr()[0]:.2e}")

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
        layout.append(dcc.Graph(id="loss", figure=_loss_figure(losses)))
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


if __name__ == "__main__":
    import fire

    fire.Fire({"verifier": ebm_verifier_example, "score_matching": ebm_score_matching_example})
