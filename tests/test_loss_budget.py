"""Load definitions without changing the submission's top-level script layout."""

import math
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--no-compile"])
    module = ModuleType("loss_budget_test")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    source = (
        Path("research/loss_budget/train.py").read_text().split("# Compute init")[0]
    )
    exec(compile(source, "research/loss_budget/train.py", "exec"), module.__dict__)
    return module


def test_paper_initialization_routing_and_projection(trainer):
    t = trainer
    model = t.GPT(
        t.GPTConfig(
            sequence_len=8, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2, n_embd=32
        )
    )
    model.init_weights()
    opt = model.setup_optimizer()
    groups = {id(p): g for g in opt.param_groups for p in g["params"]}
    assert (
        sum(len(g["params"]) for g in opt.param_groups)
        == len(list(model.parameters()))
        == len(groups)
    )
    assert not any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    before = {name: p.clone() for name, p in model.named_parameters()}
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
            assert groups[id(m.weight)]["kind"] == (
                "adamw" if m.row_normalized else "muon"
            )
            torch.testing.assert_close(
                torch.nn.functional.softplus(m.row_scale), torch.ones_like(m.row_scale)
            )
            torch.testing.assert_close(
                torch.nn.functional.softplus(m.col_scale), torch.ones_like(m.col_scale)
            )
            for p in (m.row_scale, m.col_scale):
                assert groups[id(p)]["kind"] == "adamw"
                assert groups[id(p)]["lr"] == groups[id(m.weight)]["lr"]
            norm = m.weight.norm(dim=1) if m.row_normalized else m.weight.norm()
            torch.testing.assert_close(norm, torch.ones_like(norm) * m.sphere_radius)
    assert all(g["weight_decay"] == 0 for g in opt.param_groups)
    assert all(
        g["betas"] == (0.9, 0.99) and g["eps"] == 1e-8
        for g in opt.param_groups
        if g["kind"] == "adamw"
    )
    for name, p in model.named_parameters():
        if not name.endswith(("row_scale", "col_scale")):
            p.grad = torch.zeros_like(p)
    t.prepare_md_step_(model)
    t.project_weights_(model, 0.1, torch.Generator().manual_seed(1))
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
            norm = m.weight.norm(dim=1) if m.row_normalized else m.weight.norm()
            torch.testing.assert_close(norm, torch.ones_like(norm) * m.sphere_radius)
            if m.row_normalized:
                torch.testing.assert_close(m.weight, before[name + ".weight"])
            else:
                assert not torch.equal(m.weight, before[name + ".weight"])
            assert torch.equal(m.row_scale, before[name + ".row_scale"])
            assert torch.equal(m.col_scale, before[name + ".col_scale"])


def test_fused_gradient_clipping_and_gain_chain_rule(trainer):
    t = trainer
    model = torch.nn.Sequential(torch.nn.Linear(5, 3, bias=False))
    t.parameterize_model_(model)
    m = model[0]
    with torch.no_grad():
        m.row_scale.add_(0.4)
        m.col_scale.sub_(0.2)
    row, col = torch.nn.functional.softplus(m.row_scale), torch.nn.functional.softplus(
        m.col_scale
    )
    direction = m.weight.detach().clone()
    with torch.no_grad():
        m.weight.mul_(row[:, None]).mul_(col)
    m.weight.grad = torch.arange(1.0, 16.0).view(3, 5)
    grad = m.weight.grad.clone()
    expected_norm = grad.norm()
    grad *= 1 / (expected_norm + 1e-6)
    # Independent autograd reference to Algorithm 2, after fused-gradient clipping.
    d = direction.clone().requires_grad_()
    r = m.row_scale.detach().clone().requires_grad_()
    c = m.col_scale.detach().clone().requires_grad_()
    (
        torch.nn.functional.softplus(r)[:, None]
        * d
        * torch.nn.functional.softplus(c)
        * grad
    ).sum().backward()
    actual_norm = t.prepare_md_step_(model)
    torch.testing.assert_close(actual_norm, expected_norm)
    torch.testing.assert_close(m.weight, direction)
    torch.testing.assert_close(m.weight.grad, d.grad)
    torch.testing.assert_close(m.row_scale.grad, r.grad)
    torch.testing.assert_close(m.col_scale.grad, c.grad)
    t.project_weights_(model, 0, torch.Generator())
    torch.testing.assert_close(m.weight, row[:, None] * direction * col)


def test_feedback_and_checkpoints(trainer):
    t = trainer
    assert t.update_noise(0, 2, 100) == 0
    assert t.update_noise(0, 2, 110) == 1e-4
    assert t.update_noise(0.01, 2, 110) == pytest.approx(0.0102)
    assert t.update_noise(0.01, 4, 110) == pytest.approx(0.01 / 1.02)
    assert t.update_noise(0.01, 3.505, 110) == 0.01
    steps = t.snapshot_steps(101, 10)
    assert steps[0] == 51 and steps[-1] == 101 and len(set(steps)) == 10
    assert (
        max(b - a for a, b in zip(steps, steps[1:]))
        - min(b - a for a, b in zip(steps, steps[1:]))
        <= 1
    )


def test_probability_ensemble(trainer, tmp_path):
    from contextlib import nullcontext

    t = trainer
    t.autocast_ctx = nullcontext()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(2))

        def get_device(self):
            return torch.device("cpu")

        def forward(self, x, y, loss_reduction):
            return torch.nn.functional.cross_entropy(
                self.logits.expand(y.numel(), 2), y.flatten(), reduction=loss_reduction
            )

    model = Model()
    paths = [tmp_path / "a.pt", tmp_path / "b.pt"]
    for path, logits in zip(paths, [torch.tensor([8.0, 0.0]), torch.zeros(2)]):
        torch.save({"model": {"logits": logits}}, path)
    loader = lambda: iter(
        [(torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 1, dtype=torch.long), 1)]
    )
    loss = t.evaluate_ensemble(model, paths, loader, 1)
    expected = -torch.log((torch.tensor([8.0, 0.0]).softmax(-1)[0] + 0.5) / 2)
    assert loss == pytest.approx(expected.item())


def test_paper_lr_defaults_and_linear_schedule(trainer):
    import ast

    t = trainer
    source = ast.parse(Path("research/loss_budget/train.py").read_text())
    scheduler = next(
        n
        for n in source.body
        if isinstance(n, ast.FunctionDef) and n.name == "get_lr_multiplier"
    )
    exec(
        compile(ast.Module(body=[scheduler], type_ignores=[]), "schedule", "exec"),
        t.__dict__,
    )
    t.num_iterations = 101
    assert t.MATRIX_LR == pytest.approx(0.008)
    assert t.EMBEDDING_LR == pytest.approx(0.003)
    assert t.UNEMBEDDING_LR == pytest.approx(0.001)
    assert t.WARMUP_RATIO == 0 and t.WARMDOWN_RATIO == 1
    assert t.MIN_LR == 1e-8
    assert [t.get_lr_multiplier(i) for i in (0, 50, 100)] == [1, 0.5, 0]
