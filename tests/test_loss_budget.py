import json
import math

import pytest
import torch

from research.loss_budget.train import (
    GPT,
    GPTConfig,
    LossBudgetController,
    ScaledLinear,
    TokenBatches,
    atomic_save,
    ensemble_log_probs,
    evaluate,
    lr_multiplier,
    main,
    make_optimizers,
    project_embeddings_,
    snapshot_steps,
    tangent_noise_,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_model():
    torch.manual_seed(123)
    model = GPT(
        GPTConfig(
            sequence_len=8, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2, n_embd=32
        )
    )
    model.init_weights()
    return model


def test_routing_normalization_and_zero_initialization():
    model = small_model()
    assert not any(isinstance(module, torch.nn.Dropout) for module in model.modules())
    muon, adam = make_optimizers(model)
    muon_ids = {id(p) for group in muon.param_groups for p in group["params"]}
    adam_ids = {id(p) for group in adam.param_groups for p in group["params"]}
    assert not muon_ids & adam_ids
    assert muon_ids | adam_ids == {id(p) for p in model.parameters()}
    for module in model.modules():
        if isinstance(module, ScaledLinear):
            assert (id(module.weight) in adam_ids) == module.row_normalized
            assert id(module.row_scale) in adam_ids
            assert id(module.col_scale) in adam_ids
    for block in model.transformer.h:
        assert torch.count_nonzero(block.attn.c_proj(torch.randn(2, 32))) == 0
        assert torch.count_nonzero(block.attn.attn_gate(torch.randn(2, 12))) == 0
    embeddings = model.transformer.wte.weight.detach().clone()
    scales = model.transformer.h[0].attn.c_q.row_scale.detach().clone()
    matrix = model.transformer.h[0].attn.c_q.weight.detach().clone()
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    muon.step(0.01, torch.Generator().manual_seed(0))
    assert not torch.equal(matrix, model.transformer.h[0].attn.c_q.weight)
    assert torch.equal(embeddings, model.transformer.wte.weight)
    assert torch.equal(scales, model.transformer.h[0].attn.c_q.row_scale)
    x = torch.randint(0, 64, (2, 8))
    model.zero_grad(set_to_none=True)
    model(x, x).backward()
    adam.step()
    project_embeddings_(model)
    muon.step(0.01, torch.Generator().manual_seed(1))
    for module in model.modules():
        if isinstance(module, ScaledLinear):
            if module.row_normalized:
                torch.testing.assert_close(
                    module.weight.norm(dim=1), torch.ones(module.weight.shape[0])
                )
            else:
                assert module.weight.norm().item() == pytest.approx(
                    math.sqrt(module.weight.shape[0]), rel=1e-6
                )
    assert model.transformer.h[0].attn.c_proj.row_scale.abs().max() > 0


def test_noise_tangent_and_expected_size():
    p = torch.zeros(100, 100)
    p[0, 0] = 10
    original = p.clone()
    tangent_noise_(p, 0.1, torch.Generator().manual_seed(17))
    delta = p - original
    assert (delta * original).sum() == 0
    assert delta.norm().item() == pytest.approx(1, rel=0.03)


def test_controller_feedback_cold_start_resume_and_invalid_loss():
    c = LossBudgetController(
        target=3, beta=0.5, seed_std=0.01, max_std=0.02, growth=2, warmup=2, interval=1
    )
    assert c.observe(2) == 0
    assert c.ema == 2  # bias correction, no artificial initial low loss
    assert c.observe(2) == 0
    assert c.observe(2) == 0.01
    assert c.observe(2) == 0.02
    assert c.observe(2) == 0.02
    restored = LossBudgetController(**c.state_dict())
    assert restored.observe(10) == c.observe(10) == 0.01
    assert restored.state_dict() == c.state_dict()
    with pytest.raises(FloatingPointError):
        c.observe(float("nan"))
    disabled = LossBudgetController(enabled=False, warmup=0, interval=1)
    assert disabled.observe(1) == 0
    with pytest.raises(ValueError):
        LossBudgetController(beta=1)


def test_schedule_lr_and_distinct_snapshot_steps():
    steps = snapshot_steps(101)
    assert steps[0] == 51 and steps[-1] == 101
    assert len(set(steps)) == 10
    assert (
        max(b - a for a, b in zip(steps, steps[1:]))
        - min(b - a for a, b in zip(steps, steps[1:]))
        <= 1
    )
    with pytest.raises(ValueError):
        snapshot_steps(16)
    assert snapshot_steps(20, 1) == [20]
    assert lr_multiplier(0, 10, 2, 0.2) == 0.5
    assert lr_multiplier(9, 10, 2, 0.2) == pytest.approx(0.2)


def test_data_formats_epoch_determinism_and_finite_eval(tmp_path):
    rows = torch.arange(54).reshape(6, 9)
    token_path, packed_path = tmp_path / "tokens.pt", tmp_path / "packed.pt"
    torch.save({"tokens": rows.flatten()}, token_path)
    torch.save(
        {
            "chunks": [
                rows[:4].flatten(),
                torch.cat([rows[4:], torch.zeros(2, 9, dtype=torch.long)]).flatten(),
            ],
            "valid_counts": [4, 2],
            "batch_size": 4,
            "sequence_size": 9,
        },
        packed_path,
    )
    a = TokenBatches(token_path, 2, 8, shuffle=True)
    b = TokenBatches(packed_path, 2, 8, shuffle=True)
    for i in [0, 1, 2, 6, 4, 0]:
        for x, y in zip(a.batch(i, "cpu"), b.batch(i, "cpu")):
            assert torch.equal(x, y)
    with pytest.raises(ValueError):
        evaluate(small_model(), a, a.num_batches + 1)


def test_probability_mixture_not_logit_or_loss_average(tmp_path):
    class FixedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(2))

        def forward(self, x):
            return self.logits.expand(*x.shape, 2)

    model = FixedModel()
    first, second = tmp_path / "a.pt", tmp_path / "b.pt"
    atomic_save({"model": {"logits": torch.tensor([8.0, 0.0])}}, first)
    atomic_save({"model": {"logits": torch.tensor([0.0, 0.0])}}, second)
    result = ensemble_log_probs(
        model, [first, second], torch.zeros(1, 1, dtype=torch.long)
    )
    expected = (
        torch.tensor([8.0, 0.0]).softmax(-1) + torch.tensor([0.0, 0.0]).softmax(-1)
    ) / 2
    torch.testing.assert_close(result.exp()[0, 0], expected)
    assert not torch.allclose(result.exp()[0, 0], torch.tensor([4.0, 0.0]).softmax(-1))
    assert result.exp().sum().item() == pytest.approx(1.0)


def test_training_resume_matches_uninterrupted_including_rng_and_snapshots(tmp_path):
    train_path, val_path = tmp_path / "train.pt", tmp_path / "val.pt"
    torch.save({"tokens": torch.arange(90) % 64}, train_path)
    torch.save({"tokens": torch.arange(36).flip(0) % 64}, val_path)
    common = [
        "--train-data",
        str(train_path),
        "--val-data",
        str(val_path),
        "--steps",
        "6",
        "--sequence-len",
        "8",
        "--n-layer",
        "2",
        "--n-head",
        "2",
        "--n-embd",
        "32",
        "--total-batch-size",
        "16",
        "--snapshots",
        "2",
        "--eval-batches",
        "1",
        "--lr-warmup-steps",
        "0",
        "--noise-warmup-steps",
        "0",
        "--noise-interval",
        "1",
        "--loss-target",
        "20",
        "--device",
        "cpu",
    ]
    whole, split = tmp_path / "whole", tmp_path / "split"
    main(common + ["--output", str(whole)])
    main(common + ["--output", str(split), "--stop-after", "3"])
    main(common + ["--output", str(split), "--resume", str(split / "latest.pt")])
    a = torch.load(whole / "latest.pt", weights_only=True)
    b = torch.load(split / "latest.pt", weights_only=True)
    assert a["step"] == b["step"] == 6
    assert a["micro_seen"] == b["micro_seen"] == 12
    assert a["controller"] == b["controller"]
    for name in a["model"]:
        assert torch.equal(a["model"][name], b["model"][name]), name
    assert torch.equal(a["noise_rng"], b["noise_rng"])
    assert torch.equal(a["torch_rng"], b["torch_rng"])
    assert (
        json.loads((whole / "result.json").read_text())["ensemble"]
        == json.loads((split / "result.json").read_text())["ensemble"]
    )
    assert json.loads((split / "snapshots.json").read_text())["complete"]
    main(common + ["--output", str(whole), "--eval-only"])
    assert json.loads((whole / "ensemble-eval.json").read_text()) == json.loads(
        (whole / "result.json").read_text()
    )["ensemble"]
    with pytest.raises(ValueError, match="differ"):
        main(
            common
            + [
                "--output",
                str(split),
                "--loss-target",
                "19",
                "--resume",
                str(split / "latest.pt"),
            ]
        )


def test_accumulation_observes_every_microbatch_and_stops_at_epoch_end(tmp_path):
    train_path, val_path = tmp_path / "train.pt", tmp_path / "val.pt"
    torch.save({"tokens": torch.arange(45)}, train_path)  # five microbatches
    torch.save({"tokens": torch.arange(27).flip(0)}, val_path)
    torch.manual_seed(42)
    model = GPT(GPTConfig(sequence_len=8, n_layer=2, n_head=2, n_kv_head=2, n_embd=32))
    model.init_weights()
    model.eval()
    batches = TokenBatches(train_path, 1, 8, shuffle=True)
    with torch.no_grad():
        losses = [model(*batches.batch(i, "cpu")).item() for i in range(2)]
    output = tmp_path / "run"
    main(
        [
            "--output",
            str(output),
            "--train-data",
            str(train_path),
            "--val-data",
            str(val_path),
            "--num-epochs",
            "1",
            "--sequence-len",
            "8",
            "--n-layer",
            "2",
            "--n-head",
            "2",
            "--n-embd",
            "32",
            "--total-batch-size",
            "16",
            "--snapshots",
            "0",
            "--eval-batches",
            "1",
            "--device",
            "cpu",
        ]
    )
    records = [
        json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    assert records[0]["train_loss"] == pytest.approx(sum(losses) / 2, abs=1e-6)
    assert len(records) == 3
    assert records[-1]["tokens"] == 8
    assert records[-1]["epoch"] == 1
    assert torch.load(output / "latest.pt", weights_only=True)["micro_seen"] == 5


def test_parameterization_preserves_baseline_initial_weights_and_outputs(monkeypatch):
    from research.loss_budget import train as model_module
    from research.loss_budget.train import parameterize_model_

    monkeypatch.setattr(model_module, "parameterize_model_", lambda model: None)
    model = small_model()
    model.eval()
    original = {
        name: module.weight.detach().clone()
        for name, module in model.named_modules()
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding))
    }
    tokens = torch.randint(0, 64, (1, 8))
    expected = model(tokens).detach()
    parameterize_model_(model)
    for name, weight in original.items():
        module = model.get_submodule(name)
        actual = module.row_scale[:, None] * module.weight * module.col_scale
        torch.testing.assert_close(actual, weight, atol=1e-7, rtol=2e-6)
    torch.testing.assert_close(model(tokens), expected, atol=1e-6, rtol=1e-5)
