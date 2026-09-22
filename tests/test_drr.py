from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import fxr.drr as public_drr
from fxr.drr import (
    DRRRenderRequest,
    PoseSampler,
    RandomPoseGenerator,
    build_drr_camera_info,
    build_render_intrinsics,
    compute_isocenter,
    render_drr,
    sample_attenuation,
    sample_isocenters_from_centroids,
    subject_from_tensors,
)


def _random_pose_params() -> dict[str, dict[str, tuple[float, float]]]:
    return {
        "rot_range": {
            "alpha": (-10.0, 10.0),
            "beta": (0.0, 5.0),
            "gamma": (-3.0, 3.0),
        },
        "xyz_range": {
            "x": (-20.0, 20.0),
            "y": (900.0, 1100.0),
            "z": (-20.0, 20.0),
        },
    }


def test_public_drr_api_exports_planned_names() -> None:
    assert set(public_drr.__all__) == {
        "DRRIntrinsics",
        "DRRRenderRequest",
        "DRRRenderResult",
        "IsocenterConfig",
        "PoseSampler",
        "RandomPoseGenerator",
        "build_drr_camera_info",
        "build_render_intrinsics",
        "compute_isocenter",
        "render_drr",
        "sample_attenuation",
        "sample_isocenters_from_centroids",
        "subject_from_tensors",
    }
    assert not hasattr(public_drr, "DRRModel")
    assert not hasattr(public_drr, "DRRRenderer")


def test_drr_initializer_is_reexport_only() -> None:
    tree = ast.parse(Path(public_drr.__file__).read_text(encoding="utf-8"))
    disallowed = (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef)

    assert not any(isinstance(node, disallowed) for node in tree.body)


def test_pose_sampler_fixed_presets_and_explicit_pose_repeat() -> None:
    frontal = PoseSampler("frontal", camera_displacement=1000.0)
    rot, xyz = frontal.sample(3)

    assert torch.equal(rot, torch.zeros(3, 3))
    assert torch.equal(xyz, torch.tensor([[0.0, 1000.0, 0.0]]).repeat(3, 1))

    lateral_left = PoseSampler("lateral_left", camera_displacement=900.0)
    rot, xyz = lateral_left.sample()
    assert torch.equal(rot, torch.tensor([[-90.0, 0.0, 0.0]]))
    assert torch.equal(xyz, torch.tensor([[0.0, 900.0, 0.0]]))

    fixed = PoseSampler(
        "fixed",
        fixed_params={"rot": (1.0, 2.0, 3.0), "xyz": (4.0, 5.0, 6.0)},
    )
    rot, xyz = fixed.sample(2)
    assert torch.equal(rot, torch.tensor([[1.0, 2.0, 3.0]]).repeat(2, 1))
    assert torch.equal(xyz, torch.tensor([[4.0, 5.0, 6.0]]).repeat(2, 1))


def test_pose_sampler_random_validation_and_sdd_sorting() -> None:
    with pytest.raises(ValueError, match="rot_range"):
        PoseSampler(
            "random", sample_params={"xyz_range": _random_pose_params()["xyz_range"]}
        )
    with pytest.raises(ValueError, match="gamma"):
        RandomPoseGenerator(
            {
                "rot_range": {"alpha": (0, 0), "beta": (0, 0)},
                "xyz_range": _random_pose_params()["xyz_range"],
            }
        )

    params = {
        "rot_range": {axis: (0.0, 0.0) for axis in ("alpha", "beta", "gamma")},
        "xyz_range": {
            "x": (0.0, 0.0),
            "y": (1500.0, 1500.0),
            "z": (0.0, 0.0),
        },
    }
    generator = RandomPoseGenerator(params)
    sampled_sdd = torch.tensor([1000.0, 2000.0])
    _, xyz = generator.generate_poses(2, sampled_sdd=sampled_sdd)

    assert torch.all(xyz[:, 1] <= sampled_sdd)
    assert torch.equal(xyz[:, 1], torch.tensor([1000.0, 1500.0]))
    assert torch.equal(sampled_sdd, torch.tensor([1500.0, 2000.0]))


def test_pose_sampler_list_cycles_and_nested_choices_are_seeded() -> None:
    sampler = PoseSampler(["frontal", "lateral"], camera_displacement=10.0)
    rot, xyz = sampler.sample(3)

    assert torch.equal(
        rot,
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [90.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
    )
    assert torch.equal(xyz, torch.tensor([[0.0, 10.0, 0.0]]).repeat(3, 1))

    nested_a = PoseSampler(
        [["frontal", "lateral"], ["above", "offangle"]],
        seed=123,
    )
    nested_b = PoseSampler(
        [["frontal", "lateral"], ["above", "offangle"]],
        seed=123,
    )
    rot_a, xyz_a = nested_a.sample(6)
    rot_b, xyz_b = nested_b.sample(6)

    assert torch.equal(rot_a, rot_b)
    assert torch.equal(xyz_a, xyz_b)
    with pytest.raises(ValueError, match="optax"):
        PoseSampler("optax")


def test_build_render_intrinsics_normalizes_and_matches_nanodrr() -> None:
    from nanodrr.camera import make_k_inv

    intrinsics = build_render_intrinsics(
        sdd=[1000.0, 1200.0],
        delx=0.5,
        dely=[0.5, 0.6],
        x0=0.0,
        y0=[0.0, 1.0],
        height=4,
        width=6,
        orthographic=True,
    )

    assert torch.equal(intrinsics.sdd, torch.tensor([1000.0, 1200.0]))
    assert torch.equal(intrinsics.delx, torch.tensor([0.5, 0.5]))
    assert torch.equal(intrinsics.y0, torch.tensor([0.0, 1.0]))
    assert torch.allclose(
        intrinsics.k_inv[0],
        make_k_inv(1000.0, 0.5, 0.5, 0.0, 0.0, 4, 6).squeeze(0),
    )
    assert torch.allclose(
        intrinsics.k_inv[1],
        make_k_inv(1200.0, 0.5, 0.6, 0.0, 1.0, 4, 6).squeeze(0),
    )

    expanded = build_render_intrinsics(
        sdd=1000.0,
        delx=1.0,
        dely=1.0,
        height=2,
        width=3,
        num_views=3,
    )
    assert torch.equal(expanded.sdd, torch.tensor([1000.0, 1000.0, 1000.0]))


def test_build_drr_camera_info_exposes_per_view_intrinsics() -> None:
    intrinsics = build_render_intrinsics(
        sdd=[1000.0, 1200.0],
        delx=0.7,
        dely=0.8,
        height=5,
        width=7,
        orthographic=True,
    )

    info = build_drr_camera_info(intrinsics)

    assert info["sdd"] == [1000.0, 1200.0]
    assert info["delx"] == pytest.approx([0.7, 0.7])
    assert info["dely"] == pytest.approx([0.8, 0.8])
    assert info["x0"] == [0.0, 0.0]
    assert info["y0"] == [0.0, 0.0]
    assert info["height"] == 5
    assert info["width"] == 7
    assert info["orthographic"] is True


def test_subject_from_tensors_applies_mu_conversion_and_per_label_attenuation() -> None:
    from nanodrr.data.preprocess import hu_to_mu

    volume = torch.tensor(
        [
            [
                [[-1000.0, 0.0], [1000.0, 500.0]],
                [[250.0, -500.0], [0.0, 100.0]],
            ]
        ]
    )
    label = torch.tensor(
        [
            [[0, 2], [0, 0]],
            [[5, 0], [0, 0]],
        ]
    )
    subject = subject_from_tensors(
        volume,
        label,
        torch.eye(4),
        attenuation=(0.5, 0.5),
        attenuated_label_ids=[2, 5],
        do_per_label_attenuation=True,
        max_label=5,
    )

    volume_5d = volume.unsqueeze(0)
    label_5d = label.reshape(1, 1, 2, 2, 2).to(torch.float32)
    label_perm = label_5d.permute(0, 1, 4, 3, 2)
    expected_multiplier = torch.ones_like(label_perm)
    expected_multiplier[(label_perm == 2) | (label_perm == 5)] = 0.5

    assert subject.n_classes == 6
    assert torch.equal(subject.label, label_perm)
    assert torch.allclose(
        subject.image,
        hu_to_mu(volume_5d.permute(0, 1, 4, 3, 2)) * expected_multiplier,
    )


def test_sample_attenuation_distributions_and_validation() -> None:
    uniform = sample_attenuation((0.5, 1.5), 100)
    beta = sample_attenuation(
        (0.5, 1.5), 100, dist={"type": "beta", "alpha": 2, "beta": 3}
    )
    lognormal = sample_attenuation(
        (0.5, 1.5),
        20,
        dist={"type": "truncated_lognormal", "mode": 1.0, "sigma": 0.25},
    )

    for samples in (uniform, beta, lognormal):
        assert samples.shape[0] > 0
        assert torch.all(samples >= 0.5)
        assert torch.all(samples <= 1.5)

    with pytest.raises(ValueError, match="lower bound"):
        sample_attenuation((2.0, 1.0), 1)
    with pytest.raises(ValueError, match="alpha and beta"):
        sample_attenuation((0.0, 1.0), 1, dist={"type": "beta", "alpha": 1})
    with pytest.raises(ValueError, match="positive"):
        sample_attenuation(
            (0.0, 1.0), 1, dist={"type": "lognormal", "mode": 0.5, "sigma": 1.0}
        )


def test_compute_isocenter_volume_center_and_label_centroid() -> None:
    volume = torch.zeros(1, 3, 5, 7)
    label = torch.zeros(3, 5, 7, dtype=torch.long)
    label[0, 0, 0] = 1
    label[2, 4, 6] = 1
    affine = torch.eye(4)
    affine[0, 0] = 2.0
    affine[1, 1] = 3.0
    affine[2, 2] = 4.0
    affine[:3, 3] = torch.tensor([10.0, 20.0, 30.0])

    expected = torch.tensor([12.0, 26.0, 42.0])
    assert torch.allclose(compute_isocenter(volume, label, affine), expected)
    assert torch.allclose(
        compute_isocenter(volume, label, affine, "label_centroid"),
        expected,
    )


def test_sample_isocenters_from_centroids_replacement_modes() -> None:
    centroids = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    replacement = sample_isocenters_from_centroids(
        centroids,
        torch.eye(4),
        5,
        replacement=True,
        generator=torch.Generator().manual_seed(0),
    )
    assert replacement.shape == (5, 3)

    no_replacement = sample_isocenters_from_centroids(
        centroids,
        torch.eye(4),
        3,
        replacement=False,
        generator=torch.Generator().manual_seed(1),
    )
    assert sorted(no_replacement[:, 0].tolist()) == [0.0, 1.0, 2.0]

    mixed = sample_isocenters_from_centroids(
        centroids,
        torch.eye(4),
        5,
        replacement=False,
        generator=torch.Generator().manual_seed(2),
    )
    assert len(set(mixed[:3, 0].tolist())) == 3


def test_render_drr_reconstructs_background_and_forwards_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nanodrr.drr as nanodrr_drr

    calls: list[dict[str, object]] = []

    def fake_render(**kwargs: object) -> torch.Tensor:
        calls.append(kwargs)
        rt_inv = kwargs["rt_inv"]
        subject = kwargs["subject"]
        height = int(kwargs["height"])
        width = int(kwargs["width"])
        out = torch.zeros(
            rt_inv.shape[0],
            subject.n_classes,
            height,
            width,
            dtype=rt_inv.dtype,
            device=rt_inv.device,
        )
        out[:, 0].fill_(2.0)
        out[:, 1].fill_(0.6)
        out[:, 2].fill_(0.4)
        return out

    monkeypatch.setattr(nanodrr_drr, "render", fake_render)

    volume = torch.zeros(1, 2, 2, 2)
    label = torch.tensor([[[0, 1], [2, 0]], [[0, 0], [0, 0]]])
    request = DRRRenderRequest(
        rot=torch.zeros(2, 3),
        xyz=torch.tensor([[0.0, 900.0, 0.0], [0.0, 900.0, 0.0]]),
        intrinsics={
            "sdd": 1000.0,
            "delx": 1.0,
            "dely": 1.0,
            "height": 3,
            "width": 4,
        },
        n_samples=7,
        render_kwargs={"src": torch.zeros(2, 1, 3)},
    )

    result = render_drr(
        volume=volume,
        label=label,
        affine=torch.eye(4),
        request=request,
        max_label=2,
    )

    assert calls[0]["n_samples"] == 7
    assert "src" in calls[0]
    assert result.images.shape == (2, 1, 3, 4)
    assert torch.allclose(result.images, torch.full((2, 1, 3, 4), 3.0))
    assert torch.equal(result.labels[:, 0], torch.zeros(2, 3, 4))
    assert torch.equal(result.labels[:, 1], torch.ones(2, 3, 4))
    assert torch.equal(result.labels[:, 2], torch.zeros(2, 3, 4))
    assert result.camera_info["sdd"] == [1000.0, 1000.0]


def test_render_drr_soft_label_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    import nanodrr.drr as nanodrr_drr

    def fake_render(**kwargs: object) -> torch.Tensor:
        rt_inv = kwargs["rt_inv"]
        out = torch.zeros(rt_inv.shape[0], 3, 2, 2, dtype=rt_inv.dtype)
        out[:, 1].fill_(0.6)
        out[:, 2].fill_(0.4)
        return out

    monkeypatch.setattr(nanodrr_drr, "render", fake_render)

    request = DRRRenderRequest(
        rot=torch.zeros(1, 3),
        xyz=torch.tensor([[0.0, 900.0, 0.0]]),
        intrinsics={"sdd": 1000.0, "delx": 1.0, "dely": 1.0, "height": 2, "width": 2},
        render_soft_labels=True,
    )
    result = render_drr(
        volume=torch.zeros(1, 2, 2, 2),
        label=torch.zeros(2, 2, 2, dtype=torch.long),
        affine=torch.eye(4),
        request=request,
        max_label=2,
    )

    assert torch.allclose(result.labels[:, 0], torch.full((1, 2, 2), 0.4))
    assert torch.allclose(result.labels[:, 1], torch.full((1, 2, 2), 0.6))
    assert torch.allclose(result.labels[:, 2], torch.full((1, 2, 2), 0.4))


@pytest.mark.parametrize("render_soft_labels", [False, True])
def test_render_drr_foreground_collapse_map_merges_channels(
    monkeypatch: pytest.MonkeyPatch, render_soft_labels: bool
) -> None:
    import nanodrr.drr as nanodrr_drr

    def fake_render(**kwargs: object) -> torch.Tensor:
        rt_inv = kwargs["rt_inv"]
        out = torch.zeros(rt_inv.shape[0], 4, 2, 2, dtype=rt_inv.dtype)
        out[:, 0].fill_(1.0)
        out[:, 1].fill_(0.7)
        out[:, 2].fill_(0.6)
        out[:, 3].fill_(0.8)
        return out

    monkeypatch.setattr(nanodrr_drr, "render", fake_render)

    request = DRRRenderRequest(
        rot=torch.zeros(1, 3),
        xyz=torch.tensor([[0.0, 900.0, 0.0]]),
        intrinsics={"sdd": 1000.0, "delx": 1.0, "dely": 1.0, "height": 2, "width": 2},
        render_soft_labels=render_soft_labels,
        seg_threshold=0.65,
    )
    result = render_drr(
        volume=torch.zeros(1, 2, 2, 2),
        label=torch.zeros(2, 2, 2, dtype=torch.long),
        affine=torch.eye(4),
        request=request,
        max_label=3,
        foreground_collapse_map=(2, 2, 1),
        output_num_label_channels=4,
    )

    assert result.labels.shape == (1, 4, 2, 2)
    if render_soft_labels:
        assert torch.allclose(result.labels[:, 0], torch.full((1, 2, 2), 0.2))
        assert torch.allclose(result.labels[:, 1], torch.full((1, 2, 2), 0.8))
        assert torch.allclose(result.labels[:, 2], torch.full((1, 2, 2), 0.7))
    else:
        assert torch.equal(result.labels[:, 0], torch.zeros(1, 2, 2))
        assert torch.equal(result.labels[:, 1], torch.ones(1, 2, 2))
        assert torch.equal(result.labels[:, 2], torch.ones(1, 2, 2))
    assert torch.equal(result.labels[:, 3], torch.zeros(1, 2, 2))
