#!/usr/bin/env python3
import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scene.colmap_loader import read_extrinsics_binary, read_extrinsics_text, read_intrinsics_binary, read_intrinsics_text, qvec2rotmat


SH_C0 = 0.28209479177387814


def rgb_to_sh(rgb):
    return (rgb.astype(np.float32) - 0.5) / SH_C0


def inverse_sigmoid(x):
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x)).astype(np.float32)


def read_obj(obj_path):
    vertices, uvs, faces = [], [], []
    with open(obj_path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("vt "):
                parts = line.split()
                uvs.append([float(parts[1]), float(parts[2])])
            elif line.startswith("f "):
                verts = []
                for token in line.split()[1:]:
                    vals = token.split("/")
                    vi = int(vals[0]) - 1
                    ti = int(vals[1]) - 1 if len(vals) > 1 and vals[1] else -1
                    verts.append((vi, ti))
                for i in range(1, len(verts) - 1):
                    faces.append([verts[0], verts[i], verts[i + 1]])
    return np.asarray(vertices, np.float32), np.asarray(uvs, np.float32), faces


def texture_for_obj(obj_path):
    folder = Path(obj_path).parent
    mtl = folder / "model.mtl"
    if mtl.exists():
        for line in mtl.read_text(errors="ignore").splitlines():
            if line.lower().startswith("map_kd"):
                tex = folder / line.split(maxsplit=1)[1].strip()
                if tex.exists():
                    return Image.open(tex).convert("RGB")
    fallback = folder / "texture_kd.jpg"
    return Image.open(fallback).convert("RGB") if fallback.exists() else None


def sample_obj(obj_path, count, rng):
    vertices, uvs, faces = read_obj(obj_path)
    tri_idx = np.array([[v[0] for v in face] for face in faces], dtype=np.int64)
    tri_uv_idx = np.array([[v[1] for v in face] for face in faces], dtype=np.int64)
    tris = vertices[tri_idx]
    areas = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    probs = areas / areas.sum()
    choices = rng.choice(len(tris), size=count, p=probs)
    r1 = np.sqrt(rng.random(count, dtype=np.float32))
    r2 = rng.random(count, dtype=np.float32)
    bary = np.stack([1.0 - r1, r1 * (1.0 - r2), r1 * r2], axis=1).astype(np.float32)
    pts = (tris[choices] * bary[:, :, None]).sum(axis=1)

    tex = texture_for_obj(obj_path)
    colors = np.full((count, 3), 0.65, dtype=np.float32)
    if tex is not None and len(uvs):
        tuv_idx = tri_uv_idx[choices]
        valid = (tuv_idx >= 0).all(axis=1)
        if valid.any():
            suv = (uvs[tuv_idx[valid]] * bary[valid, :, None]).sum(axis=1)
            w, h = tex.size
            px = np.clip((suv[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(np.int32)
            py = np.clip((1.0 - (suv[:, 1] % 1.0)) * (h - 1), 0, h - 1).astype(np.int32)
            colors[valid] = np.asarray(tex)[py, px].astype(np.float32) / 255.0
    return pts, colors


def sample_obj_with_normals(obj_path, count, rng):
    vertices, uvs, faces = read_obj(obj_path)
    tri_idx = np.array([[v[0] for v in face] for face in faces], dtype=np.int64)
    tri_uv_idx = np.array([[v[1] for v in face] for face in faces], dtype=np.int64)
    tris = vertices[tri_idx]
    raw_normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    areas = 0.5 * np.linalg.norm(raw_normals, axis=1)
    face_normals = raw_normals / np.maximum(np.linalg.norm(raw_normals, axis=1, keepdims=True), 1e-8)
    probs = areas / areas.sum()
    choices = rng.choice(len(tris), size=count, p=probs)
    r1 = np.sqrt(rng.random(count, dtype=np.float32))
    r2 = rng.random(count, dtype=np.float32)
    bary = np.stack([1.0 - r1, r1 * (1.0 - r2), r1 * r2], axis=1).astype(np.float32)
    pts = (tris[choices] * bary[:, :, None]).sum(axis=1)
    normals = face_normals[choices].astype(np.float32)

    tex = texture_for_obj(obj_path)
    colors = np.full((count, 3), 0.65, dtype=np.float32)
    if tex is not None and len(uvs):
        tuv_idx = tri_uv_idx[choices]
        valid = (tuv_idx >= 0).all(axis=1)
        if valid.any():
            suv = (uvs[tuv_idx[valid]] * bary[valid, :, None]).sum(axis=1)
            w, h = tex.size
            px = np.clip((suv[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(np.int32)
            py = np.clip((1.0 - (suv[:, 1] % 1.0)) * (h - 1), 0, h - 1).astype(np.int32)
            colors[valid] = np.asarray(tex)[py, px].astype(np.float32) / 255.0
    return pts, colors, normals


def shade_colors(colors, normals, mode):
    if mode == "none":
        return colors
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    if mode == "normal":
        light = np.asarray([-0.25, -0.60, 0.76], np.float32)
        light = light / np.linalg.norm(light)
        diffuse = np.maximum(normals @ light, 0.0)[:, None]
        fill = np.maximum(normals @ -light, 0.0)[:, None]
        shade = 0.42 + 0.55 * diffuse + 0.12 * fill
    elif mode == "fake_texture":
        key = np.asarray([-0.35, -0.70, 0.62], np.float32)
        fill_dir = np.asarray([0.70, 0.20, 0.68], np.float32)
        up = np.asarray([0.0, 1.0, 0.0], np.float32)
        key = key / np.linalg.norm(key)
        fill_dir = fill_dir / np.linalg.norm(fill_dir)
        key_lobe = np.maximum(np.abs(normals @ key), 0.0)[:, None]
        fill_lobe = np.maximum(np.abs(normals @ fill_dir), 0.0)[:, None]
        vertical = (0.5 + 0.5 * np.clip(normals @ up, -1.0, 1.0))[:, None]
        shade = 0.36 + 0.42 * key_lobe + 0.18 * fill_lobe + 0.18 * vertical
    else:
        raise ValueError(f"Unknown shading mode: {mode}")
    return np.clip(colors * shade, 0.08, 0.92).astype(np.float32)


def center_scale_translate(points, target_center, target_size):
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = (lo + hi) * 0.5
    extent = float(np.max(hi - lo))
    scale = target_size / extent
    transformed = (points - center) * scale + np.asarray(target_center, np.float32)
    return transformed, scale


def scale_around_bounds_center(points, target_size):
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = (lo + hi) * 0.5
    extent = float(np.max(hi - lo))
    scale = target_size / extent
    return (points - center) * scale, scale


def place_on_plane(points, anchor, normal, plane_d, clearance=0.02):
    min_normal_coord = float((points @ normal).min())
    translation = anchor + normal * (clearance - min_normal_coord)
    placed = points + translation
    min_clearance = float((placed @ normal + plane_d).min())
    print(f"Placed object minimum plane clearance: {min_clearance:.4f}")
    return placed


def table_z_at_xy(x, y, normal, plane_d):
    if abs(float(normal[2])) < 1e-6:
        raise RuntimeError("Estimated table plane is too close to vertical to solve z(x, y)")
    return float(-(normal[0] * x + normal[1] * y + plane_d) / normal[2])


def place_upright_on_plane(points, anchor, normal, plane_d, clearance=0.10):
    min_plane_coord = float((points @ normal).min())
    translation = anchor + normal * (clearance - min_plane_coord)
    placed = points + translation
    min_clearance = float((placed @ normal + plane_d).min())
    print(f"Placed upright object minimum table-plane clearance={min_clearance:.4f}")
    return placed


def make_gaussian_rows(template_dtype, points, colors, splat_scale):
    rows = np.zeros(len(points), dtype=template_dtype)
    rows["x"], rows["y"], rows["z"] = points[:, 0], points[:, 1], points[:, 2]
    for name in rows.dtype.names:
        if name.startswith("f_rest_"):
            rows[name] = 0.0
    sh = rgb_to_sh(colors)
    rows["f_dc_0"], rows["f_dc_1"], rows["f_dc_2"] = sh[:, 0], sh[:, 1], sh[:, 2]
    rows["opacity"] = inverse_sigmoid(0.92)
    log_scale = np.float32(math.log(splat_scale))
    rows["scale_0"], rows["scale_1"], rows["scale_2"] = log_scale, log_scale, log_scale
    rows["rot_0"], rows["rot_1"], rows["rot_2"], rows["rot_3"] = 1.0, 0.0, 0.0, 0.0
    return rows


def apply_color_override(colors, override):
    if override is None:
        return colors
    tint = np.asarray(override, np.float32)
    luminance = (0.2126 * colors[:, 0] + 0.7152 * colors[:, 1] + 0.0722 * colors[:, 2])[:, None]
    return np.clip(0.35 * luminance + 0.65 * tint[None, :], 0.0, 1.0).astype(np.float32)


def transformed_ply_rows(ply_path, target_center, target_size):
    ply = PlyData.read(ply_path)
    rows = np.array(ply["vertex"].data, copy=True)
    xyz = np.vstack([rows["x"], rows["y"], rows["z"]]).T.astype(np.float32)
    xyz, scale = center_scale_translate(xyz, target_center, target_size)
    rows["x"], rows["y"], rows["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    for key in ("scale_0", "scale_1", "scale_2"):
        if key in rows.dtype.names:
            rows[key] = rows[key] + np.float32(math.log(scale))
    return rows


def transformed_ply_rows_on_plane(ply_path, anchor, target_size, normal, plane_d):
    ply = PlyData.read(ply_path)
    rows = np.array(ply["vertex"].data, copy=True)
    xyz = np.vstack([rows["x"], rows["y"], rows["z"]]).T.astype(np.float32)
    xyz, scale = scale_around_bounds_center(xyz, target_size)
    xyz = place_upright_on_plane(xyz, anchor, normal, plane_d)
    rows["x"], rows["y"], rows["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    for key in ("scale_0", "scale_1", "scale_2"):
        if key in rows.dtype.names:
            rows[key] = rows[key] + np.float32(math.log(scale))
    return rows


def load_colmap_camera(source, train_frame_idx, resolution_width):
    sparse = source / "sparse" / "0"
    try:
        extrinsics = read_extrinsics_binary(str(sparse / "images.bin"))
        intrinsics = read_intrinsics_binary(str(sparse / "cameras.bin"))
    except Exception:
        extrinsics = read_extrinsics_text(str(sparse / "images.txt"))
        intrinsics = read_intrinsics_text(str(sparse / "cameras.txt"))

    all_names = sorted(extr.name for extr in extrinsics.values())
    test_names = {name for idx, name in enumerate(all_names) if idx % 8 == 0}
    train_names = [name for name in all_names if name not in test_names]
    image_name = train_names[train_frame_idx]
    extr = {entry.name: entry for entry in extrinsics.values()}[image_name]
    intr = intrinsics[extr.camera_id]

    R = np.transpose(qvec2rotmat(extr.qvec))
    T = np.asarray(extr.tvec, np.float32)
    world_to_camera = np.eye(4, dtype=np.float32)
    world_to_camera[:3, :3] = R.T
    world_to_camera[:3, 3] = T
    camera_to_world = np.linalg.inv(world_to_camera)

    scale = intr.width / float(resolution_width)
    if intr.model == "SIMPLE_PINHOLE":
        fx = fy = intr.params[0] / scale
        cx = intr.params[1] / scale
        cy = intr.params[2] / scale
    elif intr.model == "PINHOLE":
        fx = intr.params[0] / scale
        fy = intr.params[1] / scale
        cx = intr.params[2] / scale
        cy = intr.params[3] / scale
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {intr.model}")

    return {
        "name": image_name,
        "world_to_camera": world_to_camera,
        "camera_to_world": camera_to_world,
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
    }


def project_points(points, camera):
    w2c = camera["world_to_camera"]
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    x = camera["fx"] * cam[:, 0] / z + camera["cx"]
    y = camera["fy"] * cam[:, 1] / z + camera["cy"]
    return x, y, z


def estimate_table_plane(base_rows, source, train_frame_idx, resolution_width, rng):
    camera = load_colmap_camera(source, train_frame_idx, resolution_width)
    xyz = np.vstack([base_rows["x"], base_rows["y"], base_rows["z"]]).T.astype(np.float32)
    sample_count = min(1_200_000, len(xyz))
    sample = xyz[rng.choice(len(xyz), size=sample_count, replace=False)]
    x, y, z = project_points(sample, camera)

    table_ellipse = ((x - 505.0) / 335.0) ** 2 + ((y - 380.0) / 150.0) ** 2 < 1.0
    center_obstacles = ((x - 515.0) / 120.0) ** 2 + ((y - 310.0) / 85.0) ** 2 < 1.0
    mask = (z > 0.0) & table_ellipse & (~center_obstacles) & (y > 230.0) & (y < 510.0)
    candidates = sample[mask]
    candidate_depth = z[mask]
    if len(candidates) < 1000:
        raise RuntimeError(f"Too few table candidates for plane fitting: {len(candidates)}")

    low, high = np.quantile(candidate_depth, [0.02, 0.45])
    candidates = candidates[(candidate_depth >= low) & (candidate_depth <= high)]
    if len(candidates) < 1000:
        raise RuntimeError(f"Too few depth-filtered table candidates: {len(candidates)}")

    best = None
    for _ in range(1500):
        ids = rng.choice(len(candidates), 3, replace=False)
        a, b, c = candidates[ids]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm
        plane_d = -float(normal @ a)
        distances = np.abs(candidates @ normal + plane_d)
        inliers = distances < 0.025
        score = int(inliers.sum())
        if best is None or score > best[0]:
            best = (score, inliers)

    if best is None:
        raise RuntimeError("Unable to fit table plane")

    inlier_points = candidates[best[1]]
    center = inlier_points.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_points - center, full_matrices=False)
    normal = vh[-1].astype(np.float32)
    if normal[2] < 0:
        normal = -normal
    normal = normal / np.linalg.norm(normal)
    plane_d = -float(normal @ center)
    print(f"Estimated table plane from {len(inlier_points)} inliers in {camera['name']}: n={normal.tolist()}, d={plane_d:.6f}")
    return normal, plane_d, camera


def pixel_to_plane(pixel, camera, normal, plane_d):
    u, v = pixel
    ray_camera = np.asarray([(u - camera["cx"]) / camera["fx"], (v - camera["cy"]) / camera["fy"], 1.0], np.float32)
    origin = camera["camera_to_world"][:3, 3]
    direction = camera["camera_to_world"][:3, :3] @ ray_camera
    denom = float(normal @ direction)
    if abs(denom) < 1e-6:
        raise RuntimeError(f"Pixel ray is nearly parallel to table plane: {pixel}")
    t = -float(normal @ origin + plane_d) / denom
    if t <= 0:
        raise RuntimeError(f"Pixel ray intersects table behind the camera: {pixel}")
    return origin + t * direction


def prepare_output_model(src_model, dst_model, iteration):
    dst_model.mkdir(parents=True, exist_ok=True)
    for name in ["cfg_args", "cameras.json", "exposure.json", "input.ply"]:
        src = src_model / name
        if src.exists():
            shutil.copy2(src, dst_model / name)
    pc_dir = dst_model / "point_cloud" / f"iteration_{iteration}"
    pc_dir.mkdir(parents=True, exist_ok=True)
    return pc_dir / "point_cloud.ply"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="output/mipnerf360/garden")
    parser.add_argument("--objects", default="3D_data")
    parser.add_argument("--output", default="output/mipnerf360/garden_on_table_gray_v12")
    parser.add_argument("--source", default="data/mipnerf360/garden")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--resolution", type=int, default=960)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--objb-shading", choices=["none", "fake_texture", "normal"], default="none")
    args = parser.parse_args()

    root = Path.cwd()
    src_model = root / args.model
    dst_model = root / args.output
    objects = root / args.objects
    rng = np.random.default_rng(20260622)

    base_ply = src_model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    print(f"Loading base garden model: {base_ply}")
    base = PlyData.read(base_ply)
    base_rows = np.array(base["vertex"].data, copy=True)
    dtype = base_rows.dtype
    table_normal, table_d, table_camera = estimate_table_plane(base_rows, root / args.source, 80, args.resolution, rng)
    table_normal = -table_normal
    table_d = -table_d
    print(f"Using flipped placement normal: n={table_normal.tolist()}, d={table_d:.6f}")

    object_rows = []
    placements = [
        ("objB/model.obj", 220000, (360.0, 365.0), 0.44, 0.0022, None),
        ("objC/model.obj", 120000, (505.0, 395.0), 0.46, 0.0060, None),
    ]
    for rel_path, samples, pixel, size, splat_scale, color_override in placements:
        obj_path = objects / rel_path
        print(f"Sampling {samples} points from {obj_path}")
        if rel_path == "objB/model.obj" and args.objb_shading != "none":
            pts, colors, normals = sample_obj_with_normals(obj_path, samples, rng)
            colors = shade_colors(colors, normals, args.objb_shading)
        else:
            pts, colors = sample_obj(obj_path, samples, rng)
        colors = apply_color_override(colors, color_override)
        pts, _ = scale_around_bounds_center(pts, size)
        anchor = pixel_to_plane(pixel, table_camera, table_normal, table_d)
        print(f"Placing {rel_path} at table pixel {pixel}, anchor {anchor.tolist()}")
        pts = place_upright_on_plane(pts, anchor, table_normal, table_d)
        object_rows.append(make_gaussian_rows(dtype, pts, colors, splat_scale))

    third_ply = objects / "point_cloud.ply"
    if third_ply.exists():
        print(f"Transforming Gaussian object: {third_ply}")
        anchor = pixel_to_plane((645.0, 365.0), table_camera, table_normal, table_d)
        print(f"Placing point_cloud.ply at table pixel {(645.0, 365.0)}, anchor {anchor.tolist()}")
        object_rows.append(transformed_ply_rows_on_plane(third_ply, anchor, 0.42, table_normal, table_d))

    out_ply = prepare_output_model(src_model, dst_model, args.iteration)
    combined = np.concatenate([base_rows] + object_rows)
    print(f"Writing augmented model with {len(combined):,} Gaussians: {out_ply}")
    PlyData([PlyElement.describe(combined, "vertex")]).write(out_ply)

    if args.skip_render:
        return

    render_cmd = [
        ".venv/bin/python", "render.py",
        "-s", args.source,
        "-i", "images_4",
        "-m", args.output,
        "--iteration", str(args.iteration),
        "--skip_test",
        "--quiet",
        "-r", str(args.resolution),
    ]
    print("Rendering inserted scene frames...")
    subprocess.run(render_cmd, check=True)

    render_dir = dst_model / "train" / f"ours_{args.iteration}" / "renders"
    video_path = dst_model / "garden_with_objects_walkthrough.mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", str(render_dir / "%05d.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-c:v", "libx264",
        "-crf", "18",
        str(video_path),
    ]
    print("Encoding video...")
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Video written to {video_path}")


if __name__ == "__main__":
    main()
