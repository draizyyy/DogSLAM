import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve
from scipy.spatial.transform import Rotation
from .geom import apply, pose, inv, magnitude


def occupancy(keys, cfg):
    res = cfg["resolution"]
    rays = []
    for k in keys:
        T, p, origin = k["T"], k["points"], k["origin"]
        center = apply(T, origin)
        q = apply(T, p)
        q = q[(q[:, 2] >= cfg["height_min"]) & (q[:, 2] <= cfg["height_max"])]
        if len(q):
            # Each keyframe casts at most one nearest obstacle ray per horizontal bin.
            xy = q[:, :2] - center[:2]
            ang = np.floor(
                (np.arctan2(xy[:, 1], xy[:, 0]) + np.pi) * 720 / (2 * np.pi)
            ).astype(int)
            order = np.argsort(np.linalg.norm(xy, axis=1))
            _, first = np.unique(ang[order], return_index=True)
            rays.append((center[:2], q[order[first], :2]))
    if not rays:
        raise ValueError(
            "No obstacle returns inside map height band; check axes, units, height_min/max"
        )
    allxy = np.vstack([v for a, b in rays for v in (a[None], b)])
    origin = np.floor(allxy.min(axis=0) / res) * res - 2 * res
    width, height = (np.ceil((allxy.max(axis=0) - origin) / res) + 3).astype(int)
    if width * height > 16_000_000:
        raise ValueError("Grid exceeds 16M cells. Check scale or increase resolution.")
    odds = np.zeros((height, width), np.float32)
    min_free = cfg.get("map_min_free_observations", 1)
    free_observations = np.zeros_like(odds, dtype=np.uint32) if min_free > 1 else None
    for center, q in rays:
        start = np.floor((center - origin) / res).astype(int)
        end = np.unique(np.floor((q - origin) / res).astype(int), axis=0)
        delta = end - start
        steps = np.max(np.abs(delta), axis=1)
        # Chunk rays to bound temporary arrays.
        for lo in range(0, len(end), 128):
            sl = slice(lo, lo + 128)
            n = steps[sl]
            u = np.arange(n.max() + 1)[None, :]
            valid = u < n[:, None]
            cell = start + np.rint(
                delta[sl, None, :] * u[:, :, None] / np.maximum(n[:, None, None], 1)
            ).astype(int)
            if lo == 0:
                free = []
            free.append(cell[valid, 1] * width + cell[valid, 0])
        free = np.unique(np.concatenate(free))
        hit = end[:, 1] * width + end[:, 0]
        free = np.setdiff1d(free, hit, assume_unique=True)
        odds.flat[free] -= 0.46
        if free_observations is not None:
            free_observations.flat[free] += 1
        odds.flat[hit] += 0.92
        np.clip(odds, -4, 4, out=odds)
    pixels = np.full(odds.shape, 205, np.uint8)
    is_free = odds < -0.15
    if free_observations is not None:
        is_free &= free_observations >= min_free
    pixels[is_free] = 254
    pixels[odds > 0.75] = 0
    return pixels, origin, res


def render(folder, grid, origin, res, traj):
    im = Image.fromarray(grid[::-1])
    im.save(folder / "map.pgm")
    overlay(im, origin, res, traj, folder / "map_route.png")


def overlay(im, origin, res, traj, path):
    im = im.convert("RGB")
    im = im.resize((im.width * 2, im.height * 2), Image.Resampling.NEAREST)
    xy = (np.array(traj)[:, :2, 3] - origin) / res
    pts = [(2 * x, im.height - 1 - 2 * y) for x, y in xy]
    draw = ImageDraw.Draw(im)
    if len(pts) > 1:
        draw.line(pts, fill=(22, 111, 188), width=3)
    for (x, y), col in [(pts[0], "green"), (pts[-1], "red")]:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=col)
    im.save(path)


def seeds(scan, level, grid, origin, res, cfg):
    # Coarse global search across observed free cells for each orientation.
    stride = max(1, round(0.25 / res))
    h, w = grid.shape
    hc, wc = (h + stride - 1) // stride, (w + stride - 1) // stride
    padded = np.pad(
        grid, ((0, hc * stride - h), (0, wc * stride - w)), constant_values=205
    )
    blocks = padded.reshape(hc, stride, wc, stride)
    occupied = (blocks == 0).any(axis=(1, 3))
    free = (blocks == 254).any(axis=(1, 3)) & ~occupied
    step = stride * res
    field = np.exp(-0.5 * (distance_transform_edt(~occupied) * step / 0.32) ** 2)
    q = apply(level, scan)
    xy = q[(q[:, 2] >= cfg["height_min"]) & (q[:, 2] <= cfg["height_max"]), :2]
    if len(xy) < 12 or not free.any():
        raise ValueError(
            "Insufficient features/free cells for global start; set --initial"
        )
    radius = int(np.ceil(np.max(np.linalg.norm(xy, axis=1)) / step)) + 1
    candidates = []
    for yaw in np.arange(-180, 180, 10):
        r = Rotation.from_euler("z", yaw, degrees=True)
        offsets = np.unique(
            np.rint(xy @ r.as_matrix()[:2, :2].T / step).astype(int), axis=0
        )
        kernel = np.zeros((radius * 2 + 1,) * 2, np.float32)
        kernel[offsets[:, 1] + radius, offsets[:, 0] + radius] = 1
        scores = fftconvolve(field, kernel[::-1, ::-1], mode="same") / len(offsets)
        scores[~free] = -1
        for _ in range(3):
            y, x = np.unravel_index(np.argmax(scores), scores.shape)
            if scores[y, x] < 0:
                break
            candidates.append(
                (
                    scores[y, x],
                    pose(r, [*(origin + (np.array([x, y]) + 0.5) * step), 0]) @ level,
                )
            )
            gap = max(1, round(0.8 / step))
            scores[max(0, y - gap) : y + gap + 1, max(0, x - gap) : x + gap + 1] = -1
    selected = []
    for score, T in sorted(candidates, key=lambda x: -x[0]):
        if all(
            any(v > lim for v, lim in zip(magnitude(inv(U) @ T), (0.65, 16)))
            for U in selected
        ):
            selected.append(T)
        if len(selected) >= 10:
            break
    return selected
