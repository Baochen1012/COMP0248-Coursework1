import os
import re
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


# gesture -> class id
GESTURE_TO_CLASS = {
    "call": 0,
    "dislike": 1,
    "like": 2,
    "ok": 3,
    "one": 4,
    "palm": 5,
    "peace": 6,
    "rock": 7,
    "stop": 8,
    "three": 9,
}
CLASS_TO_GESTURE = {v: k for k, v in GESTURE_TO_CLASS.items()}


# ---------------------------
# adapt long path
# ---------------------------
def _lp(p):
    """
    long path wrapper
    """
    p = str(Path(p).resolve())
    if os.name != "nt":
        return p
    if p.startswith("\\\\?\\"):
        return p
    if p.startswith("\\\\"):
        return "\\\\?\\UNC\\" + p.lstrip("\\")
    return "\\\\?\\" + p


def _exists(p):
    return os.path.exists(_lp(p))


def _listdir(p):
    return os.listdir(_lp(p))


def _walk(root):
    """
    os.walk but with long-path prefix & prune broken entries.
    """
    root_lp = _lp(root)

    def onerror(err):
        # don't crash, just warn
        print("[WARN]", err)

    for dirpath, dirnames, filenames in os.walk(root_lp, topdown=True, onerror=onerror):
        keep = []
        for d in dirnames:
            sub = os.path.join(dirpath, d)
            if os.path.exists(sub):
                keep.append(d)
        dirnames[:] = keep
        yield dirpath, dirnames, filenames



# small io helpers

def _is_gesture_dir(name):
    return re.match(r"^G\d{2}_[A-Za-z]+$", name) is not None


def _gesture_name(folder):
    # "G01_call" -> "call"
    if "_" not in folder:
        return folder.lower()
    return folder.split("_", 1)[1].lower()


def _read_rgb(p):
    img = Image.open(_lp(p)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _read_mask(p):
    img = Image.open(_lp(p)).convert("L")
    arr = np.asarray(img, dtype=np.uint8)
    return (arr > 127).astype(np.uint8)  # 0/1


def _read_depth(p):
    img = Image.open(_lp(p))
    arr = np.asarray(img)
    return arr.astype(np.float32)


def _norm_depth(d, mode="minmax", clip_min=None, clip_max=None, eps=1e-6):
    # normalize depth to [0,1]
    d = d.astype(np.float32)
    invalid = d <= 0
    if np.all(invalid):
        return np.zeros_like(d, dtype=np.float32)

    if clip_min is not None:
        d = np.maximum(d, clip_min)
    if clip_max is not None:
        d = np.minimum(d, clip_max)

    if mode == "fixed":
        if clip_min is None or clip_max is None:
            raise ValueError("fixed needs clip_min/clip_max")
        out = (d - clip_min) / (clip_max - clip_min + eps)
        out = np.clip(out, 0.0, 1.0)
        out[invalid] = 0.0
        return out

    v = ~invalid
    vmin = float(d[v].min())
    vmax = float(d[v].max())
    out = (d - vmin) / (vmax - vmin + eps)
    out = np.clip(out, 0.0, 1.0)
    out[invalid] = 0.0
    return out


def _to_tensor(arr):
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr.copy()
        return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    if arr.ndim == 2:
        return torch.from_numpy(arr)[None, ...].float()
    raise ValueError("bad shape: %s" % (arr.shape,))


def _mask_to_bbox(mask01):
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return (x1, y1, x2, y2)


def _project_root():
    return Path(__file__).resolve().parents[1]


# ---------------------------
# index samples
# ---------------------------
def _find_student_roots(collated_root):
    """
    Find dirs that directly contain G01_call etc.
    """
    collated_root = Path(collated_root)
    if not _exists(collated_root):
        raise FileNotFoundError("not found: %s" % collated_root)

    roots = []
    for dirpath, dirnames, _ in _walk(collated_root):
        dp = dirpath.replace("\\\\?\\UNC\\", "\\\\").replace("\\\\?\\", "")
        g = [d for d in dirnames if _is_gesture_dir(d)]
        if len(g) >= 5:
            roots.append(Path(dp))

    roots = sorted(set(roots), key=lambda x: len(str(x)), reverse=True)
    out = []
    for r in roots:
        if not any(str(r).startswith(str(k) + os.sep) for k in out):
            out.append(r)
    return sorted(out)


def index_collated_samples(collated_root, use_depth=False, depth_mode="depth", strict=True):
    student_roots = _find_student_roots(collated_root)
    samples = []
    stats = {
        "dataset_root": str(Path(collated_root)),
        "num_student_roots_found": len(student_roots),
        "num_students_used": 0,
        "num_samples": 0,
        "missing_rgb": 0,
        "missing_depth": 0,
        "total_masks_seen": 0,
        "use_depth": use_depth,
        "depth_mode": depth_mode,
    }

    used_students = set()

    for sroot in student_roots:
        student_id = sroot.name

        try:
            children = _listdir(sroot)
        except Exception:
            continue

        gesture_dirs = [d for d in children if _is_gesture_dir(d) and _exists(sroot / d)]
        for gd in sorted(gesture_dirs):
            gname = _gesture_name(gd)
            if gname not in GESTURE_TO_CLASS:
                continue
            class_id = GESTURE_TO_CLASS[gname]
            gpath = sroot / gd

            try:
                clip_dirs = [d for d in _listdir(gpath) if _exists(gpath / d)]
            except Exception:
                continue

            for cd in sorted(clip_dirs):
                clip = gpath / cd
                ann_dir = clip / "annotation"
                rgb_dir = clip / "rgb"
                if not _exists(ann_dir) or not _exists(rgb_dir):
                    continue

                try:
                    masks = [f for f in _listdir(ann_dir) if f.lower().endswith(".png")]
                except Exception:
                    continue
                if not masks:
                    continue

                for mf in sorted(masks):
                    stats["total_masks_seen"] += 1

                    mask_path = ann_dir / mf
                    rgb_path = rgb_dir / mf
                    if not _exists(rgb_path):
                        stats["missing_rgb"] += 1
                        if strict:
                            continue

                    depth_path = None
                    depth_raw_path = None
                    if use_depth:
                        if depth_mode == "depth":
                            depth_path = clip / "depth" / mf
                            if not _exists(depth_path):
                                stats["missing_depth"] += 1
                                if strict:
                                    continue
                        elif depth_mode == "depth_raw":
                            depth_raw_path = clip / "depth_raw" / mf
                            if not _exists(depth_raw_path):
                                stats["missing_depth"] += 1
                                if strict:
                                    continue
                        else:
                            raise ValueError("depth_mode must be depth/depth_raw")

                    samples.append({
                        "student_id": student_id,
                        "gesture_folder": gd,
                        "gesture_name": gname,
                        "class_id": class_id,
                        "clip_name": cd,
                        "frame_name": mf,
                        "rgb_path": str(rgb_path),
                        "mask_path": str(mask_path),
                        "depth_path": str(depth_path) if depth_path is not None else None,
                        "depth_raw_path": str(depth_raw_path) if depth_raw_path is not None else None,
                    })
                    used_students.add(student_id)

    stats["num_students_used"] = len(used_students)
    stats["num_samples"] = len(samples)
    return samples, stats


def split_by_student(samples, val_ratio=0.2, seed=42):
    students = sorted({s["student_id"] for s in samples})
    if not students:
        raise RuntimeError("no students (empty samples)")
    rng = random.Random(seed)
    rng.shuffle(students)
    n_val = max(1, int(round(len(students) * val_ratio)))
    val_students = set(students[:n_val])
    train_students = set(students[n_val:])

    train = [s for s in samples if s["student_id"] in train_students]
    val = [s for s in samples if s["student_id"] in val_students]

    info = {
        "seed": seed,
        "val_ratio": val_ratio,
        "num_students_total": len(students),
        "num_students_train": len(train_students),
        "num_students_val": len(val_students),
        "train_students": sorted(train_students),
        "val_students": sorted(val_students),
        "num_train_samples": len(train),
        "num_val_samples": len(val),
    }
    return train, val, info


def index_test_samples(test_root, use_depth=True, depth_mode="depth", strict=True):
    test_root = Path(test_root)
    if not _exists(test_root):
        raise FileNotFoundError("test_root not found: %s" % test_root)

    samples = []
    stats = {
        "test_root": str(test_root),
        "num_samples": 0,
        "missing_rgb": 0,
        "missing_depth": 0,
        "total_masks_seen": 0,
        "use_depth": use_depth,
        "depth_mode": depth_mode,
    }

    try:
        gesture_dirs = [d for d in _listdir(test_root) if _is_gesture_dir(d) and _exists(test_root / d)]
    except Exception:
        gesture_dirs = []

    for gd in sorted(gesture_dirs):
        gname = _gesture_name(gd)
        if gname not in GESTURE_TO_CLASS:
            continue
        class_id = GESTURE_TO_CLASS[gname]
        gpath = test_root / gd

        try:
            clip_dirs = [d for d in _listdir(gpath) if _exists(gpath / d)]
        except Exception:
            continue

        for cd in sorted(clip_dirs):
            clip = gpath / cd
            ann_dir = clip / "annotation"
            rgb_dir = clip / "rgb"
            if not _exists(ann_dir) or not _exists(rgb_dir):
                continue

            try:
                masks = [f for f in _listdir(ann_dir) if f.lower().endswith(".png")]
            except Exception:
                continue
            if not masks:
                continue

            for mf in sorted(masks):
                stats["total_masks_seen"] += 1

                mask_path = ann_dir / mf
                rgb_path = rgb_dir / mf
                if not _exists(rgb_path):
                    stats["missing_rgb"] += 1
                    if strict:
                        continue

                depth_path = None
                depth_raw_path = None
                if use_depth:
                    if depth_mode == "depth":
                        depth_path = clip / "depth" / mf
                        if not _exists(depth_path):
                            stats["missing_depth"] += 1
                            if strict:
                                continue
                    elif depth_mode == "depth_raw":
                        depth_raw_path = clip / "depth_raw" / mf
                        if not _exists(depth_raw_path):
                            stats["missing_depth"] += 1
                            if strict:
                                continue
                    else:
                        raise ValueError("depth_mode must be depth/depth_raw")

                samples.append({
                    "student_id": None,
                    "gesture_folder": gd,
                    "gesture_name": gname,
                    "class_id": class_id,
                    "clip_name": cd,
                    "frame_name": mf,
                    "rgb_path": str(rgb_path),
                    "mask_path": str(mask_path),
                    "depth_path": str(depth_path) if depth_path is not None else None,
                    "depth_raw_path": str(depth_raw_path) if depth_raw_path is not None else None,
                })

    stats["num_samples"] = len(samples)
    return samples, stats


class HandGestureDataset(Dataset):
    def __init__(self, samples, use_depth=False, depth_mode="depth",
                 depth_norm="minmax", depth_clip_min=None, depth_clip_max=None,
                 augment=None, return_meta=True):
        self.samples = samples
        self.use_depth = use_depth
        self.depth_mode = depth_mode
        self.depth_norm = depth_norm
        self.depth_clip_min = depth_clip_min
        self.depth_clip_max = depth_clip_max
        self.augment = augment
        self.return_meta = return_meta

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        rgb = _read_rgb(s["rgb_path"])
        mask01 = _read_mask(s["mask_path"])

        dep01 = None
        if self.use_depth:
            if self.depth_mode == "depth":
                dp = s["depth_path"]
            else:
                dp = s["depth_raw_path"]
            if dp is None:
                raise FileNotFoundError("depth missing for sample")
            dep = _read_depth(dp)
            dep01 = _norm_depth(dep, mode=self.depth_norm,
                                clip_min=self.depth_clip_min, clip_max=self.depth_clip_max)

        # if possible: augmentation hook (keep alignment!)
        if self.augment is not None:
            rgb, mask01, dep01 = self.augment(rgb, mask01, dep01)

        bbox = _mask_to_bbox(mask01)
        if bbox is None:
            bbox = (0, 0, 1, 1)

        rgb_t = _to_tensor(rgb)
        mask_t = _to_tensor(mask01)
        bbox_t = torch.tensor(bbox, dtype=torch.float32)
        label_t = torch.tensor(int(s["class_id"]), dtype=torch.long)

        if self.use_depth:
            dep_t = _to_tensor(dep01)
            img_t = torch.cat([rgb_t, dep_t], dim=0)
        else:
            img_t = rgb_t

        out = {
            "image": img_t,
            "mask": mask_t,
            "bbox": bbox_t,
            "label": label_t,
        }
        if self.return_meta:
            out["meta"] = dict(s)
        return out


def build_train_val(collated_root, val_ratio=0.2, seed=42,
                    use_depth=False, depth_mode="depth",
                    depth_norm="minmax", depth_clip_min=None, depth_clip_max=None,
                    strict=True, save_split_to=None):
    samples, stats = index_collated_samples(collated_root, use_depth=use_depth, depth_mode=depth_mode, strict=strict)
    if not samples:
        raise RuntimeError("no samples indexed (check path / naming)")

    tr, va, split = split_by_student(samples, val_ratio=val_ratio, seed=seed)
    stats["split"] = split

    if save_split_to is not None:
        p = Path(save_split_to)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"stats": stats, "split": split}, f, indent=2)

    train_ds = HandGestureDataset(tr, use_depth=use_depth, depth_mode=depth_mode,
                                  depth_norm=depth_norm, depth_clip_min=depth_clip_min, depth_clip_max=depth_clip_max)
    val_ds = HandGestureDataset(va, use_depth=use_depth, depth_mode=depth_mode,
                                depth_norm=depth_norm, depth_clip_min=depth_clip_min, depth_clip_max=depth_clip_max)
    return train_ds, val_ds, stats


def build_test(test_root, use_depth=True, depth_mode="depth",
               depth_norm="minmax", depth_clip_min=None, depth_clip_max=None,
               strict=True):
    samples, stats = index_test_samples(test_root, use_depth=use_depth, depth_mode=depth_mode, strict=strict)
    test_ds = HandGestureDataset(samples, use_depth=use_depth, depth_mode=depth_mode,
                                 depth_norm=depth_norm, depth_clip_min=depth_clip_min, depth_clip_max=depth_clip_max)
    return test_ds, stats


if __name__ == "__main__":
    pr = _project_root()
    data_dir = pr / "dataset"
    collated = data_dir / "RGB_depth_annotations"
    test_root = data_dir / "Test data" / "COMP0248_Test_data_23"

    print("project:", pr)
    print("collated:", collated, "exists=", _exists(collated))
    print("test:", test_root, "exists=", _exists(test_root))

    train_ds, val_ds, stats = build_train_val(
        collated_root=collated,
        val_ratio=0.2,
        seed=42,
        use_depth=False,   # baseline: RGB only
        depth_mode="depth",
        strict=True,
        save_split_to=pr / "results" / "splits" / "split_seed42.json",
    )
    print("train/val:", len(train_ds), len(val_ds))

    test_ds, test_stats = build_test(
        test_root=test_root,
        use_depth=True,
        depth_mode="depth",  # or "depth_raw", I think maybe just use RGH is enough.
        strict=True,
    )
    print("test:", len(test_ds))
