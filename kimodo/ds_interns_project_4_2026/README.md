

# KIMODO — Deep Dive Technical Analysis

<!-- > **Project 4** | *Data Science Internship* -->



---

## About KIMODO

**Kimodo** is a **Ki**nematic **Mo**tion **D**iffusi**o**n model developed by NVIDIA's Spatial Intelligence Lab. It generates high-quality 3D human and robot motions, controlled through intuitive text prompts and a comprehensive suite of kinematic constraints.

- 🔗 [Project Page](https://research.nvidia.com/labs/sil/projects/kimodo)
- 💻 [GitHub Code](https://github.com/nv-tlabs/kimodo)
- 🤗 [Models on HuggingFace](https://huggingface.co/collections/nvidia/kimodo-v1)
- 📄 [Technical Report (PDF)](https://research.nvidia.com/labs/sil/projects/kimodo/assets/kimodo_tech_report.pdf)

---

## <span style='color:#5C5CFF'> System Architecture </span>

<!-- ![Kimodo Pipeline Architecture](https://research.nvidia.com/labs/sil/projects/kimodo/assets/arch.png) -->
<div style="border: 2px solid #76b900; border-radius: 8px; background: #1a1a1a; overflow: hidden; width: 100%;">
  <div style="background: #76b900; padding: 3px 10px; font-size: 11px; font-weight: 500; color: #000; font-family: monospace; letter-spacing: 0.04em;">KIMODO PIPELINE ARCHITECTURE</div>
  <div style="padding: 12px;">
    <img src="https://research.nvidia.com/labs/sil/projects/kimodo/assets/arch.png" alt="Kimodo Pipeline Architecture" style="width: 100%; border-radius: 4px; display: block;" />
  </div>
</div>

Kimodo is built on a **two-stage transformer denoiser** that decomposes root and body motion prediction:

1. **Root Denoiser** — predicts global root/pelvis motion from text + constraints
2. **Body Denoiser** — fills in full-body joint motion in local, root-relative space

This decomposition minimizes common artifacts like floating and foot skating.

---

## Key Capabilities

### Text-to-Motion Generation

Kimodo supports intuitive text control for diverse behaviors — locomotion, dancing, stunts, gestures, and more. It also handles compositional prompts like *"A person walks forward then waves their arms."*

### Constraint Types
<!-- 
| Constraint | Description |
|---|---|
| **Root2D** | Controls horizontal trajectory and global heading (2D waypoints / dense paths) |
| **FullBody** | Full high-dimensional keyframe pose across all joints |
| **EndEffector** | Sparse joint targets (hands, feet) with position and rotation | -->
<div style="border: 2px solid #5e7f25; border-radius: 8px; background: #1a1a1a; overflow: hidden; width: 100%;">
  <div style="background: #5e7f25; padding: 3px 10px; font-size: 12px; font-weight: 700; color: #000; font-family: monospace; letter-spacing: 0.04em;">CONSTRAINT TYPES</div>
  <table style="width: 100%; border-collapse: collapse; font-family: monospace; font-size: 14px; color: #ffffff;">
    <thead>
      <tr style="border-bottom: 2px solid #76b900;">
        <th style="padding: 10px 14px; text-align: left; border-right: 1px solid #333; color: #76b900; white-space: nowrap;">Constraint</th>
        <th style="padding: 10px 14px; text-align: left; color: #76b900;">Description</th>
      </tr>
    </thead>
    <tbody>
      <tr style="border-bottom: 1px solid #333;">
        <td style="padding: 10px 14px; border-right: 1px solid #333; white-space: nowrap; color: #76b900;">Root2D</td>
        <td style="padding: 10px 14px;">Controls horizontal trajectory and global heading (2D waypoints / dense paths)</td>
      </tr>
      <tr style="border-bottom: 1px solid #333; background: #222;">
        <td style="padding: 10px 14px; border-right: 1px solid #333; white-space: nowrap; color: #76b900;">FullBody</td>
        <td style="padding: 10px 14px;">Full high-dimensional keyframe pose across all joints</td>
      </tr>
      <tr>
        <td style="padding: 10px 14px; border-right: 1px solid #333; white-space: nowrap; color: #76b900;">EndEffector</td>
        <td style="padding: 10px 14px;">Sparse joint targets (hands, feet) with position and rotation</td>
      </tr>
    </tbody>
  </table>
</div>
---

## Q&A: Technical Design Decisions

---

<!-- ### 1. Why Use Type-Based Polymorphism Instead of a Unified Constraint Representation? -->
### <span style="color:#76b900">1. Why Use Type-Based Polymorphism Instead of a Unified Constraint Representation?</span>

Kimodo uses heterogeneous constraint types (`Root2D`, `FullBody`, `EndEffector`) with **polymorphic deserialization** rather than a single unified class.

**Problem with a unified representation:**
A "God Class" containing every possible field for every constraint type would be wasteful. For example:
- `Root2D` only needs horizontal trajectory and heading
- `FullBody` requires high-dimensional tensors for every joint
- `EndEffector` needs sparse data for specific joints plus joint name metadata

Most fields would be `null` for most instances — creating fragile, ambiguous schemas.

**Why polymorphism is essential:**

- **Safety-Critical Masking:** Each constraint type "owns" the logic to write into its specific slots in the binary control mask `m`. This prevents the wrong masking code from ever running on mismatched data.
- **Type-Specific Validation:** `Root2D` validates 2D inputs, `FullBody` checks joint count against the skeleton, `EndEffector` verifies joint names exist in the hierarchy. This validation is enforced at construction time — an invalid object cannot be instantiated.
- **Extensibility:** Adding a new constraint type requires only a new subclass. The existing deserializer, trainer, and mask-generator remain untouched (Open/Closed Principle).

- **In a unified representation**, adding a new constraint requires modifying the core data structure and every function that handles it (the "Switch-Statement" problem).

- **In a polymorphic system**, a developer can simply define a new subclass. The existing pipeline (deserializer, trainer, and mask-generator) remains untouched because it interacts with the constraints through a common interface.



### Constraint Classes

Here is the code for the constraint sets:
# [constraints.py]
```python

class Root2DConstraintSet:

    name = "root2d"

    def __init__(
        self,
        skeleton: SkeletonBase,
        frame_indices: Tensor,
        smooth_root_2d: Tensor,
        to_crop: bool = False,
        global_root_heading: Optional[Tensor] = None,
    ) -> None:
        self.skeleton = skeleton

        # if we pass the full smooth root 3D as input
        if smooth_root_2d.shape[-1] == 3:
            smooth_root_2d = smooth_root_2d[..., [0, 1]]

        if to_crop:
            smooth_root_2d = smooth_root_2d[frame_indices]
            if global_root_heading is not None:
                global_root_heading = global_root_heading[frame_indices]
        else:
            assert len(smooth_root_2d) == len(
                frame_indices
            ), "The number of smooth root 2d should be match the number of frames"
            if global_root_heading is not None:
                assert len(global_root_heading) == len(
                    frame_indices
                ), "The number of global root heading should be match the number of frames"

        self.smooth_root_2d = smooth_root_2d
        self.global_root_heading = global_root_heading
        self.frame_indices = frame_indices

    def update_constraints(self, data_dict: dict, index_dict: dict) -> None:
        """Append this constraint's smooth_root_2d (and optional global_root_heading) to data/index
        dicts."""
        data_dict["smooth_root_2d"].append(self.smooth_root_2d)
        index_dict["smooth_root_2d"].append(self.frame_indices)

        if self.global_root_heading is not None:
            # constraint the global heading
            data_dict["global_root_heading"].append(self.global_root_heading)
            index_dict["global_root_heading"].append(self.frame_indices)

    def crop_move(self, start: int, end: int) -> "Root2DConstraintSet":
        """Return a new constraint set for the cropped frame range [start, end)."""
        mask = (self.frame_indices >= start) & (self.frame_indices < end)

        if self.global_root_heading is not None:
            masked_global_root_heading = self.global_root_heading[mask]
        else:
            masked_global_root_heading = None

        return Root2DConstraintSet(
            self.skeleton,
            self.frame_indices[mask] - start,
            self.smooth_root_2d[mask],
            global_root_heading=masked_global_root_heading,
        )

    def get_save_info(self) -> dict:
        """Return a dict suitable for JSON serialization (frame_indices, smooth_root_2d, optional
        global_root_heading)."""
        out = {
            "type": self.name,
            "frame_indices": self.frame_indices,
            "smooth_root_2d": self.smooth_root_2d,
        }
        if self.global_root_heading is not None:
            out["global_root_heading"] = self.global_root_heading
        return out

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Root2DConstraintSet":
        self.smooth_root_2d = _tensor_to(self.smooth_root_2d, device, dtype)
        self.frame_indices = _tensor_to(self.frame_indices, device, dtype)
        if self.global_root_heading is not None:
            self.global_root_heading = _tensor_to(self.global_root_heading, device, dtype)
        if device is not None and hasattr(self.skeleton, "to"):
            self.skeleton = self.skeleton.to(device)
        return self

    @classmethod
    def from_dict(cls, skeleton: SkeletonBase, dico: dict) -> "Root2DConstraintSet":
        """Build a Root2DConstraintSet from a dict (e.g. loaded from JSON)."""
        device = skeleton.device if hasattr(skeleton, "device") else "cpu"

        if "global_root_heading" in dico:
            global_root_heading = torch.tensor(dico["global_root_heading"], device=device)
        else:
            global_root_heading = None

        return cls(
            skeleton,
            frame_indices=torch.tensor(dico["frame_indices"]),
            smooth_root_2d=torch.tensor(dico["smooth_root_2d"], device=device),
            global_root_heading=global_root_heading,
        )

```

```python

class FullBodyConstraintSet:
    """Constraint set fixing full-body global positions and rotations on given keyframes."""

    name = "fullbody"

    def __init__(
        self,
        skeleton: SkeletonBase,
        frame_indices: Tensor,
        global_joints_positions: Tensor,
        global_joints_rots: Tensor,
        smooth_root_2d: Optional[Tensor] = None,
        to_crop: bool = False,
    ):
        self.skeleton = skeleton
        self.frame_indices = frame_indices

        # if we pass the full smooth root 3D as input
        if smooth_root_2d is not None and smooth_root_2d.shape[-1] == 3:
            smooth_root_2d = smooth_root_2d[..., [0, 1]]

        if to_crop:
            global_joints_positions = global_joints_positions[frame_indices]
            global_joints_rots = global_joints_rots[frame_indices]
            if smooth_root_2d is not None:
                smooth_root_2d = smooth_root_2d[frame_indices]
        else:
            assert len(global_joints_positions) == len(
                frame_indices
            ), "The number of global positions should be match the number of frames"
            assert len(global_joints_rots) == len(
                frame_indices
            ), "The number of global joint rotations should be match the number of frames"

            if smooth_root_2d is not None:
                assert len(smooth_root_2d) == len(
                    frame_indices
                ), "The number of smooth root 2d (if specified) should be match the number of frames"

        if smooth_root_2d is None:
            # substitute the smooth root 2d with the real root
            smooth_root_2d = global_joints_positions[:, skeleton.root_idx, [0, 2]]

        # root y: from smooth or pelvis is the same
        self.root_y_pos = global_joints_positions[:, skeleton.root_idx, 1]

        self.global_joints_positions = global_joints_positions
        self.global_joints_rots = global_joints_rots
        self.global_root_heading = compute_global_heading(global_joints_positions, skeleton)
        self.smooth_root_2d = smooth_root_2d

    def update_constraints(self, data_dict: dict, index_dict: dict) -> None:
        """Append global positions, smooth root 2D, root y, and global heading to data/index
        dicts."""
        nbjoints = self.skeleton.nbjoints
        indices_lst = create_pairs(
            self.frame_indices,
            torch.arange(nbjoints, device=self.frame_indices.device),
        )
        data_dict["global_joints_positions"].append(
            self.global_joints_positions.reshape(-1, 3)
        )  # flatten the global positions
        index_dict["global_joints_positions"].append(indices_lst)

        # global rotations are not used here

        # as we use smooth root, also constraint the smooth root to get the same full body
        # maybe keep storing the hips offset, if we smooth it ourselves
        data_dict["smooth_root_2d"].append(self.smooth_root_2d)
        index_dict["smooth_root_2d"].append(self.frame_indices)

        # constraint the y pos of the root
        data_dict["root_y_pos"].append(self.root_y_pos)
        index_dict["root_y_pos"].append(self.frame_indices)

        # constraint the global heading
        data_dict["global_root_heading"].append(self.global_root_heading)
        index_dict["global_root_heading"].append(self.frame_indices)

    def crop_move(self, start: int, end: int) -> "FullBodyConstraintSet":
        """Return a new FullBodyConstraintSet for the cropped frame range [start, end)."""
        mask = (self.frame_indices >= start) & (self.frame_indices < end)
        return FullBodyConstraintSet(
            self.skeleton,
            self.frame_indices[mask] - start,
            self.global_joints_positions[mask],
            self.global_joints_rots[mask],
            self.smooth_root_2d[mask],
        )

    def get_save_info(self) -> dict:
        """Return a dict for JSON save: type, frame_indices, local_joints_rot, root_positions, smooth_root_2d."""
        local_joints_rot = self.skeleton.global_rots_to_local_rots(self.global_joints_rots)
        if isinstance(self.skeleton, SOMASkeleton30):
            local_joints_rot = self.skeleton.to_SOMASkeleton77(local_joints_rot)
        local_joints_rot = matrix_to_axis_angle(local_joints_rot)

        root_positions = self.global_joints_positions[:, self.skeleton.root_idx]
        return {
            "type": self.name,
            "frame_indices": self.frame_indices,
            "local_joints_rot": local_joints_rot,
            "root_positions": root_positions,
            "smooth_root_2d": self.smooth_root_2d,
        }

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "FullBodyConstraintSet":
        self.frame_indices = _tensor_to(self.frame_indices, device, dtype)
        self.global_joints_positions = _tensor_to(self.global_joints_positions, device, dtype)
        self.global_joints_rots = _tensor_to(self.global_joints_rots, device, dtype)
        self.root_y_pos = _tensor_to(self.root_y_pos, device, dtype)
        self.global_root_heading = _tensor_to(self.global_root_heading, device, dtype)
        self.smooth_root_2d = _tensor_to(self.smooth_root_2d, device, dtype)
        if device is not None and hasattr(self.skeleton, "to"):
            self.skeleton = self.skeleton.to(device)
        return self

    @classmethod
    def from_dict(cls, skeleton: SkeletonBase, dico: dict) -> "FullBodyConstraintSet":
        """Build a FullBodyConstraintSet from a dict (e.g. loaded from JSON)."""
        frame_indices = torch.tensor(dico["frame_indices"])
        device = skeleton.device if hasattr(skeleton, "device") else "cpu"
        local_rot = torch.tensor(dico["local_joints_rot"], device=device)
        local_rot_mats = axis_angle_to_matrix(local_rot)
        local_rot_mats = _convert_constraint_local_rots_to_skeleton(local_rot_mats, skeleton)
        global_joints_rots, global_joints_positions, _ = skeleton.fk(
            local_rot_mats,
            torch.tensor(dico["root_positions"], device=device),
        )
        smooth_root_2d = None
        if "smooth_root_2d" in dico:
            smooth_root_2d = torch.tensor(dico["smooth_root_2d"], device=device)

        return cls(
            skeleton,
            frame_indices=frame_indices,
            global_joints_positions=global_joints_positions,
            global_joints_rots=global_joints_rots,
            smooth_root_2d=smooth_root_2d,
        )

```

```python

class EndEffectorConstraintSet:
    """Constraint set fixing selected end-effector positions and rotations on given frames."""

    name = "end-effector"

    def __init__(
        self,
        skeleton: SkeletonBase,
        frame_indices: Tensor,
        global_joints_positions: Tensor,
        global_joints_rots: Tensor,
        smooth_root_2d: Optional[Tensor],
        *,
        joint_names: list[str],
        to_crop: bool = False,
    ) -> None:
        self.skeleton = skeleton
        self.frame_indices = frame_indices
        self.joint_names = joint_names

        # joint_names are constant for all the frames
        rot_joint_names, pos_joint_names = self.skeleton.expand_joint_names(self.joint_names)
        # indexing works for motion_rep with smooth root only (contains pelvis index)
        self.pos_indices = torch.tensor([self.skeleton.bone_index[jname] for jname in pos_joint_names])
        self.rot_indices = torch.tensor([self.skeleton.bone_index[jname] for jname in rot_joint_names])

        # if we pass the full smooth root 3D as input
        if smooth_root_2d is not None and smooth_root_2d.shape[-1] == 3:
            smooth_root_2d = smooth_root_2d[..., [0, 1]]

        if to_crop:
            global_joints_positions = global_joints_positions[frame_indices]
            global_joints_rots = global_joints_rots[frame_indices]
            if smooth_root_2d is not None:
                smooth_root_2d = smooth_root_2d[frame_indices]
        else:
            assert len(global_joints_positions) == len(
                frame_indices
            ), "The number of global positions should be match the number of frames"
            assert len(global_joints_rots) == len(
                frame_indices
            ), "The number of global joint rotations should be match the number of frames"
            if smooth_root_2d is not None:
                assert len(smooth_root_2d) == len(
                    frame_indices
                ), "The number of smooth root 2d (if specified) should be match the number of frames"

        if smooth_root_2d is None:
            # substitute the smooth root 2d with the real root
            smooth_root_2d = global_joints_positions[:, skeleton.root_idx, [0, 2]]

        # root y: from smooth or pelvis is the same
        self.root_y_pos = global_joints_positions[:, skeleton.root_idx, 1]

        self.global_joints_positions = global_joints_positions
        self.global_root_heading = compute_global_heading(global_joints_positions, skeleton)
        self.global_joints_rots = global_joints_rots
        self.smooth_root_2d = smooth_root_2d

    def update_constraints(self, data_dict: dict, index_dict: dict) -> None:
        """Append constrained joint positions/rots, smooth root 2D, root y, and heading to
        data/index dicts."""
        crop_frames_indexing = torch.arange(len(self.frame_indices), device=self.frame_indices.device)

        # constraint positions
        pos_indices_real = create_pairs(
            self.frame_indices,
            self.pos_indices,
        )
        pos_indices_crop = create_pairs(
            crop_frames_indexing,
            self.pos_indices,
        )
        data_dict["global_joints_positions"].append(self.global_joints_positions[tuple(pos_indices_crop.T)])
        index_dict["global_joints_positions"].append(pos_indices_real)

        # constraint rotations
        rot_indices_real = create_pairs(
            self.frame_indices,
            self.rot_indices,
        )
        rot_indices_crop = create_pairs(
            crop_frames_indexing,
            self.rot_indices,
        )
        data_dict["global_joints_rots"].append(self.global_joints_rots[tuple(rot_indices_crop.T)])
        index_dict["global_joints_rots"].append(rot_indices_real)

        # as we use smooth root, also constraint the smooth root to get the same full body
        # maybe keep storing the hips offset, if we smooth it ourselves
        data_dict["smooth_root_2d"].append(self.smooth_root_2d)
        index_dict["smooth_root_2d"].append(self.frame_indices)

        # constraint the y pos of the root
        data_dict["root_y_pos"].append(self.root_y_pos)
        index_dict["root_y_pos"].append(self.frame_indices)

        # constraint the global heading
        data_dict["global_root_heading"].append(self.global_root_heading)
        index_dict["global_root_heading"].append(self.frame_indices)

    def crop_move(self, start: int, end: int) -> "EndEffectorConstraintSet":
        """Return a new EndEffectorConstraintSet for the cropped frame range [start, end)."""
        mask = (self.frame_indices >= start) & (self.frame_indices < end)

        cls = type(self)
        kwargs = {}
        if not hasattr(cls, "joint_names"):
            kwargs["joint_names"] = self.joint_names

        return cls(
            self.skeleton,
            self.frame_indices[mask] - start,
            self.global_joints_positions[mask],
            self.global_joints_rots[mask],
            self.smooth_root_2d[mask],
            **kwargs,
        )

    def get_save_info(self) -> dict:
        """Return a dict for JSON save: type, frame_indices, local_joints_rot, root_positions, smooth_root_2d, joint_names."""
        local_joints_rot = self.skeleton.global_rots_to_local_rots(self.global_joints_rots)
        if isinstance(self.skeleton, SOMASkeleton30):
            local_joints_rot = self.skeleton.to_SOMASkeleton77(local_joints_rot)
        local_joints_rot = matrix_to_axis_angle(local_joints_rot)

        root_positions = self.global_joints_positions[:, self.skeleton.root_idx]
        output = {
            "type": self.name,
            "frame_indices": self.frame_indices,
            "local_joints_rot": local_joints_rot,
            "root_positions": root_positions,
            "smooth_root_2d": self.smooth_root_2d,
        }
        if not hasattr(self.__class__, "joint_names"):
            # save the joint_names for this base class
            # but not for children
            output["joint_names"] = self.joint_names
        return output

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "EndEffectorConstraintSet":
        self.frame_indices = _tensor_to(self.frame_indices, device, dtype)
        self.pos_indices = _tensor_to(self.pos_indices, device, dtype)
        self.rot_indices = _tensor_to(self.rot_indices, device, dtype)
        self.root_y_pos = _tensor_to(self.root_y_pos, device, dtype)
        self.global_joints_positions = _tensor_to(self.global_joints_positions, device, dtype)
        self.global_root_heading = _tensor_to(self.global_root_heading, device, dtype)
        self.global_joints_rots = _tensor_to(self.global_joints_rots, device, dtype)
        self.smooth_root_2d = _tensor_to(self.smooth_root_2d, device, dtype)
        if device is not None and hasattr(self.skeleton, "to"):
            self.skeleton = self.skeleton.to(device)
        return self

    @classmethod
    def from_dict(cls, skeleton: SkeletonBase, dico: dict) -> "EndEffectorConstraintSet":
        """Build an EndEffectorConstraintSet from a dict (e.g. loaded from JSON)."""
        frame_indices = torch.tensor(dico["frame_indices"])
        device = skeleton.device if hasattr(skeleton, "device") else "cpu"
        local_rot = torch.tensor(dico["local_joints_rot"], device=device)
        local_rot_mats = axis_angle_to_matrix(local_rot)
        local_rot_mats = _convert_constraint_local_rots_to_skeleton(local_rot_mats, skeleton)
        global_joints_rots, global_joints_positions, _ = skeleton.fk(
            local_rot_mats,
            torch.tensor(dico["root_positions"], device=device),
        )
        smooth_root_2d = None
        if "smooth_root_2d" in dico:
            smooth_root_2d = torch.tensor(dico["smooth_root_2d"], device=device)

        kwargs = {}
        if not hasattr(cls, "joint_names"):
            kwargs["joint_names"] = dico["joint_names"]

        return cls(
            skeleton,
            frame_indices=frame_indices,
            global_joints_positions=global_joints_positions,
            global_joints_rots=global_joints_rots,
            smooth_root_2d=smooth_root_2d,
            **kwargs,
        )



```

---

### <span style="color:#76b900">2. Why Is Dynamic Cropping (`crop_move()`) Necessary Instead of Fixed-Length Processing?</span>

Kimodo is trained on a maximum window of **10 seconds**, but users work on arbitrary-duration timelines.

**Problems with fixed-length processing:**
- Short clips (e.g. 2 seconds) would waste GPU memory with padding artifacts
- Long timelines (e.g. 45 seconds) would have late-timestamp constraints simply cut off

**How `crop_move()` solves this:**

- **Sliding Context Window:** The method "slides" the 10-second window to the relevant timestamp. A constraint at the 25-second mark causes the system to crop a context window around that point, generate the local motion, and move on.
- **Sequential Generation + Temporal Stitching:** Long sequences are generated in overlapping segments. The transition zones defined by `crop_move()` ensure physical continuity (momentum, foot placement) across segments.
- **Constraint Stability:** By isolating frames a specific constraint needs to influence, the system prevents a single late-timeline constraint from causing jitter in earlier motion.
# [constraints.py]
```python
def crop_move(self, start: int, end: int) -> "Root2DConstraintSet":
    """Return a new constraint set for the cropped frame range [start, end)."""
    mask = (self.frame_indices >= start) & (self.frame_indices < end)

    if self.global_root_heading is not None:
        masked_global_root_heading = self.global_root_heading[mask]
    else:
        masked_global_root_heading = None

    return Root2DConstraintSet(
        self.skeleton,
        self.frame_indices[mask] - start,
        self.smooth_root_2d[mask],
        global_root_heading=masked_global_root_heading,
    )

```

```python
def crop_move(self, start: int, end: int) -> "FullBodyConstraintSet":
    """Return a new FullBodyConstraintSet for the cropped frame range [start, end)."""
    mask = (self.frame_indices >= start) & (self.frame_indices < end)
    return FullBodyConstraintSet(
        self.skeleton,
        self.frame_indices[mask] - start,
        self.global_joints_positions[mask],
        self.global_joints_rots[mask],
        self.smooth_root_2d[mask],
    )
```

```python
def crop_move(self, start: int, end: int) -> "EndEffectorConstraintSet":
    """Return a new EndEffectorConstraintSet for the cropped frame range [start, end)."""
    mask = (self.frame_indices >= start) & (self.frame_indices < end)

    cls = type(self)
    kwargs = {}
    if not hasattr(cls, "joint_names"):
        kwargs["joint_names"] = self.joint_names

    return cls(
        self.skeleton,
        self.frame_indices[mask] - start,
        self.global_joints_positions[mask],
        self.global_joints_rots[mask],
        self.smooth_root_2d[mask],
        **kwargs,
    )
```
# [kimodo_mode.py]
```python
constraint_lst_base = [
    constraint.crop_move(current_frame, current_frame + num_frame) for constraint in constraint_lst
]  # this move temporally but not spatially
```


---

### <span style="color:#76b900">3.Analyze conflicts between simultaneously constraining root position and end-effector position. How does soft constraint guidance resolve them?</span>

**The conflict scenario:** A `Root2D` constraint places the pelvis at `(5.0, 0.0)` while an `EndEffector` constraint demands the right hand at `(0.0, 1.5)` — physically impossible if the arm+torso span only 2 meters.

**Why "hard" constraints fail:** Directly overwriting feature vector values produces anatomically impossible configurations — joints stretched across the scene — causing visual artifacts and jitter.

**Kimodo's solution — Classifier-Free Guidance (CFG):**

The denoising prediction at each step is computed as:

<!-- ```
x̂₀ = D∅ + w_text(D_text − D∅) + w_constr(D_constr − D∅)
``` -->

<div style="border: 2px solid #76b900; border-radius: 8px; padding: 12px; background:#1a1a1a;">

$$\hat{x}_0 = \mathcal{D}_\emptyset + w_{text}(\mathcal{D}_{text} - \mathcal{D}_\emptyset) + w_{constr}(\mathcal{D}_{constr} - \mathcal{D}_\emptyset)$$

</div>

Where:
- `D∅` = unconditional output (generic human motion prior)
- `D_text` = text-only conditioned output
- `D_constr` = constraint-only conditioned output
- Default weights: `w_text = 2`, `w_constr = 2` (user-adjustable)

**Three critical advantages of this soft approach:**

1. **Negotiated Compromise:** At moderate weights, the denoiser finds a middle ground. If the root and hand targets are too far apart, the model might shift the root slightly closer to the hand and extend the hand as far as possible without breaking the shoulder joint.

2. **Preservation of Plausibility:** Because the "unconditional" and "text-conditioned" components of the formula represent the model's knowledge of natural human movement, the final pose is always anchored in physical reality. The model effectively says, "I will get as close to your target as I can without looking unnatural."

3. **Iterative Refinement:** Diffusion is a multi-step process. In early steps, the model moves toward the general area of the constraints. In later steps, it fine-tunes the pose. This allows the system to globally balance competing objectives across the entire timeline, ensuring temporal smoothness even when spatial targets are difficult to reach.

```python
[generation.py]
model_constraints = compute_model_constraints_lst(session, model_bundle, sum(num_frames), device)
cfg_weight = cfg_weight or [2.0, 2.0]
postprocess_parameters = postprocess_parameters or {}
transitions_parameters = transitions_parameters or {}
```

```python
[kimodo_model.py]
cfg_weight: Optional[float] = [2.0, 2.0],
cfg_weight: Optional[float] = [2.0, 2.0],
cfg_weight: Classifier-free guidance scale(s).  
# A two-element list [text_cfg, constraint_cfg] controls text and constraint guidance independently.
```

```python
cur_mot = self.denoising_step(
  cur_mot,
  pad_mask,
  text_feat,
  text_pad_mask,
  t,
  first_heading_angle,
  motion_mask,
  observed_motion,
  num_denoising_steps,
  cfg_weight,
  guide_masks=guide_masks,
  cfg_type=cfg_type,
)
```

```python
return {
  "cfg_type": "separated",
  "cfg_weight": [
      float(cfg.get("text_weight", 2.0)),
      float(cfg.get("constraint_weight", 2.0)),
  ],
}
```
---

### <span style="color:#76b900">4. Why would increasing diffusion_steps indefinitely not monotonically improve quality?</span>

Kimodo is trained with **1,000 DDPM steps** but uses **DDIM inference** allowing high-quality results in **~100 steps**.

**Why more steps don't monotonically help:**

- **Over-Smoothing:** Excessive denoising causes "mean-seeking" behavior — removing the organic high-frequency details that make motion look human, resulting in robotic or "floaty" movement.
- **Numerical Drift:** Each step is an estimation. More steps accumulate more numerical errors, potentially drifting the motion off the learned manifold of natural movement.
- **Constraint Tension:** Kimodo performs control mask imputation at every step:
<!-- 
```
x̃ₜ = m ⊙ x_tgt + (1 − m) ⊙ xₜ
``` -->
<div style="border: 2px solid #76b900; border-radius: 8px; padding: 12px; background:#1a1a1a;">

$$\tilde{x}_t = m \odot x_{tgt} + (1 - m) \odot x_t$$

</div>


At a large step count, adjacent noise levels are so close that the model's denoising prediction becomes meaningless. Going beyond 100 steps offers no additional correction opportunity — it only compounds instability.
Each inference step jumps across a large chunk of the noise schedule. If you force 2000 DDIM steps instead, each step operates at a noise level so close to adjacent that the model's denoising prediction becomes meaningless noise.
Going beyond 100 steps doesn't give the model "more chances to correct itself" .


```python
[diffusion.py]
def space_timesteps(self, num_denoising_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
  """Return (use_timesteps, map_tensor) for a subsampled denoising schedule of
  num_denoising_steps."""
  nsteps_train = self.num_base_steps
  frac_stride = (nsteps_train - 1) / max(1, num_denoising_steps - 1)
  use_timesteps = torch.round(torch.arange(nsteps_train, device=self.device) * frac_stride).to(torch.long)
  use_timesteps = torch.clamp(use_timesteps, max=nsteps_train - 1)
  map_tensor = torch.arange(nsteps_train, device=self.device, dtype=torch.long)[use_timesteps]
  return use_timesteps, map_tensor
```

```python
[kimodo_model.py]
def denoising_step(
  self,
  motion: torch.Tensor,
  pad_mask: torch.Tensor,
  text_feat: torch.Tensor,
  text_pad_mask: torch.Tensor,
  t: torch.Tensor,
  first_heading_angle: Optional[torch.Tensor],
  motion_mask: torch.Tensor,
  observed_motion: torch.Tensor,
  num_denoising_steps: torch.Tensor,
  cfg_weight: Union[float, Tuple[float, float]],
  guide_masks: Optional[Dict] = None,
  cfg_type: Optional[str] = None,
) -> torch.Tensor:
  """Single denoising step.

  Returns:
      torch.Tensor: [B, T, D] noisy motion input to t-1
  """
  # subsample timesteps
  #   NOTE: do this at every step due to ONNX export, i.e. num_samp_stepsmay change dynamically when
  #       running onnx version so need to account for that.
  num_denoising_steps = num_denoising_steps[0]
  use_timesteps, map_tensor = self.diffusion.space_timesteps(num_denoising_steps)
  self.diffusion.calc_diffusion_vars(use_timesteps)

  # first compute initial clean prediction from denoiser
  t_map = map_tensor[t]

  with torch.inference_mode():
      pred_clean = self.denoiser(
          cfg_weight,
          motion,
          pad_mask,
          text_feat,
          text_pad_mask,
          t_map,
          first_heading_angle,
          motion_mask,
          observed_motion,
          cfg_type=cfg_type,
      )

  # sampler computes next step noisy motion
  x_tm1 = self.sampler(use_timesteps, motion, pred_clean, t)
  return x_tm1

```

```python
[kimodo_model.py]
def _generate(
  self,
  texts: List[str],
  max_frames: int,
  num_denoising_steps: int,
  pad_mask: torch.Tensor,
  first_heading_angle: Optional[torch.Tensor],
  motion_mask: torch.Tensor,
  observed_motion: torch.Tensor,
  cfg_weight: Optional[float] = 2.0,
  text_feat: Optional[torch.Tensor] = None,
  text_pad_mask: Optional[torch.Tensor] = None,
  guide_masks: Optional[Dict] = None,
  cfg_type: Optional[str] = None,
  progress_bar=tqdm,
) -> torch.Tensor:
  """Sample full denoising loop.

  Args:
      texts (List[str]): batch of text prompts to use for sampling (if text_feat is not passed in)
  """

  device = self.device
  if text_feat is None:
      assert text_pad_mask is None
      log.info("Encoding text...")
      text_feat, text_length = self.text_encoder(texts)
      text_feat = text_feat.to(device)

      # handle empty string (set to zero)
      empty_text_mask = [len(text.strip()) == 0 for text in texts]
      text_feat[empty_text_mask] = 0

      # Create the pad mask for the text
      batch_size, maxlen = text_feat.shape[:2]
      tensor_text_length = torch.tensor(text_length, device=device)
      tensor_text_length[empty_text_mask] = 0
      text_pad_mask = torch.arange(maxlen, device=device).expand(batch_size, maxlen) < tensor_text_length[:, None]

  if motion_mask is not None:
      if motion_mask.dtype == torch.bool:
          motion_mask = 1 * motion_mask

  batch_size = text_feat.shape[0]

  # sample loop
  indices = list(range(num_denoising_steps))[::-1]
  shape = (batch_size, max_frames, self.motion_rep.motion_rep_dim)
  cur_mot = torch.randn(shape, device=self.device)
  num_denoising_steps = torch.tensor(
      [num_denoising_steps], device=self.device
  )  # this and t need to be tensor for onnx export
  # init diffusion with correct num steps before looping
  use_timesteps = self.diffusion.space_timesteps(num_denoising_steps[0])[0]
  self.diffusion.calc_diffusion_vars(use_timesteps)
  for i in progress_bar(indices):
      t = torch.tensor([i] * cur_mot.size(0), device=self.device)
      with torch.inference_mode():
          cur_mot = self.denoising_step(
              cur_mot,
              pad_mask,
              text_feat,
              text_pad_mask,
              t,
              first_heading_angle,
              motion_mask,
              observed_motion,
              num_denoising_steps,
              cfg_weight,
              guide_masks=guide_masks,
              cfg_type=cfg_type,
          )
  return cur_mot

```



---

### <span style="color:#76b900">5. How do SOMA 30↔77 joint conversion utilities enable skeleton compatibility while maintaining constraint semantics?

</span>

The SOMA body model exists in two representations: a simpler **30-joint** rig (user-facing) and a high-fidelity **77-joint** rig (model-facing, includes fingers, toes, detailed spine).

**The challenge:** A `FullBodyConstraintSet` specified in 30 joints produces a feature vector of the wrong length for the 77-joint denoiser.

**How the conversion utilities preserve constraint semantics:**

- **Up-sampling:** The `to_SOMASkeleton77` method maps 30-joint rotations into the correct 77-joint slots, ensuring a "Right Arm" rotation influences the right parent/child joints.
- **FK Validation:** After conversion, Forward Kinematics (FK) is run to verify that the global end-effector positions (hands, feet) remain identical to what the user requested — regardless of joint count change.
- **Mask Alignment:** The 30-joint constraint is up-sampled into a 77-joint control mask *before* the denoiser runs, allowing the model to correctly "lock" the relevant channels.
- **Serialization Invariance:** The `get_save_info` / `from_dict` logic saves rotations in a universal format (axis-angle), re-applying the SOMA conversion upon loading — so a motion generated on 77 joints can be re-constrained by a user on a 30-joint rig without spatial loss.

```{note}
For SOMA models, constraints may be authored or displayed on the full `somaskel77` skeleton, but Kimodo converts them to the reduced `somaskel30` representation before passing them to the model.
```

```python
[constraints.py]
def _convert_constraint_local_rots_to_skeleton(local_rot_mats: Tensor, skeleton: SkeletonBase) -> Tensor:
  """Convert loaded local rotation matrices to match the skeleton's joint count.

  Handles SOMA 30↔77: constraint files may have been saved with 30 or 77 joints while the session
  skeleton (e.g. from the SOMA30 model) uses SOMASkeleton77.
  """
  n_joints = local_rot_mats.shape[-3]
  skeleton_joints = skeleton.nbjoints
  if n_joints == skeleton_joints:
      return local_rot_mats
  if n_joints == 77 and skeleton_joints == 30 and isinstance(skeleton, SOMASkeleton30):
      return skeleton.from_SOMASkeleton77(local_rot_mats)
  if n_joints == 30 and skeleton_joints == 77 and isinstance(skeleton, SOMASkeleton77):
      skel30 = SOMASkeleton30()
      return skel30.to_SOMASkeleton77(local_rot_mats)
  raise ValueError(
      f"Constraint joint count ({n_joints}) does not match skeleton joint count "
      f"({skeleton_joints}). Only SOMA 30↔77 conversion is supported."
  )

```

```python
[soma_skin.py]

def skin(self, joint_rotmat, joint_pos, rot_is_global=False):
    """
    joint_rotmat: [T, J, 3, 3] local or global joint rotation matrices
    joint_pos: [T, J, 3] global joint positions
    rot_is_global: bool, if True, joint_rotmat is global rotation matrices, otherwise it is local rotation matrices and FK is performed internally
    """
    nF, nJ = joint_pos.shape[:2]
    device = joint_rotmat.device

    if nJ != self.skeleton_skin.nbjoints:
        assert nJ == 30, "SOMASkin currently only supports 30-joint or 77-joint skeletons"

        # make sure we have local joint rotations
        if rot_is_global:
            local_joint_rots_mats_subset = global_rots_to_local_rots(joint_rotmat, self.skeleton_input)
        else:
            local_joint_rots_mats_subset = joint_rotmat

        local_joint_rots_mats = self.skeleton_input.to_SOMASkeleton77(local_joint_rots_mats_subset)

        # FK to get the global joint pos and rot
        neutral_joints_seq = self.skeleton_skin.neutral_joints[None].repeat((nF, 1, 1)).to(device)
        new_joint_pos, joint_rotmat = batch_rigid_transform(
            local_joint_rots_mats,
            neutral_joints_seq,
            self.skeleton_skin.joint_parents.to(device),
            self.skeleton_skin.root_idx,
        )
        joint_pos = new_joint_pos + joint_pos[:, self.skeleton_input.root_idx : self.skeleton_input.root_idx + 1]
        nJ = self.skeleton_skin.nbjoints
        rot_is_global = True

    # prepare full transformation matrices
    fk_transform = torch.eye(4, device=device)[None, None].repeat(nF, nJ, 1, 1)
    fk_transform[..., :3, 3] = joint_pos
    if rot_is_global:
        fk_transform[..., :3, :3] = joint_rotmat
    else:
        neutral_joints_seq = self.skeleton_skin.neutral_joints[None].repeat((nF, 1, 1)).to(device)
        # FK to get the global rotations
        _, global_joint_rotmat = batch_rigid_transform(
            joint_rotmat,
            neutral_joints_seq,
            self.skeleton_skin.joint_parents.to(device),
            self.skeleton_skin.root_idx,
        )
        fk_transform[..., :3, :3] = global_joint_rotmat

    vertices = self.lbs(fk_transform)
    return vertices

```

```python
[playback.py]

elif isinstance(self.skeleton, SOMASkeleton77):
  skel30_names = {name for name, _ in SOMASkeleton30.bone_order_names_with_parents}
  hidden_gizmo_joints = {name for name in self.skeleton.bone_order_names if name not in skel30_names}
  hidden_gizmo_joints |= {
      "RightHandThumbEnd",
      "RightHandMiddleEnd",
      "LeftHandThumbEnd",
      "LeftHandMiddleEnd",
      "LeftEye",
      "RightEye",
      "Jaw",
  }
```


```python
[definition.py]


class SOMASkeleton77(SkeletonBase):
  """High-detail 77-joint SOMA skeleton with full finger and toe chains."""

  name = "somaskel77"

  right_foot_joint_names = [
      "RightFoot",
      "RightToeBase",
      "RightToeEnd",
  ]  # in order of chain
  left_foot_joint_names = [
      "LeftFoot",
      "LeftToeBase",
      "LeftToeEnd",
  ]  # in order of chain
  right_hand_joint_names = [
      "RightHand",
      "RightHandThumb1",
      "RightHandThumb2",
      "RightHandThumb3",
      "RightHandThumbEnd",
      "RightHandIndex1",
      "RightHandIndex2",
      "RightHandIndex3",
      "RightHandIndex4",
      "RightHandIndexEnd",
      "RightHandMiddle1",
      "RightHandMiddle2",
      "RightHandMiddle3",
      "RightHandMiddle4",
      "RightHandMiddleEnd",
      "RightHandRing1",
      "RightHandRing2",
      "RightHandRing3",
      "RightHandRing4",
      "RightHandRingEnd",
      "RightHandPinky1",
      "RightHandPinky2",
      "RightHandPinky3",
      "RightHandPinky4",
      "RightHandPinkyEnd",
  ]  # in order of chain
  left_hand_joint_names = [
      "LeftHand",
      "LeftHandThumb1",
      "LeftHandThumb2",
      "LeftHandThumb3",
      "LeftHandThumbEnd",
      "LeftHandIndex1",
      "LeftHandIndex2",
      "LeftHandIndex3",
      "LeftHandIndex4",
      "LeftHandIndexEnd",
      "LeftHandMiddle1",
      "LeftHandMiddle2",
      "LeftHandMiddle3",
      "LeftHandMiddle4",
      "LeftHandMiddleEnd",
      "LeftHandRing1",
      "LeftHandRing2",
      "LeftHandRing3",
      "LeftHandRing4",
      "LeftHandRingEnd",
      "LeftHandPinky1",
      "LeftHandPinky2",
      "LeftHandPinky3",
      "LeftHandPinky4",
      "LeftHandPinkyEnd",
  ]  # in order of chain

  hip_joint_names = ["RightLeg", "LeftLeg"]  # in order [right, left]

  bone_order_names_with_parents = [
      ("Hips", None),
      ("Spine1", "Hips"),
      ("Spine2", "Spine1"),
      ("Chest", "Spine2"),
      ("Neck1", "Chest"),
      ("Neck2", "Neck1"),
      ("Head", "Neck2"),
      ("HeadEnd", "Head"),
      ("Jaw", "Head"),
      ("LeftEye", "Head"),
      ("RightEye", "Head"),
      ("LeftShoulder", "Chest"),
      ("LeftArm", "LeftShoulder"),
      ("LeftForeArm", "LeftArm"),
      ("LeftHand", "LeftForeArm"),
      ("LeftHandThumb1", "LeftHand"),
      ("LeftHandThumb2", "LeftHandThumb1"),
      ("LeftHandThumb3", "LeftHandThumb2"),
      ("LeftHandThumbEnd", "LeftHandThumb3"),
      ("LeftHandIndex1", "LeftHand"),
      ("LeftHandIndex2", "LeftHandIndex1"),
      ("LeftHandIndex3", "LeftHandIndex2"),
      ("LeftHandIndex4", "LeftHandIndex3"),
      ("LeftHandIndexEnd", "LeftHandIndex4"),
      ("LeftHandMiddle1", "LeftHand"),
      ("LeftHandMiddle2", "LeftHandMiddle1"),
      ("LeftHandMiddle3", "LeftHandMiddle2"),
      ("LeftHandMiddle4", "LeftHandMiddle3"),
      ("LeftHandMiddleEnd", "LeftHandMiddle4"),
      ("LeftHandRing1", "LeftHand"),
      ("LeftHandRing2", "LeftHandRing1"),
      ("LeftHandRing3", "LeftHandRing2"),
      ("LeftHandRing4", "LeftHandRing3"),
      ("LeftHandRingEnd", "LeftHandRing4"),
      ("LeftHandPinky1", "LeftHand"),
      ("LeftHandPinky2", "LeftHandPinky1"),
      ("LeftHandPinky3", "LeftHandPinky2"),
      ("LeftHandPinky4", "LeftHandPinky3"),
      ("LeftHandPinkyEnd", "LeftHandPinky4"),
      ("RightShoulder", "Chest"),
      ("RightArm", "RightShoulder"),
      ("RightForeArm", "RightArm"),
      ("RightHand", "RightForeArm"),
      ("RightHandThumb1", "RightHand"),
      ("RightHandThumb2", "RightHandThumb1"),
      ("RightHandThumb3", "RightHandThumb2"),
      ("RightHandThumbEnd", "RightHandThumb3"),
      ("RightHandIndex1", "RightHand"),
      ("RightHandIndex2", "RightHandIndex1"),
      ("RightHandIndex3", "RightHandIndex2"),
      ("RightHandIndex4", "RightHandIndex3"),
      ("RightHandIndexEnd", "RightHandIndex4"),
      ("RightHandMiddle1", "RightHand"),
      ("RightHandMiddle2", "RightHandMiddle1"),
      ("RightHandMiddle3", "RightHandMiddle2"),
      ("RightHandMiddle4", "RightHandMiddle3"),
      ("RightHandMiddleEnd", "RightHandMiddle4"),
      ("RightHandRing1", "RightHand"),
      ("RightHandRing2", "RightHandRing1"),
      ("RightHandRing3", "RightHandRing2"),
      ("RightHandRing4", "RightHandRing3"),
      ("RightHandRingEnd", "RightHandRing4"),
      ("RightHandPinky1", "RightHand"),
      ("RightHandPinky2", "RightHandPinky1"),
      ("RightHandPinky3", "RightHandPinky2"),
      ("RightHandPinky4", "RightHandPinky3"),
      ("RightHandPinkyEnd", "RightHandPinky4"),
      ("LeftLeg", "Hips"),
      ("LeftShin", "LeftLeg"),
      ("LeftFoot", "LeftShin"),
      ("LeftToeBase", "LeftFoot"),
      ("LeftToeEnd", "LeftToeBase"),
      ("RightLeg", "Hips"),
      ("RightShin", "RightLeg"),
      ("RightFoot", "RightShin"),
      ("RightToeBase", "RightFoot"),
      ("RightToeEnd", "RightToeBase"),
  ]

  @property
  def relaxed_hands_rest_pose(self):
      # lazy loading
      if hasattr(self, "_relaxed_hands_rest_pose"):
          return self._relaxed_hands_rest_pose

      relaxed_hands_pose_path = Path(self.folder) / "relaxed_hands_rest_pose.npy"
      relaxed_hands_rest_pose = torch.from_numpy(np.load(relaxed_hands_pose_path)).squeeze()
      self.register_buffer(
          "_relaxed_hands_rest_pose",
          relaxed_hands_rest_pose,
          persistent=False,
      )
      return self._relaxed_hands_rest_pose

  

```

```python
[app.py]

def load_model(self, model_name: str) -> ModelBundle:
    if model_name in self.models:
        return self.models[model_name]

    print(f"Loading model {model_name}...")
    try:
        model = load_model(modelname=model_name, device=self.device)
    except Exception as e:
        print(f"Error loading model: {e}\nMake sure text encoder server is running!")
        raise e

    if hasattr(model, "text_encoder"):
        model.text_encoder = CachedTextEncoder(model.text_encoder, model_name=model_name)

    skeleton = model.motion_rep.skeleton
    if isinstance(skeleton, SOMASkeleton30):
        skeleton = skeleton.somaskel77.to(model.device)
    bundle = ModelBundle(
        model=model,
        motion_rep=model.motion_rep,
        skeleton=skeleton,
        model_fps=model.motion_rep.fps,
    )
    self.models[model_name] = bundle
    print(f"Model {model_name} loaded successfully")
    self.prewarm_embedding_cache(model_name, bundle.model)
    return bundle
```

---

###  <span style="color:#76b900">6. Why does the constraint system maintain local/global rotation representations? What does each enable?
</span>

Both representations serve distinct, irreplaceable roles in the two-stage pipeline.

#### Global Rotations (orientation relative to the world)

<!-- | Role | Reason |
|---|---|
| Direct world-space imputation | User constraints (e.g., "reach for door handle") are defined in world space; global rotations allow direct overwriting without parent-offset math |
| Constraint independence | Locks a hand's orientation regardless of shoulder/elbow state; prevents parent-joint errors from propagating |
| Root-Body coordination | Stage 1 (Root Denoiser) uses global features to synchronize trajectory and body orientation | -->

<table style="border-collapse: collapse; width: 100%;">
  <thead>
    <tr style="background-color: #5e7f25; color: white;">
      <th style="border: 1px solid #76b900; padding: 10px;">Role</th>
      <th style="border: 1px solid #76b900; padding: 10px;">Reason</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #1a1a1a;">
      <td style="border: 1px solid #76b900; padding: 10px;">Direct world-space imputation</td>
      <td style="border: 1px solid #76b900; padding: 10px;">User constraints are defined in world space; global rotations allow direct overwriting without parent-offset math</td>
    </tr>
    <tr style="background-color: #2a2a2a;">
      <td style="border: 1px solid #76b900; padding: 10px;">Constraint independence</td>
      <td style="border: 1px solid #76b900; padding: 10px;">Locks a hand's orientation regardless of shoulder/elbow state; prevents parent-joint errors from propagating</td>
    </tr>
    <tr style="background-color: #1a1a1a;">
      <td style="border: 1px solid #76b900; padding: 10px;">Root-Body coordination</td>
      <td style="border: 1px solid #76b900; padding: 10px;">Stage 1 (Root Denoiser) uses global features to synchronize trajectory and body orientation</td>
    </tr>
  </tbody>
</table>

#### Local Rotations (orientation relative to parent joint)

<!-- | Role | Reason |
|---|---|
| Feature invariance | Neural networks learn "walking" better in root-relative space; motion meaning doesn't change based on character position |
| Prevention of "heading flips" | Global space can cause abrupt representation flips during somersaults; local rotations keep pose consistent when upside-down |
| Linear Blend Skinning (LBS) | The final mesh deformation step requires local transformations to maintain constant bone lengths and valid FK hierarchy | -->

<table style="border-collapse: collapse; width: 100%;">
  <thead>
    <tr style="background-color: #5e7f25; color: white;">
      <th style="border: 1px solid #76b900; padding: 10px;">Role</th>
      <th style="border: 1px solid #76b900; padding: 10px;">Reason</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #1a1a1a;">
      <td style="border: 1px solid #76b900; padding: 10px;">Feature invariance</td>
      <td style="border: 1px solid #76b900; padding: 10px;">Neural networks learn "walking" better in root-relative space; motion meaning doesn't change based on character position</td>
    </tr>
    <tr style="background-color: #2a2a2a;">
      <td style="border: 1px solid #76b900; padding: 10px;">Prevention of "heading flips"</td>
      <td style="border: 1px solid #76b900; padding: 10px;">Global space can cause abrupt representation flips during somersaults; local rotations keep pose consistent when upside-down</td>
    </tr>
    <tr style="background-color: #1a1a1a;">
      <td style="border: 1px solid #76b900; padding: 10px;">Linear Blend Skinning (LBS)</td>
      <td style="border: 1px solid #76b900; padding: 10px;">The final mesh deformation step requires local transformations to maintain constant bone lengths and valid FK hierarchy</td>
    </tr>
  </tbody>
</table>

**The two-stage flow:**
1. **Stage 1 (Global):** Satisfies spatial constraints; determines where the character goes.
2. **Conversion:** Predicted global root motion localizes body features.
3. **Stage 2 (Local):** Fills in realistic body movement style, ensuring natural limb motion relative to the predicted root.


```python
[base.py]

def global_rots_to_local_rots(self, global_joint_rots: torch.Tensor):
    """Convert global joint rotations to local rotations for this hierarchy."""
    return global_rots_to_local_rots(global_joint_rots, self)

```

```python
[kimodo_motionrep.py]

global_rot_mats = cont6d_to_matrix(global_rot_data)
local_rot_mats = global_rots_to_local_rots(global_rot_mats, self.skeleton)
```

```python
def global_rots_to_local_rots(global_joint_rots: torch.Tensor, skeleton):
    """Convert global rotations to local rotations using a skeleton hierarchy.

    Args:
        global_joint_rots: Global rotation matrices with shape `(..., J, 3, 3)`.
        skeleton: Skeleton object exposing `joint_parents` and `root_idx`.

    Returns:
        Local rotation matrices with the same leading shape as the input.
    """
    # Doing big batch
    global_joint_mats, ps = einops.pack(
        [global_joint_rots],
        "* nbjoints dim1 dim2",
    )

    # obtain back the local rotations from the new global rotations
    parent_rot_mats = global_joint_mats[:, skeleton.joint_parents]

    parent_rot_mats[:, skeleton.root_idx] = torch.eye(3)  # the root joint
    parent_rot_mats_inv = parent_rot_mats.transpose(2, 3)
    local_rot_mats = torch.einsum(
        "T N m n, T N n o -> T N m o",
        parent_rot_mats_inv,
        global_joint_mats,
    )
    [local_rot_mats] = einops.unpack(local_rot_mats, ps, "* nbjoints dim1 dim2")
    return local_rot_mats
```


```python
[constraints.py]

def _convert_constraint_local_rots_to_skeleton(local_rot_mats: Tensor, skeleton: SkeletonBase) -> Tensor:
    """Convert loaded local rotation matrices to match the skeleton's joint count.

    Handles SOMA 30↔77: constraint files may have been saved with 30 or 77 joints while the session
    skeleton (e.g. from the SOMA30 model) uses SOMASkeleton77.
    """
    n_joints = local_rot_mats.shape[-3]
    skeleton_joints = skeleton.nbjoints
    if n_joints == skeleton_joints:
        return local_rot_mats
    if n_joints == 77 and skeleton_joints == 30 and isinstance(skeleton, SOMASkeleton30):
        return skeleton.from_SOMASkeleton77(local_rot_mats)
    if n_joints == 30 and skeleton_joints == 77 and isinstance(skeleton, SOMASkeleton77):
        skel30 = SOMASkeleton30()
        return skel30.to_SOMASkeleton77(local_rot_mats)
    raise ValueError(
        f"Constraint joint count ({n_joints}) does not match skeleton joint count "
        f"({skeleton_joints}). Only SOMA 30↔77 conversion is supported."
    )
  
def compute_global_heading(global_joints_positions: Tensor, skeleton: SkeletonBase) -> Tensor:
    """Compute global root heading (cos, sin) from global joint positions using skeleton."""
    root_heading_angle = compute_heading_angle(global_joints_positions, skeleton)
    global_root_heading = torch.stack([torch.cos(root_heading_angle), torch.sin(root_heading_angle)], dim=-1)
    return global_root_heading
```
---

### <span style="color:#76b900"> 7. How does the constraint system handle frame-rate independent specification? Why is this important for user experience?</span>

Kimodo maintains a constant `model_fps` (typically `30.0`) as the bridge between the user's continuous time domain and the model's discrete tensor domain.

**The mapping:**
<!-- ```
frame_index = timestamp_seconds × model_fps
``` -->

<div style="border: 2px solid #76b900; border-radius: 8px; padding: 12px; background:#1a1a1a; color:#ffffff; font-family:monospace;">
  frame_index = timestamp_seconds × model_fps
</div>

**Why this matters for UX:** Animators reason in time ("the jump takes 0.8 seconds"), not frames. If frame-rate forced users to recalculate on every project setting change, it would introduce friction and errors.

**Code pattern (from the demo):**
```python
# Seconds → frame index
session.max_frame_idx = int(session.cur_duration * session.model_fps - 1)

# Re-mapping across frame rate changes
start_sec = start_frame / old_model_fps
new_start  = int(round(start_sec * session.model_fps))
```

This architecture means that if the model is retrained at 60fps, only the `model_fps` constant changes — the entire UI timeline scales correctly without any code changes.

```python
def set_timeline_defaults(self, timeline, model_fps: float) -> None:
    timeline.set_defaults(
        default_text=DEFAULT_PROMPT,
        default_duration=int(DEFAULT_CUR_DURATION * model_fps - 1),
        min_duration=int(MIN_DURATION * model_fps - 1),  # 2 seconds minimum,
        max_duration=int(
            MAX_DURATION * model_fps - 1  # - NB_TRANSITION_FRAMES
        ),  # 10 seconds maximum, minus the transition frames, if needed
        default_num_frames_zoom=int(1.10 * 10 * model_fps),  # a bit more than the max
        max_frames_zoom=1000,
        fps=model_fps,
    )
```


```python
[mujoco_load.py]
fps = 30  # adjust to your intended playback rate

with mujoco.viewer.launch_passive(model, data) as viewer:
    # loop the motion
    while viewer.is_running():
        for frame in qpos:
            data.qpos[:] = frame
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / fps)
```
---

### <span style="color:#76b900"> 8. Why Do Under-Constrained Motions Sometimes Look More Natural Than Heavily Constrained Ones?</span>

**The training distribution matters:** 10% of Kimodo's training used zero constraints, teaching the model a rich prior over natural human motion. Dense constraints suppress this prior.
Two constraint patterns are mixed together25% of the time, and 10% of the time no constraints are used (leaving only text input). During phase 2, themaximum number of keyframes sampled for sparse constraints increases linearly from 1 to 20, and sampling is biased towards fewer keyframes to reflect common real-world use cases. Dropout with a rate of 0.1 is used during phase 1, but is removed for phase 2 to avoid dropping out conditioning constraints that are directly overwritten to the noisy motion input. During both phases, the text input is dropped 10% of the time to enable classifier-free guidance at test time.


**What happens during denoising:**

<!-- | Constraint Density | Denoising Behavior | Result |
|---|---|---|
| **Sparse** | Model finds a globally consistent shape in early steps, polishes details in late steps | Fluid, organic motion |
| **Dense** | Constrained channels are hard-reset at every single step; model can never "negotiate" | Foot sliding, jitter, overshoot artifacts | -->

<table style="border-collapse: collapse; width: 100%;">
  <thead>
    <tr style="background-color: #5e7f25; color: white;">
      <th style="border: 1px solid #76b900; padding: 10px;">Constraint Density</th>
      <th style="border: 1px solid #76b900; padding: 10px;">Denoising Behavior</th>
      <th style="border: 1px solid #76b900; padding: 10px;">Result</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #1a1a1a;">
      <td style="border: 1px solid #76b900; padding: 10px;"><strong>Sparse</strong></td>
      <td style="border: 1px solid #76b900; padding: 10px;">Model finds a globally consistent shape in early steps, polishes details in late steps</td>
      <td style="border: 1px solid #76b900; padding: 10px;">Fluid, organic motion</td>
    </tr>
    <tr style="background-color: #2a2a2a;">
      <td style="border: 1px solid #76b900; padding: 10px;"><strong>Dense</strong></td>
      <td style="border: 1px solid #76b900; padding: 10px;">Constrained channels are hard-reset at every single step; model can never "negotiate"</td>
      <td style="border: 1px solid #76b900; padding: 10px;">Foot sliding, jitter, overshoot artifacts</td>
    </tr>
  </tbody>
</table>

**The root cause:** At high constraint density, the gradient signal becomes so loud that the model is constantly over-correcting — producing  artifacts where limbs snap past targets and snap back, rather than flowing naturally toward them.

**Key insight:** The model's learned prior of human movement *is the naturalness*. Sparse constraints let more of that prior express itself.

---

### <span style="color:#76b900">9. How Does the `cfg_weight` Parameter Trade Off Constraint Adherence vs. Motion Naturalness?</span>

The CFG formula:
<!-- ```
x̂₀ = D∅ + w_text(D_text − D∅) + w_constr(D_constr − D∅)
``` -->

<div style="border: 2px solid #5e7f25; border-radius: 8px; background: #1a1a1a; overflow: hidden; display: inline-block;">
  <div style="background: #5e7f25; padding: 3px 10px; font-size: 11px; font-weight: 700; color: #000; font-family: monospace; letter-spacing: 0.04em;">DIFFUSION FORMULA</div>
  <div style="padding: 12px 16px; color: #ffffff; font-family: monospace; font-size: 15px; white-space: nowrap;">
    x̂₀ = D∅ + w_text(D_text − D∅) + w_constr(D_constr − D∅)
  </div>
</div>

<!-- | Weight Scale | Adheres To | Resulting Motion |
|---|---|---|
| **0.0 – 1.0 (Low)** | Training data distribution | Fluid & organic; may miss target path |
| **1.0 – 3.0 (Balanced)** | "The sweet spot" | Natural intent; realistic weight shifts and momentum |
| **3.0 – 5.0+ (High)** | Geometric targets | Pixel-accurate; robotic, jerky, "staccato" | -->
<div style="border: 2px solid #5e7f25; border-radius: 8px; background: #1a1a1a; overflow: hidden; display: inline-block; width: 100%;">
  <div style="background: #5e7f25; padding: 3px 10px; font-size: 11px; font-weight: 700; color: #000; font-family: monospace; letter-spacing: 0.04em;">WEIGHT SCALE</div>
  <table style="width: 100%; border-collapse: collapse; font-family: monospace; font-size: 14px; color: #ffffff;">
    <thead>
      <tr style="border-bottom: 2px solid #76b900;">
        <th style="padding: 10px 14px; text-align: left; border-right: 1px solid #333; color: #76b900;">Weight Scale</th>
        <th style="padding: 10px 14px; text-align: left; border-right: 1px solid #333; color: #76b900;">Adheres To</th>
        <th style="padding: 10px 14px; text-align: left; color: #76b900;">Resulting Motion</th>
      </tr>
    </thead>
    <tbody>
      <tr style="border-bottom: 1px solid #333;">
        <td style="padding: 10px 14px; border-right: 1px solid #333; color: #ffcc00;">0.0 – 1.0 (Low)</td>
        <td style="padding: 10px 14px; border-right: 1px solid #333;">Training data distribution</td>
        <td style="padding: 10px 14px;">Fluid & organic; may miss target path</td>
      </tr>
      <tr style="border-bottom: 1px solid #333; background: #222;">
        <td style="padding: 10px 14px; border-right: 1px solid #333; color: #76b900;">1.0 – 3.0 (Balanced)</td>
        <td style="padding: 10px 14px; border-right: 1px solid #333;">"The sweet spot"</td>
        <td style="padding: 10px 14px;">Natural intent; realistic weight shifts and momentum</td>
      </tr>
      <tr>
        <td style="padding: 10px 14px; border-right: 1px solid #333; color: #ff6b6b;">3.0 – 5.0+ (High)</td>
        <td style="padding: 10px 14px; border-right: 1px solid #333;">Geometric targets</td>
        <td style="padding: 10px 14px;">Pixel-accurate; robotic, jerky, "staccato"</td>
      </tr>
    </tbody>
  </table>
</div>

Default: `w_text = 2`, `w_constr = 2` — both in the balanced range.


After training, motions are generated using the DDIM [36] inference process with, by default, 100 denoising
steps. We leverage a classifier-free guidance approach that decomposes text and constraint conditioning to
allow control over each one individually. In particular, the model output at each denoising step is computed as
xˆ0 = 𝒟∅ + 𝑤text(𝒟text − 𝒟∅) + 𝑤constr(𝒟constr − 𝒟∅) where 𝒟∅ is the model output using no text or constraint
conditioning, 𝒟text uses only text conditioning (no constraints), and 𝒟constr uses only constraint conditioning
(no text). By default, we use 𝑤text = 2 and 𝑤constr = 2, but a user can adjust each to vary the influence of text
and constraint conditioning on the model output. Several prior works make use of gradient-based guidance to
further improve constraint following at test-time [31, 14], but we found that since our model is already directly
conditioned on the constraints, adding gradient-based guidance gave minimal improvement, substantially
increased generation time, and was generally unstable and difficult to tune.


Low weights → Natural motion (weak conditioning)
Motion follows the learned data distribution
Looks smooth, realistic, physically plausible
But: Text prompt may be ignored
Constraints may be violated.

Naturalness ↑, Constraint adherence ↓


High constraint weight → Strong constraint adherence
gets amplified. Motion is pulled strongly toward satisfying constraints
End-effectors, trajectories, etc., are followed more strictly
But: Motion may become jerky, stiff, or unnatural

Constraint adherence ↑, Naturalness ↓

```python
[cfg.py]


class ClassifierFreeGuidedModel(nn.Module):
    """Wrapper around denoiser to use classifier-free guidance at sampling time."""

    def __init__(self, model: nn.Module, cfg_type: Optional[str] = "separated"):
        """Wrap the denoiser for classifier-free guidance; cfg_type in CFG_TYPES (e.g. 'regular',
        'nocfg')."""
        super().__init__()
        self.model = model
        assert cfg_type in CFG_TYPES, f"Invalid cfg_type: {cfg_type}"
        self.cfg_type_default = cfg_type

    def forward(
        self,
        cfg_weight: Union[float, Tuple[float, float]],
        x: torch.Tensor,
        x_pad_mask: torch.Tensor,
        text_feat: torch.Tensor,
        text_feat_pad_mask: torch.Tensor,
        timesteps: torch.Tensor,
        first_heading_angle: Optional[torch.Tensor] = None,
        motion_mask: Optional[torch.Tensor] = None,
        observed_motion: Optional[torch.Tensor] = None,
        cfg_type: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Args:
            cfg_weight (float): guidance weight float or tuple of floats with (text, constraint) weights if using separated cfg
            x (torch.Tensor): [B, T, dim_motion] current noisy motion
            x_pad_mask (torch.Tensor): [B, T] attention mask, positions with True are allowed to attend, False are not
            text_feat (torch.Tensor): [B, max_text_len, llm_dim] embedded text prompts
            text_feat_pad_mask (torch.Tensor): [B, max_text_len] attention mask, positions with True are allowed to attend, False are not
            timesteps (torch.Tensor): [B,] current denoising step
            motion_mask
            observed_motion
            neutral_joints (torch.Tensor): [B, nbjoints] The neutral joints of the motions

        Returns:
            torch.Tensor: same size as input x
        """

        if cfg_type is None:
            cfg_type = self.cfg_type_default

        assert cfg_type in CFG_TYPES, f"Invalid cfg_type: {cfg_type}"

        # batched conditional and uncond pass together
        if cfg_type == "nocfg":
            return self.model(
                x,
                x_pad_mask,
                text_feat,
                text_feat_pad_mask,
                timesteps,
                first_heading_angle=first_heading_angle,
                motion_mask=motion_mask,
                observed_motion=observed_motion,
            )
        elif cfg_type == "regular":
            assert isinstance(cfg_weight, (float, int)), "cfg_weight must be a single float for regular CFG"
            # out_uncond + w * (out_text_and_constraint - out_uncond)
            text_feat = torch.concatenate([text_feat, 0 * text_feat], dim=0)
            if motion_mask is not None:
                motion_mask = torch.concatenate([motion_mask, 0 * motion_mask], dim=0)
            if observed_motion is not None:
                observed_motion = torch.concatenate([observed_motion, observed_motion], dim=0)
            if first_heading_angle is not None:
                first_heading_angle = torch.concatenate([first_heading_angle, first_heading_angle], dim=0)

            out_cond_uncond = self.model(
                torch.concatenate([x, x], dim=0),
                torch.concatenate([x_pad_mask, x_pad_mask], dim=0),
                text_feat,
                torch.concatenate([text_feat_pad_mask, False * text_feat_pad_mask], dim=0),
                torch.concatenate([timesteps, timesteps], dim=0),
                first_heading_angle=first_heading_angle,
                motion_mask=motion_mask,
                observed_motion=observed_motion,
            )

            out, out_uncond = torch.chunk(out_cond_uncond, 2)
            out_new = out_uncond + (cfg_weight * (out - out_uncond))
        elif cfg_type == "separated":
            assert len(cfg_weight) == 2, "cfg_weight must be a tuple of two floats for separated CFG"
            # out_uncond + w_text * (out_text - out_uncond) + w_constraint * (out_constraint - out_uncond)
            text_feat = torch.concatenate([text_feat, 0 * text_feat, 0 * text_feat], dim=0)
            if motion_mask is not None:
                motion_mask = torch.concatenate([0 * motion_mask, motion_mask, 0 * motion_mask], dim=0)
            if observed_motion is not None:
                observed_motion = torch.concatenate([observed_motion, observed_motion, observed_motion], dim=0)
            if first_heading_angle is not None:
                first_heading_angle = torch.concatenate(
                    [first_heading_angle, first_heading_angle, first_heading_angle],
                    dim=0,
                )

            out_cond_uncond = self.model(
                torch.concatenate([x, x, x], dim=0),
                torch.concatenate([x_pad_mask, x_pad_mask, x_pad_mask], dim=0),
                text_feat,
                torch.concatenate(
                    [
                        text_feat_pad_mask,
                        False * text_feat_pad_mask,
                        False * text_feat_pad_mask,
                    ],
                    dim=0,
                ),
                torch.concatenate([timesteps, timesteps, timesteps], dim=0),
                first_heading_angle=first_heading_angle,
                motion_mask=motion_mask,
                observed_motion=observed_motion,
            )

            out_text, out_constraint, out_uncond = torch.chunk(out_cond_uncond, 3)
            out_new = (
                out_uncond + (cfg_weight[0] * (out_text - out_uncond)) + (cfg_weight[1] * (out_constraint - out_uncond))
            )
        else:
            raise ValueError(f"Invalid cfg_type: {cfg_type}")

        return out_new
```

---

### <span style="color:#76b900"> 10. How Does the Motion Post-Processing Pipeline Handle Issues Constraint Guidance Cannot?</span>

The `MotionCorrection` module addresses artifacts that survive the diffusion process itself.

**Pipeline roles:**

- **Standardization:** Extracts uniform hip translations and local joint quaternions from all heterogeneous constraint types, giving the optimizer a single clean reference frame.
- **XZ-Translation Priority:** For `Root2D`, only horizontal (x, z) components are extracted; for 3D constraints, global positions are reconciled with the smoothed root path.

**Foot Contact Locking:**

The diffusion model predicts contact boolean flags `f ∈ {0,1}⁴` (left/right heel and toe). Even when the model correctly predicts contact, numerical denoising often causes "micro-sliding" or "skating." Post-processing:
- Reads the predicted contact flags
- **Mathematically freezes** the joint's 3D coordinates during `contact = 1` frames
- This makes the character look grounded and heavy rather than "floaty"


Motion Post-Processing. In practice, post-processing can be performed on the model outputs to improve the
generated motion. Simple foot locking and IK can clean up any undesirable foot skate using the foot contact
classification directly from the model output. It is also helpful to perform a short optimization on the output
motion to ensure it exactly hits the kinematic constraints, which is challenging for the model to achieve.

```python


def post_process_motion(
    local_rot_mats: torch.Tensor,
    root_positions: torch.Tensor,
    contacts: torch.Tensor,
    skeleton: SkeletonBase,
    constraint_lst: Optional[List] = None,
    contact_threshold: float = 0.5,
    root_margin: float = 0.04,
) -> Dict[str, torch.Tensor]:
    """Post-process generated motion to reduce foot skating and improve quality.

    Args:
        local_rot_mats: Local joint rotation matrices, shape (B, T, J, 3, 3)
        root_positions: Root joint positions, shape (B, T, 3)
        contacts: Foot contact labels, shape (B, T, num_contacts)
        skeleton: Skeleton instance
        constraint_lst: Optional list of constraints (or list of lists of constraints for batched inference)(FullBodyConstraintSet, etc.)
        contact_threshold: Threshold for foot contact detection
        root_margin: Margin for root position correction

    Returns:
        Dictionary with corrected motion data:
            - local_rot_mats: Corrected local rotation matrices (B, T, J, 3, 3)
            - root_positions: Corrected root positions (B, T, 3)
            - posed_joints: Corrected global joint positions (B, T, J, 3)
            - global_rot_mats: Corrected global rotation matrices (B, T, J, 3, 3)
    """
    # Ensure batch dimension
    assert local_rot_mats.dim() == 5, "local_rot_mats should be 5D, make sure to include the batch dimension"

    batch_size, num_frames, num_joints = local_rot_mats.shape[:3]

    def _build_constraint_masks_dict(constraints: List) -> Dict[str, torch.Tensor]:
        out = {
            key: torch.zeros(num_frames, dtype=torch.float32)
            for key in [
                "FullBody",
                "LeftFoot",
                "RightFoot",
                "LeftHand",
                "RightHand",
                "Root",
            ]
        }
        for constraint in constraints:
            frame_indices = constraint.frame_indices
            if isinstance(frame_indices, torch.Tensor):
                frame_indices = frame_indices[frame_indices < num_frames]
                if frame_indices.numel() == 0:
                    continue
            else:
                frame_indices = [idx for idx in frame_indices if idx < num_frames]
                if not frame_indices:
                    continue
            if constraint.name == "fullbody":
                out["FullBody"][frame_indices] = 1.0
            elif constraint.name == "left-foot":
                out["LeftFoot"][frame_indices] = 1.0
            elif constraint.name == "right-foot":
                out["RightFoot"][frame_indices] = 1.0
            elif constraint.name == "left-hand":
                out["LeftHand"][frame_indices] = 1.0
            elif constraint.name == "right-hand":
                out["RightHand"][frame_indices] = 1.0
            elif constraint.name == "root2d":
                out["Root"][frame_indices] = 1.0
        return out

    # Create constraint masks from constraint_lst (one dict per batch item when batched)
    batched_constraints = bool(constraint_lst) and isinstance(constraint_lst[0], list)
    if batched_constraints:
        constraint_masks_dict_lst = [_build_constraint_masks_dict(constraint_lst[b]) for b in range(batch_size)]
    else:
        constraint_masks_dict = (
            _build_constraint_masks_dict(constraint_lst)
            if constraint_lst
            else {
                key: torch.zeros(num_frames, dtype=torch.float32)
                for key in [
                    "FullBody",
                    "LeftFoot",
                    "RightFoot",
                    "LeftHand",
                    "RightHand",
                    "Root",
                ]
            }
        )

    # Create working rig
    above_ground_offset = 0.02 if isinstance(skeleton, (SOMASkeleton30, SOMASkeleton77)) else 0.007
    # larger offset for SOMA since model tends to generate lower to the ground
    working_rig = create_working_rig_from_skeleton(skeleton, above_ground_offset=above_ground_offset)
    has_double_ankle_joints = isinstance(skeleton, G1Skeleton34)

    # Prepare input tensors. The generated motion will be modified in place. Clone first.
    neutral_joints_pelvis_offset = skeleton.neutral_joints[0].cpu().clone()
    hip_translations_corrected = root_positions.cpu().clone()
    rotations_corrected = matrix_to_quaternion(local_rot_mats).cpu().clone()  # (B, T, J, 4)
    contacts = contacts.cpu()

    # Extract input motion (target keyframes) from constraints for each batch
    # For constrained keyframes, use the original motion from constraints
    # For non-constrained frames, zeros are used
    hip_translations_input = torch.zeros(batch_size, num_frames, 3)
    rotations_input = torch.zeros(batch_size, num_frames, num_joints, 4)
    rotations_input[..., 0] = 1.0  # Initialize as identity quaternions (w=1, x=y=z=0)

    if constraint_lst:
        for b in range(batch_size):
            # Get constraints for this batch item (if batched) or use the same list
            constraints_lst_el = (
                constraint_lst[b]
                if isinstance(
                    constraint_lst[0], list
                )  # when the constraint_list is in batch format, each item in a list is a constraintlist for one sample
                else constraint_lst  # single constraint list shared for all samples in the batch
            )
            hip_translations_input[b], rotations_input[b] = extract_input_motion_from_constraints(
                constraints_lst_el,
                skeleton,
                num_frames,
                num_joints,
            )

    # Call the motion correction for each batch (optional package)
    try:
        from motion_correction import motion_postprocess
    except ImportError as e:
        raise RuntimeError(
            "Motion correction is required for this postprocessing path but the "
            "motion_correction package is not installed. Install with: pip install -e ."
        ) from e
    for b in range(batch_size):
        masks_b = constraint_masks_dict_lst[b] if batched_constraints else constraint_masks_dict
        motion_postprocess.correct_motion(
            hip_translations_corrected[b : b + 1],
            rotations_corrected[b : b + 1],
            contacts[b : b + 1],
            hip_translations_input[b : b + 1],
            rotations_input[b : b + 1],
            masks_b,
            contact_threshold,
            root_margin,
            working_rig,
            has_double_ankle_joints,
        )

    local_rot_mats_corrected = quaternion_to_matrix(rotations_corrected)

    # Compute posed joints using FK
    device = local_rot_mats.device
    global_rot_mats, posed_joints, _ = fk(
        local_rot_mats_corrected.to(device),
        hip_translations_corrected.to(device),
        skeleton,
    )

    result = {
        "local_rot_mats": local_rot_mats_corrected.to(device),
        "root_positions": hip_translations_corrected.to(device),
        "posed_joints": posed_joints,
        "global_rot_mats": global_rot_mats,
    }

    return result
```
---

<!-- ## <span style="color:#76b900">Summary</span> -->

<!-- | Topic | Key Insight |
|---|---|
| Polymorphism | Each constraint type owns its own masking and validation logic — safety-critical and extensible |
| Dynamic Cropping | Enables arbitrary-duration timelines without padding waste or constraint truncation |
| Soft Guidance (CFG) | Negotiates physically impossible constraint conflicts rather than producing broken poses |
| Diffusion Steps | Sweet spot at ~100 steps; beyond this, over-smoothing and numerical drift degrade quality |
| Joint Conversion | FK validation ensures end-effector positions are preserved across 30↔77 joint conversion |
| Dual Representations | Global for user-facing constraint imposition; local for the body denoiser's learned prior |
| Frame-Rate Independence | `timestamp × model_fps` mapping makes UX time-domain; UI requires no changes across fps changes |
| Constraint Density | Sparse constraints let the learned natural-motion prior express itself; dense constraints suppress it |
| CFG Weight | Tunable knob between fluid naturalness (low) and geometric precision (high) |
| Post-Processing | Hard geometric guarantees (foot locking) that soft guidance alone cannot provide | -->
<!-- 
<div style="border: 2px solid #5e7f25; border-radius: 8px; background: #1a1a1a; overflow: hidden; width: 100%;">
  <div style="background: #5e7f25; padding: 3px 10px; font-size: 11px; font-weight: 700; color: #000; font-family: monospace; letter-spacing: 0.04em;">KEY INSIGHTS</div>
  <table style="width: 100%; border-collapse: collapse; font-family: monospace; font-size: 14px; color: #ffffff;">
    <thead>
      <tr style="border-bottom: 2px solid #76b900;">
        <th style="padding: 10px 14px; text-align: left; border-right: 1px solid #333; color: #76b900; white-space: nowrap;">Topic</th>
        <th style="padding: 10px 14px; text-align: left; color: #76b900;">Key Insight</th>
      </tr>
    </thead>
    <tbody>
      <tr style="border-bottom: 1px solid #333;">
        <td style="padding: 10px 14px; border-right: 1px solid #333; white-space: nowrap; color: #76b900;">Polymorphism</td>
        <td style="padding: 10px 14px;">Each constraint type owns its own masking and validation logic — safety-critical and extensible</td>
      </tr>
      <tr style="border-bottom: 1px solid #333; background: #222;">
        <td style="padding: 10px 14px; border-right: 1px solid #333; white-space: nowrap; color: #76b900;">Dynamic Cropping</td>
        <td style="padding: 10px 14px;">Enables arbitrary-duration timelines without padding waste or constraint truncation</td>
      </tr>
      <tr style="border-bottom: 1px solid #333;">
        <td style="padding: 10px 14px; border-right: 1px solid #333; white-space: nowrap; color: #76b900;">Soft Guidance (CFG)</td>
        <td style="padding: 10px 14px;">Negotiates physically impossible constraint conflicts rather than producing broken poses</td>
      </tr>
      <tr style="border-bottom: 1px solid #333; background: #222;">
        <td style="padding: 10px 14px; border-right: 1px solid #333; white-space: nowrap; color: #76b900;">Diffusion Steps</td>
        <td style="padding: 10px 14px;">Sweet spot at ~100 steps; beyond this, over-smoothing and numerical drift degrade quality</td>
      </tr>
      <tr style="border-bottom: 1px solid #333;">
        <td style="padding: 10px 14px; border-right: 1px solid #333; white-space: nowrap; color: #76b900;">Joint Conversion</td>
        <td style="padding: 10px 14px;">FK validation ensures end-effector positions are preserved across 30↔77 joint conversion</td>
      </tr>
      <tr style="border-bottom: 1px solid #333; background: #222;">
        <td style="padding: 10px 14px; border-right: 1px solid #333; white-space: nowrap; color: #76b900;">Dual Representations</td>
        <td style="padding: 10px 14px;">Global for user-facing constraint imposition; local for the body denoiser's learned prior</td>
      </tr>
      <tr style="border-bottom: 1px solid #333;">
        <td style="padding: 10px 14px; border-right: 1px solid #333; white-space: nowrap; color: #76b900;">Frame-Rate Independence</td>
        <td style="padding: 10px 14px;"><code style="background: #2a2a2a; padding: 2px 6px; border-radius: 4px; color: #ffcc00;">timestamp × model_fps</code> mapping makes UX time-domain; UI requires no changes across fps changes</td>
      </tr>
      <tr style="border-bottom: 1px solid #333; background: #222;">
        <td style="padding: 10px 14px; border-right: 1px solid #333; white-space: nowrap; color: #76b900;">Constraint Density</td>
        <td style="padding: 10px 14px;">Sparse constraints let the learned natural-motion prior express itself; dense constraints suppress it</td>
      </tr>
      <tr style="border-bottom: 1px solid #333;">
        <td style="padding: 10px 14px; border-right: 1px solid #333; white-space: nowrap; color: #76b900;">CFG Weight</td>
        <td style="padding: 10px 14px;">Tunable knob between fluid naturalness (low) and geometric precision (high)</td>
      </tr>
      <tr>
        <td style="padding: 10px 14px; border-right: 1px solid #333; white-space: nowrap; color: #76b900; background: #222;">Post-Processing</td>
        <td style="padding: 10px 14px; background: #222;">Hard geometric guarantees (foot locking) that soft guidance alone cannot provide</td>
      </tr>
    </tbody>
  </table>
</div> -->

---


# KIMODO INFERENCE PIPELINE
```python



class Kimodo(nn.Module):
    """Helper class for test time."""

    def __init__(
        self,
        denoiser: nn.Module,
        text_encoder: nn.Module,
        num_base_steps: int,
        device: Optional[Union[str, torch.device]] = None,
        cfg_type: Optional[str] = "separated",
    ):
        super().__init__()

        self.denoiser = denoiser.eval()

        if cfg_type is None:
            cfg_type = "nocfg"

        # Add Classifier-free guidance to the model if needed
        self.denoiser = ClassifierFreeGuidedModel(self.denoiser, cfg_type=cfg_type)

        self.motion_rep = denoiser.motion_rep
        self.skeleton = self.motion_rep.skeleton

        self.fps = denoiser.motion_rep.fps

        self.diffusion = Diffusion(num_base_steps=num_base_steps)
        self.sampler = DDIMSampler(self.diffusion)
        self.text_encoder = text_encoder

        self.device = device
        # for classifier-free guidance

        self.to(device)

    @property
    def output_skeleton(self):
        """Skeleton used for model output (somaskel77 for SOMA, else unchanged)."""
        if isinstance(self.skeleton, SOMASkeleton30):
            return self.skeleton.somaskel77
        return self.skeleton

    def train(self, mode: bool):
        self.denoiser.train(mode)
        return self

    def eval(self):
        self.denoiser.eval()
        return self
    def denoising_step(
        self,
        motion: torch.Tensor,
        pad_mask: torch.Tensor,
        text_feat: torch.Tensor,
        text_pad_mask: torch.Tensor,
        t: torch.Tensor,
        first_heading_angle: Optional[torch.Tensor],
        motion_mask: torch.Tensor,
        observed_motion: torch.Tensor,
        num_denoising_steps: torch.Tensor,
        cfg_weight: Union[float, Tuple[float, float]],
        guide_masks: Optional[Dict] = None,
        cfg_type: Optional[str] = None,
    ) -> torch.Tensor:
        """Single denoising step.

        Returns:
            torch.Tensor: [B, T, D] noisy motion input to t-1
        """
        # subsample timesteps
        #   NOTE: do this at every step due to ONNX export, i.e. num_samp_stepsmay change dynamically when
        #       running onnx version so need to account for that.
        num_denoising_steps = num_denoising_steps[0]
        use_timesteps, map_tensor = self.diffusion.space_timesteps(num_denoising_steps)
        self.diffusion.calc_diffusion_vars(use_timesteps)

        # first compute initial clean prediction from denoiser
        t_map = map_tensor[t]

        with torch.inference_mode():
            pred_clean = self.denoiser(
                cfg_weight,
                motion,
                pad_mask,
                text_feat,
                text_pad_mask,
                t_map,
                first_heading_angle,
                motion_mask,
                observed_motion,
                cfg_type=cfg_type,
            )

        # sampler computes next step noisy motion
        x_tm1 = self.sampler(use_timesteps, motion, pred_clean, t)
        return x_tm1

  
    def _multiprompt(
        self,
        prompts: list[str],
        num_frames: int | list[int],
        num_denoising_steps: int,
        constraint_lst: Optional[list] = [],
        cfg_weight: Optional[float] = [2.0, 2.0],
        num_samples: Optional[int] = None,
        cfg_type: Optional[str] = None,
        return_numpy: bool = False,
        first_heading_angle: Optional[torch.Tensor] = None,
        # for transitioning
        num_transition_frames: int = 5,
        share_transition: bool = True,
        percentage_transition_override=0.10,
        # for postprocess
        post_processing: bool = False,
        root_margin: float = 0.04,
        # progress bar
        progress_bar=tqdm,
    ) -> torch.Tensor:
        device = self.device

        bs = num_samples
        texts = sanitize_texts(prompts)

        if isinstance(num_frames, int):
            # same duration for all the segments
            num_frames = [num_frames for _ in range(num_samples)]

        tosqueeze = False
        if num_samples is None:
            num_samples = 1
            tosqueeze = True

        if constraint_lst is None:
            constraint_lst = []

        # Generate one chunck at a time
        current_frame = 0
        generated_motions = []

        for idx, (text, num_frame) in enumerate(zip(texts, num_frames)):
            texts_bs = [text for _ in range(num_samples)]

            lengths = torch.tensor(
                [num_frame for _ in range(num_samples)],
                device=device,
            )

            is_first_motion = not generated_motions

            observed_motion, motion_mask = None, None

            # filter the constraint_lst to only keep the relevent ones
            constraint_lst_base = [
                constraint.crop_move(current_frame, current_frame + num_frame) for constraint in constraint_lst
            ]  # this move temporally but not spatially

            observed_motion, motion_mask = self.motion_rep.create_conditions_from_constraints_batched(
                constraint_lst_base,
                lengths,
                to_normalize=False,  # don't normalize yet, it needs to be moved around
                device=device,
            )

            if not is_first_motion:
                prev_num_frame = num_frames[idx - 1]
                if share_transition:
                    # starting the transitioning earlier, to "share" the transition between A and B
                    # in any case, we still use "num_transition_frames" for conditioning
                    # we don't condition until the end of A
                    # we compute the number of frames of transition as a percentage of the last motion
                    nb_transition_frames = num_transition_frames + int(prev_num_frame * percentage_transition_override)
                else:
                    nb_transition_frames = num_transition_frames

                latest_motions = generated_motions.pop()
                # remove the transition part of A (will be put back afterward)
                generated_motions.append(latest_motions[:, :-nb_transition_frames])
                latest_frames = latest_motions[:, -nb_transition_frames:]
                # latest_frames[..., 2] += 0.5

                last_output = self.motion_rep.inverse(
                    latest_frames,
                    is_normalized=False,
                    return_numpy=False,
                )
                smooth_root_2d = last_output["smooth_root_pos"][..., [0, 2]]

                # add constraints at the begining to allow natural transitions
                constraint_lst_transition = []
                for batch_id in range(bs):
                    new_constraint = FullBodyConstraintSet(
                        self.skeleton,
                        torch.arange(num_transition_frames),
                        last_output["posed_joints"][batch_id, :num_transition_frames],
                        last_output["local_rot_mats"][batch_id, :num_transition_frames],
                        smooth_root_2d[batch_id, :num_transition_frames],
                    )

                    # new lists
                    constraint_lst_transition.append([new_constraint])

                transition_lengths = torch.tensor(
                    [nb_transition_frames for _ in range(num_samples)],
                    device=device,
                )

                observed_motion_transition, motion_mask_transition = (
                    self.motion_rep.create_conditions_from_constraints_batched(
                        constraint_lst_transition,
                        transition_lengths,
                        to_normalize=False,  # don't normalize yet
                        device=device,
                    )
                )

                # concatenate the obversed motion / motion mask
                observed_motion = torch.cat([observed_motion_transition, observed_motion], axis=1)
                motion_mask = torch.cat([motion_mask_transition, motion_mask], axis=1)

                # we need to move each observed motion in the batch to the new starting points
                last_smooth_root_2d = smooth_root_2d[:, 0]
                observed_motion = self.motion_rep.translate_2d(
                    observed_motion, -last_smooth_root_2d
                )  # equivalent to:  self.motion_rep.translate_2d_to_zero(observed_motion)

                # remove dummy values after moving
                observed_motion = observed_motion * motion_mask

                lengths = lengths + transition_lengths
                first_heading_angle = compute_heading_angle(last_output["posed_joints"], self.skeleton)[:, 0]
            else:
                if first_heading_angle is None:
                    # Start at 0 angle, but this will change afterward
                    first_heading_angle = torch.tensor([0.0] * bs, device=device)
                else:
                    first_heading_angle = torch.as_tensor(first_heading_angle, device=device)
                    if first_heading_angle.numel() == 1:
                        first_heading_angle = first_heading_angle.repeat(bs)

            observed_motion = self.motion_rep.normalize(observed_motion)

            max_frames = max(lengths)
            motion_pad_mask = length_to_mask(lengths)

            motion = self._generate(
                texts_bs,
                max_frames,
                num_denoising_steps=num_denoising_steps,
                pad_mask=motion_pad_mask,
                first_heading_angle=first_heading_angle,
                motion_mask=motion_mask,
                observed_motion=observed_motion,
                cfg_weight=cfg_weight,
                cfg_type=cfg_type,
            )

            motion = self.motion_rep.unnormalize(motion)

            if not is_first_motion:
                motion_with_transition = self.motion_rep.translate_2d(
                    motion,
                    last_smooth_root_2d,
                )

                motion = motion_with_transition[:, num_transition_frames:]
                transition_frames = motion_with_transition[:, :num_transition_frames]
                # for sharing = True, the new motion contains the very last of A

                # linearly combine the previously generated transitions with the newly generated ones
                # so that we linearly go from previous gen to new gen
                alpha = torch.linspace(1, 0, num_transition_frames, device=device)[:, None]
                new_transition_frames = (
                    latest_frames[:, :num_transition_frames] * alpha + (1 - alpha) * transition_frames
                )

                # add new transitions frames for A (merging with B predition of the history)
                # for share_transition == True, this remove (do not add back) a small part of the end of A
                # the small last part of A has been re-generated by B
                generated_motions.append(new_transition_frames)

                # motion[..., 2] += 0.5

            generated_motions.append(motion)
            current_frame += num_frame

        generated_motions = torch.cat(generated_motions, axis=1)  # temporal axis (b, t, d)

        if tosqueeze:
            generated_motions = generated_motions[0]

        output = self.motion_rep.inverse(
            generated_motions,
            is_normalized=False,
            return_numpy=False,
        )

        # Apply post-processing if requested
        if post_processing:
            corrected = post_process_motion(
                output["local_rot_mats"],
                output["root_positions"],
                output["foot_contacts"],
                self.skeleton,
                constraint_lst,
                root_margin=root_margin,
            )
            output.update(corrected)

        # Convert SOMA output to somaskel77 for external API
        if isinstance(self.skeleton, SOMASkeleton30):
            output = self.skeleton.output_to_SOMASkeleton77(output)

        # Convert to numpy if requested
        if return_numpy:
            output = to_numpy(output)
        return output

    def __call__(
        self,
        prompts: str | list[str],
        num_frames: int | list[int],
        num_denoising_steps: int,
        multi_prompt: bool = False,
        constraint_lst: Optional[list] = [],
        cfg_weight: Optional[float] = [2.0, 2.0],
        num_samples: Optional[int] = None,
        cfg_type: Optional[str] = None,
        return_numpy: bool = False,
        first_heading_angle: Optional[torch.Tensor] = None,
        # for transitioning
        num_transition_frames: int = 5,
        share_transition: bool = True,
        percentage_transition_override=0.10,
        # for postprocess
        post_processing: bool = False,
        root_margin: float = 0.04,
        # progress bar
        progress_bar=tqdm,
    ) -> dict:
        """Generate motion from text prompts and optional kinematic constraints.

        When a single prompt/num_frames pair is given, one motion is generated.
        Passing lists of prompts and/or num_frames produces a batch of
        independent motions. With ``multi_prompt=True``, the prompts are
        treated as sequential segments that are generated and stitched together
        with smooth transitions.

        Args:
            prompts: One or more text descriptions of the desired motion.
                A single string generates one sample; a list generates a batch
                (or sequential segments when ``multi_prompt=True``).
            num_frames: Duration of the generated motion in frames.  Can be a
                single int applied to every prompt or a per-prompt list.
            num_denoising_steps: Number of DDIM denoising steps.  More steps
                generally improve quality at the cost of speed.
            multi_prompt: If ``True``, treat ``prompts`` as an ordered sequence
                of segments and concatenate them with transitions.
            constraint_lst: Per-sample list of kinematic constraints (e.g.
                keyframe poses, end-effector targets, 2-D paths).  Pass an
                empty list for unconstrained generation.
            cfg_weight: Classifier-free guidance scale(s).  A two-element list
                ``[text_cfg, constraint_cfg]`` controls text and constraint
                guidance independently.
            num_samples: Number of samples to generate.
            cfg_type: Override the default CFG strategy set at init
                (e.g. ``"separated"``).
            return_numpy: If ``True``, convert all output tensors to numpy
                arrays.
            first_heading_angle: Initial body heading in radians.  Shape
                ``(B,)`` or scalar.  Defaults to ``0`` (facing +Z).
            num_transition_frames: Number of overlapping frames used to blend
                consecutive segments in multi-prompt mode.
            share_transition: If ``True``, transition frames are shared between
                adjacent segments rather than appended.
            percentage_transition_override: Fraction of each segment's length
                that may be overridden by the transition blend.
            post_processing: If ``True``, apply post-processing
                (foot-skate cleanup and constraint enforcement).
            root_margin: Horizontal margin (in meters) used by the post-processor
                to determine when to correct root motion. When root deviates more than
                margin from the constraint, the post-processor will correct it.
            progress_bar: Callable wrapping an iterable to display progress
                (default: ``tqdm``).  Pass a no-op to silence output.

        Returns:
            dict: A dictionary of motion tensors (or numpy arrays if
            ``return_numpy=True``) with the following keys:

            - ``local_rot_mats`` – Local joint rotations as rotation matrices.
            - ``global_rot_mats`` – Global joint rotations as rotation matrices.
            - ``posed_joints`` – Joint positions in world space.
            - ``root_positions`` – Root joint positions.
            - ``smooth_root_pos`` – Smoothed root trajectory.
            - ``foot_contacts`` – Boolean foot-contact labels [left heel, left toe, right heel, right toe].
            - ``global_root_heading`` – Root heading angle over time.
        """
        device = self.device

        if multi_prompt:
            # multi prompt generation
            return self._multiprompt(
                prompts,
                num_frames,
                num_denoising_steps,
                constraint_lst,
                cfg_weight,
                num_samples,
                cfg_type,
                return_numpy,
                first_heading_angle,
                num_transition_frames,
                share_transition,
                percentage_transition_override,
                post_processing,
                root_margin,
                progress_bar,
            )

        # Input checking
        tosqueeze = False
        if isinstance(prompts, list) and isinstance(num_frames, list):
            assert len(prompts) == len(num_frames), "The number of prompts should match the number of num_frames."
            num_samples = len(prompts)
        elif isinstance(prompts, list):
            num_samples = len(prompts)
            num_frames = [num_frames for _ in range(num_samples)]
        elif isinstance(num_frames, list):
            num_samples = len(num_frames)
            prompts = [prompts for _ in range(num_samples)]
        else:
            if num_samples is None:
                tosqueeze = True
                num_samples = 1
            prompts = [prompts for _ in range(num_samples)]
            num_frames = [num_frames for _ in range(num_samples)]

        bs = num_samples
        texts = sanitize_texts(prompts)

        lengths = torch.tensor(
            num_frames,
            device=device,
        )
        max_frames = max(lengths)
        motion_pad_mask = length_to_mask(lengths)

        if first_heading_angle is None:
            # Start at 0 angle
            first_heading_angle = torch.tensor([0.0] * bs, device=device)
        else:
            first_heading_angle = torch.as_tensor(first_heading_angle, device=device)
            if first_heading_angle.numel() == 1:
                first_heading_angle = first_heading_angle.repeat(bs)

        observed_motion, motion_mask = None, None
        if constraint_lst:
            observed_motion, motion_mask = self.motion_rep.create_conditions_from_constraints_batched(
                constraint_lst,
                lengths,
                to_normalize=True,
                device=device,
            )

        motion = self._generate(
            texts,
            max_frames,
            num_denoising_steps=num_denoising_steps,
            pad_mask=motion_pad_mask,
            first_heading_angle=first_heading_angle,
            motion_mask=motion_mask,
            observed_motion=observed_motion,
            cfg_weight=cfg_weight,
            cfg_type=cfg_type,
            progress_bar=progress_bar,
        )

        if tosqueeze:
            motion = motion[0]

        output = self.motion_rep.inverse(
            motion,
            is_normalized=True,
            return_numpy=False,  # Keep as tensor for potential post-processing
        )

        # Apply post-processing if requested
        if post_processing:
            corrected = post_process_motion(
                output["local_rot_mats"],
                output["root_positions"],
                output["foot_contacts"],
                self.skeleton,
                constraint_lst,
                root_margin=root_margin,
            )
            # key frame outputs / foot contacts are not changed
            output.update(corrected)

        # Convert SOMA output to somaskel77 for external API
        if isinstance(self.skeleton, SOMASkeleton30):
            output = self.skeleton.output_to_SOMASkeleton77(output)

        # Convert to numpy if requested
        if return_numpy:
            output = to_numpy(output)
        return output

    def _generate(
        self,
        texts: List[str],
        max_frames: int,
        num_denoising_steps: int,
        pad_mask: torch.Tensor,
        first_heading_angle: Optional[torch.Tensor],
        motion_mask: torch.Tensor,
        observed_motion: torch.Tensor,
        cfg_weight: Optional[float] = 2.0,
        text_feat: Optional[torch.Tensor] = None,
        text_pad_mask: Optional[torch.Tensor] = None,
        guide_masks: Optional[Dict] = None,
        cfg_type: Optional[str] = None,
        progress_bar=tqdm,
    ) -> torch.Tensor:
        """Sample full denoising loop.

        Args:
            texts (List[str]): batch of text prompts to use for sampling (if text_feat is not passed in)
        """

        device = self.device
        if text_feat is None:
            assert text_pad_mask is None
            log.info("Encoding text...")
            text_feat, text_length = self.text_encoder(texts)
            text_feat = text_feat.to(device)

            # handle empty string (set to zero)
            empty_text_mask = [len(text.strip()) == 0 for text in texts]
            text_feat[empty_text_mask] = 0

            # Create the pad mask for the text
            batch_size, maxlen = text_feat.shape[:2]
            tensor_text_length = torch.tensor(text_length, device=device)
            tensor_text_length[empty_text_mask] = 0
            text_pad_mask = torch.arange(maxlen, device=device).expand(batch_size, maxlen) < tensor_text_length[:, None]

        if motion_mask is not None:
            if motion_mask.dtype == torch.bool:
                motion_mask = 1 * motion_mask

        batch_size = text_feat.shape[0]

        # sample loop
        indices = list(range(num_denoising_steps))[::-1]
        shape = (batch_size, max_frames, self.motion_rep.motion_rep_dim)
        cur_mot = torch.randn(shape, device=self.device)
        num_denoising_steps = torch.tensor(
            [num_denoising_steps], device=self.device
        )  # this and t need to be tensor for onnx export
        # init diffusion with correct num steps before looping
        use_timesteps = self.diffusion.space_timesteps(num_denoising_steps[0])[0]
        self.diffusion.calc_diffusion_vars(use_timesteps)
        for i in progress_bar(indices):
            t = torch.tensor([i] * cur_mot.size(0), device=self.device)
            with torch.inference_mode():
                cur_mot = self.denoising_step(
                    cur_mot,
                    pad_mask,
                    text_feat,
                    text_pad_mask,
                    t,
                    first_heading_angle,
                    motion_mask,
                    observed_motion,
                    num_denoising_steps,
                    cfg_weight,
                    guide_masks=guide_masks,
                    cfg_type=cfg_type,
                )
        return cur_mot

```
<!-- *Report prepared as part of Data Science Internship — Project 4: KIMODO Technical Analysis* -->
> **Author:** Pawan Gupta (Data Science Intern)