import os

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


class SimpleAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, num_residual_blocks=1):
        super(SimpleAdapter, self).__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=8)
        self.conv = nn.Conv2d(in_dim * 64, out_dim, kernel_size=kernel_size, stride=stride, padding=0)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(out_dim) for _ in range(num_residual_blocks)]
        )

    def forward(self, x):
        target_dtype = next(self.conv.parameters()).dtype
        bs, c, f, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(bs * f, c, h, w)
        x_unshuffled = self.pixel_unshuffle(x.to(target_dtype))
        x_conv = self.conv(x_unshuffled)
        out = self.residual_blocks(x_conv)
        out = out.view(bs, f, out.size(1), out.size(2), out.size(3))
        out = out.permute(0, 2, 1, 3, 4)
        return out

    def process_camera_coordinates(
        self,
        direction,
        length,
        height,
        width,
        speed=1 / 54,
        origin=(0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        ori_h=None,
        ori_w=None,
        start_frame=0,
        end_frame=249,
        cam_from_worlds=None,
        intrinsics=None,
    ):
        if origin is None:
            origin = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
        coordinates = generate_camera_coordinates(
            direction, length, speed, origin, ori_h, ori_w,
            start_frame, end_frame, cam_from_worlds, intrinsics
        )
        plucker_embedding = process_pose_file(coordinates, width, height, ori_w, ori_h)
        return plucker_embedding


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        target_dtype = next(self.conv1.parameters()).dtype
        x = x.to(target_dtype)
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out


class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


class CameraSup(object):
    def __init__(self, extrinsic, intrinsic):
        self.fx = float(intrinsic[0, 0])
        self.fy = float(intrinsic[1, 1])
        self.cx = float(intrinsic[0, 2])
        self.cy = float(intrinsic[1, 2])
        w2c_mat_4x4 = extrinsic
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def get_relative_pose(cam_params):
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    cam_to_origin = 0
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses


def custom_meshgrid(*args):
    return torch.meshgrid(*args, indexing='ij')


def ray_condition(K, c2w, H, W, device):
    B = K.shape[0]
    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5
    fx, fy, cx, cy = K.chunk(4, dim=-1)
    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)
    rays_o = c2w[..., :3, 3]
    rays_o = rays_o[:, :, None].expand_as(rays_d)
    rays_dxo = torch.linalg.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)
    return plucker


def process_pose_file(cam_params, width=672, height=384, original_pose_width=1280, original_pose_height=720, device='cpu', return_poses=False):
    if return_poses:
        return cam_params
    cam_params = [Camera(cam_param) for cam_param in cam_params]
    if original_pose_width is None:
        original_pose_width = width
    if original_pose_height is None:
        original_pose_height = height
    sample_wh_ratio = width / height
    pose_wh_ratio = original_pose_width / original_pose_height
    if pose_wh_ratio > sample_wh_ratio:
        resized_ori_w = height * pose_wh_ratio
        for cam_param in cam_params:
            cam_param.fx = resized_ori_w * cam_param.fx / width
    else:
        resized_ori_h = width / pose_wh_ratio
        for cam_param in cam_params:
            cam_param.fy = resized_ori_h * cam_param.fy / height
    intrinsic = np.asarray([[cam_param.fx * width,
                             cam_param.fy * height,
                             cam_param.cx * width,
                             cam_param.cy * height]
                            for cam_param in cam_params], dtype=np.float32)
    K = torch.as_tensor(intrinsic)[None]
    c2ws = get_relative_pose(cam_params)
    c2ws = torch.as_tensor(c2ws)[None]
    plucker_embedding = ray_condition(K, c2ws, height, width, device=device)[0].permute(0, 3, 1, 2).contiguous()
    plucker_embedding = plucker_embedding[None]
    plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b f h w c")[0]
    return plucker_embedding


def normalize_t_np(new_cameras, points=None, max_norm=False):
    w2c = new_cameras
    t_gt = w2c[:, :3, -1].copy()
    t_gt = t_gt[1:, :]
    if max_norm:
        t_gt_scale = np.linalg.norm(t_gt, axis=-1)
        t_gt_scale = t_gt_scale.max()
        t_gt_scale = np.clip(t_gt_scale, 0.01, 100.0)
    else:
        t_gt_scale = np.linalg.norm(t_gt, axis=(0, 1))
        t_gt_scale = t_gt_scale / np.sqrt(len(t_gt) * len(t_gt[0]))
        t_gt_scale = t_gt_scale / 2.0
        t_gt_scale = np.clip(t_gt_scale, 0.01, 1000.0)
    w2c[:, :3, -1] = w2c[:, :3, -1] / t_gt_scale
    if points is not None:
        points = points / t_gt_scale
    return w2c, points


def normalize_intrinsic_inplace(K, ori_w=1280, ori_h=720):
    if ori_w is None:
        ori_w = 1280
    if ori_h is None:
        ori_h = 720
    K[:, 0, 0] = K[:, 0, 0] / ori_w
    K[:, 1, 1] = K[:, 1, 1] / ori_h
    K[:, 0, 2] = K[:, 0, 2] / ori_w
    K[:, 1, 2] = K[:, 1, 2] / ori_h
    K[0, 2] = 0.5
    K[1, 2] = 0.5
    return K


def generate_camera_coordinates(
    direction,
    length,
    speed=1 / 54,
    origin=(0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
    ori_h=None,
    ori_w=None,
    start_frame=0,
    end_frame=249,
    cam_from_worlds=None,
    intrinsics=None,
):
    if not isinstance(direction, list):
        direction = [direction]

    # Guard against None elements in direction list
    if direction and direction[0] is None and cam_from_worlds is None:
        # No valid camera data — return default coordinates (identity poses)
        return [list(origin)] * length

    if (isinstance(direction[0], str) and (direction[0].endswith('.npy') or os.path.exists(direction[0]))) or cam_from_worlds is not None:
        if cam_from_worlds is not None:
            intrins = intrinsics
            w2cs = cam_from_worlds
        else:
            intrin_path = direction[0]
            extrin_path = direction[1]
            intrins = np.load(intrin_path)
            w2cs = np.load(extrin_path)

        # Frame slicing: clamp to available range if start_frame/end_frame don't match length exactly
        n_cam_frames = intrins.shape[0] if hasattr(intrins, 'shape') else len(intrins)
        if end_frame - start_frame != length:
            # Adjust to match the expected number of frames
            end_frame = start_frame + length
        end_frame = min(end_frame, n_cam_frames)
        if end_frame <= start_frame:
            start_frame = 0
            end_frame = min(length, n_cam_frames)
        intrins = intrins[start_frame:end_frame]
        w2cs = w2cs[start_frame:end_frame]

        # Convert torch tensors to numpy before numpy-based operations
        if hasattr(intrins, 'detach'):
            intrins = intrins.detach().cpu().to(torch.float32).numpy()
        if hasattr(w2cs, 'detach'):
            w2cs = w2cs.detach().cpu().to(torch.float32).numpy()

        intrins = normalize_intrinsic_inplace(intrins, ori_w, ori_h)
        if w2cs[0].shape == (3, 4):
            w2cs_np = w2cs
            w2cs_4x4 = np.zeros((len(w2cs), 4, 4), dtype=np.float32)
            w2cs_4x4[:, :3, :4] = w2cs_np
            w2cs_4x4[:, 3, 3] = 1.0
        else:
            w2cs_4x4 = w2cs

        cam_params = [CameraSup(w2c, intrin) for w2c, intrin in zip(w2cs_4x4, intrins)]
        c2w_poses = get_relative_pose(cam_params)
        w2cs_4x4 = np.linalg.inv(c2w_poses)
        w2cs_4x4, _ = normalize_t_np(w2cs_4x4)

        coordinates = []
        n_available = len(w2cs_4x4)
        for i in range(length):
            idx = min(i, n_available - 1)  # repeat last pose if camera data is shorter than video
            coor = list(origin).copy()
            coor[7:] = w2cs_4x4[idx][:3, :].flatten()
            coor[1:5] = [intrins[idx][0, 0], intrins[idx][1, 1], intrins[idx][0, 2], intrins[idx][1, 2]]
            coordinates.append(coor)
        return coordinates

    # string-based camera direction mode
    direction = np.repeat(list(direction), [length // len(direction)] * len(direction)).tolist() + [direction[-1]] * (length % len(direction))
    coordinates = [list(origin)]
    for i in range(length - 1):
        coor = coordinates[-1].copy()
        if "Left" in direction:
            coor[9] += speed
        if "Right" in direction:
            coor[9] -= speed
        if "Up" in direction:
            coor[13] += speed
        if "Down" in direction:
            coor[13] -= speed
        if "In" in direction:
            coor[18] -= speed
        if "Out" in direction:
            coor[18] += speed
        w2c_mat = np.array(coor[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        position = w2c_mat_4x4[:3, 3]
        forward = w2c_mat_4x4[2, :3]
        right = w2c_mat_4x4[0, :3]
        if "w" in direction[i]:
            w2c_mat_4x4[:3, 3] = position - forward * speed
        if "s" in direction[i]:
            w2c_mat_4x4[:3, 3] = position + forward * speed
        if "a" in direction[i]:
            w2c_mat_4x4[:3, 3] = position + right * speed
        if "d" in direction[i]:
            w2c_mat_4x4[:3, 3] = position - right * speed
        angle = np.deg2rad(1)
        rotation = np.eye(3)
        if "8" in direction[i]:
            rotation = get_rotation_matrix("x", -angle)
        if "2" in direction[i]:
            rotation = get_rotation_matrix("x", angle)
        if "6" in direction[i]:
            rotation = get_rotation_matrix("y", -angle)
        if "4" in direction[i]:
            rotation = get_rotation_matrix("y", angle)
        w2c_mat_4x4[:3, :3] = rotation @ w2c_mat_4x4[:3, :3]
        coor[7:] = w2c_mat_4x4[:3, :].flatten()
        coordinates.append(coor)
    return coordinates


def get_rotation_matrix(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        return np.eye(3)


def compute_camera_latents(control_adapter, camera_control_direction, num_frames, height, width, ori_h, ori_w, start_frame, end_frame, cam_from_worlds=None, intrinsics=None, device='cpu', dtype=torch.bfloat16, camera_control_speed=None):
    # Fall back to height/width when ori_h/ori_w are not provided
    if ori_h is None:
        ori_h = height
    if ori_w is None:
        ori_w = width
    if camera_control_speed is None:
        camera_control_speed = 1 / 54
    camera_control_plucker_embedding = control_adapter.process_camera_coordinates(
        camera_control_direction, num_frames, height, width, camera_control_speed, None,
        ori_h, ori_w,start_frame, end_frame, cam_from_worlds, intrinsics
    )
    control_camera_video = camera_control_plucker_embedding[:num_frames].permute([3, 0, 1, 2]).unsqueeze(0)
    control_camera_latents = torch.concat(
        [
            torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
            control_camera_video[:, :, 1:]
        ], dim=2
    ).transpose(1, 2)
    b, f, c, h, w = control_camera_latents.shape
    control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
    control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
    control_camera_latents = control_camera_latents.to(device=device, dtype=dtype)
    return control_camera_latents

