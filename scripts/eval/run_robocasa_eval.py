#!/usr/bin/env python3
"""
GR00T-N1.6 RoboCasa OpenDrawer 评估.

关键：必须先创建 mujoco 环境（OSMesa），再 import torch_npu（避免 GL/NPU 冲突）.

用法:
    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
    python3 scripts/eval/run_robocasa_eval.py \
        --model_path /home/wangzhe/models/GR00T-N1.6-3B-FP16 \
        --task OpenDrawer --episodes 1 --max_steps 20
"""

import argparse, ctypes, math, os, sys, time, traceback, json
from pathlib import Path
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# Phase 1: 创建 mujoco 环境 (必须在 torch import 之前!)
# ═══════════════════════════════════════════════════════════════════════
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
sys.path.insert(0, "/home/wangzhe/robocasa")
sys.path.insert(0, "/home/wangzhe/robosuite")

def create_env(task_name):
    import robocasa
    from robocasa.utils.gym_utils import GrootRoboCasaEnv
    import gymnasium as gym
    env_name = f"robocasa_panda_omron/{task_name}_PandaOmron_Env"
    return gym.make(env_name, enable_render=True, split=None)

# ═══════════════════════════════════════════════════════════════════════
# Phase 2: 模型与推理 (在 torch import 之后)
# ═══════════════════════════════════════════════════════════════════════

def setup_model(model_path, om_path=None):
    """Load GR00T model with all NPU patches applied."""
    import torch
    import torch.nn.functional as F

    # SDPA monkey-patch
    _orig = F.scaled_dot_product_attention
    def _manual(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
                scale=None, enable_gqa=False):
        d = q.shape[-1]; s = scale if scale is not None else 1.0 / math.sqrt(d)
        sc = torch.matmul(q, k.transpose(-2, -1)) * s
        if attn_mask is not None:
            if attn_mask.dim() == 3: attn_mask = attn_mask.unsqueeze(1)
            elif attn_mask.dim() == 2: attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
            sc = sc + attn_mask
        return torch.matmul(F.softmax(sc, dim=-1), v)
    F.scaled_dot_product_attention = _manual

    sys.path.insert(0, '.')
    import gr00t.model  # noqa
    from transformers import AutoModel, AutoTokenizer

    m = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    m.eval(); m.to(device='npu:0', dtype=torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(
        '/home/wangzhe/Isaac-GR00T/gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2',
        trust_remote_code=True)

    om_runner = None
    if om_path:
        import acl
        acl_runner = _OMRunner(om_path)
        om_runner = acl_runner

    return m, tokenizer, om_runner


class _OMRunner:
    def __init__(self, path):
        import acl
        self.acl = acl
        r = acl.init()
        if r not in (0, 100002): raise RuntimeError(f"ACL init: {r}")
        acl.rt.set_device(0)
        self.mid, _ = acl.mdl.load_from_file(path)
        self.desc = acl.mdl.create_desc()
        acl.mdl.get_desc(self.desc, self.mid)
        self.in_sz = [acl.mdl.get_input_size_by_index(self.desc, i) for i in range(acl.mdl.get_num_inputs(self.desc))]
        self.out_sz = [acl.mdl.get_output_size_by_index(self.desc, i) for i in range(acl.mdl.get_num_outputs(self.desc))]

    def run(self, inputs):
        acl, H2D, D2H, M = self.acl, 1, 2, 2
        ids = acl.mdl.create_dataset()
        ibs = []
        for i, inp in enumerate(inputs):
            b, _ = acl.rt.malloc(self.in_sz[i], M)
            acl.rt.memcpy(b, self.in_sz[i], inp.ctypes.data_as(ctypes.c_void_p).value, inp.nbytes, H2D)
            acl.mdl.add_dataset_buffer(ids, acl.create_data_buffer(b, self.in_sz[i])); ibs.append(b)
        ods = acl.mdl.create_dataset(); obs = []
        for sz in self.out_sz:
            b, _ = acl.rt.malloc(sz, M)
            acl.mdl.add_dataset_buffer(ods, acl.create_data_buffer(b, sz)); obs.append(b)
        acl.mdl.execute(self.mid, ids, ods)
        out = np.empty(self.out_sz[0], dtype=np.uint8)
        acl.rt.memcpy(out.ctypes.data_as(ctypes.c_void_p).value, self.out_sz[0], obs[0], self.out_sz[0], D2H)
        for b in ibs + obs: acl.rt.free(b)
        acl.mdl.destroy_dataset(ids); acl.mdl.destroy_dataset(ods)
        return out


def encode_obs(model, tokenizer, image, text):
    """PIL Image + str → backbone_features [1, seq, 2048]"""
    import torch, torchvision.transforms as T
    t = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                   T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    pv = t(image).unsqueeze(0).to('npu:0', dtype=torch.float16)
    tok = tokenizer(text, return_tensors="pt")
    with torch.inference_mode():
        out = model.backbone.model(
            input_ids=tok['input_ids'].to('npu:0'),
            attention_mask=tok['attention_mask'].to('npu:0'),
            pixel_values=[pv], output_hidden_states=True)
    return out['hidden_states'][-1]


def predict_action(model, om, backbone_feat, state_np, emb_id=0):
    """state_np: [state_dim] → action_np: [action_dim]"""
    import torch
    from transformers.feature_extraction_utils import BatchFeature

    B, CTX, dev = 1, 256, 'npu:0'
    bf = backbone_feat[:, :CTX, :]
    if bf.shape[1] < CTX:
        pad = torch.zeros(B, CTX - bf.shape[1], bf.shape[2], dtype=bf.dtype, device=bf.device)
        bf = torch.cat([bf, pad], dim=1)
    bb_am = torch.ones(B, CTX, dtype=torch.bool, device=dev)
    im = torch.zeros(B, CTX, dtype=torch.bool, device=dev); im[:, :CTX//2] = True
    st = torch.from_numpy(state_np).float().to(dev).unsqueeze(0).unsqueeze(0).to(torch.float16)
    emb = torch.tensor([emb_id], dtype=torch.long, device=dev)

    bb = BatchFeature({'backbone_features': bf, 'backbone_attention_mask': bb_am, 'image_mask': im})
    ai = BatchFeature({'state': st, 'embodiment_id': emb})
    ah = model.action_head; c = model.config

    with torch.inference_mode():
        feats = ah._encode_features(bb, ai)
        vl, sf = feats.backbone_features, feats.state_features
        actions = torch.randn((B, c.action_horizon, c.max_action_dim), dtype=torch.float16, device=dev)
        dt = 1.0 / ah.num_inference_timesteps

        for t_idx in range(ah.num_inference_timesteps):
            td = int(t_idx / ah.num_inference_timesteps * ah.num_timestep_buckets)
            ts = torch.full((B,), fill_value=td, device=dev)
            af = ah.action_encoder(actions, ts, emb)
            if c.add_pos_embed:
                pids = torch.arange(af.shape[1], dtype=torch.long, device=dev)
                af = af + ah.position_embedding(pids).unsqueeze(0)
            sa = torch.cat((sf, af), dim=1)

            if om:
                m_out = _run_om(om, sa, vl, ts, im, bb_am)
            else:
                m_out = ah.model(hidden_states=sa, encoder_hidden_states=vl,
                                 timestep=ts, image_mask=im, backbone_attention_mask=bb_am)
            pv = ah.action_decoder(m_out, emb)[:, -c.action_horizon:]
            actions = actions + dt * pv

    return actions[0, 0, :].detach().cpu().float().numpy()


def _run_om(om, sa, vl, ts, im, bb_am):
    import torch, numpy as np
    img_am = im & bb_am; non_img_am = (~im) & bb_am
    imf = torch.where(img_am, torch.zeros_like(im, dtype=torch.float16), torch.full_like(im, -10000.0, dtype=torch.float16))
    nmf = torch.where(non_img_am, torch.zeros_like(im, dtype=torch.float16), torch.full_like(im, -10000.0, dtype=torch.float16))
    def _n(t, d=None): x = t.detach().cpu().numpy(); return x.astype(d) if d else x
    raw = om.run([_n(sa, np.float16), _n(vl, np.float16), np.array(ts.detach().cpu(), dtype=np.int32),
                  _n(imf, np.float16), _n(nmf, np.float16)])
    return torch.from_numpy(np.frombuffer(raw, dtype=np.float16).copy().reshape(1, 51, 1024)).to('npu:0', dtype=torch.float16)


def step_env(env, action_np):
    """Convert raw action numpy array to env action dict and step."""
    action = {
        "gripper_close": action_np[0:1].astype(np.float32),
        "end_effector_position": action_np[1:4].astype(np.float32),
        "end_effector_rotation": action_np[4:7].astype(np.float32),
        "base_motion": np.array([action_np[7], action_np[8], 0.0, 0.2], dtype=np.float32),
        "control_mode": np.array(0, dtype=np.int32),
    }
    return env.step(action)

# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/home/wangzhe/models/GR00T-N1.6-3B-FP16")
    parser.add_argument("--om_path", default="/home/wangzhe/Isaac-GR00T/atc_output/dit_310p3_fp16.om")
    parser.add_argument("--task", default="OpenDrawer")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--output_dir", default="/home/wangzhe/Isaac-GR00T/eval_results")
    parser.add_argument("--no_om", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    om_path = None if args.no_om else args.om_path

    print(f"\n{'='*70}\n  GR00T-N1.6 RoboCasa Eval\n  Task: {args.task} | Episodes: {args.episodes} | OM: {om_path is not None}\n{'='*70}")

    # Step 1: 创建环境 (在 torch 之前!)
    print("\n[Step 1] Creating environment...")
    env = create_env(args.task)
    obs, info = env.reset()
    img_keys = sorted([k for k in obs if k.startswith("video.")])
    state_keys = sorted([k for k in obs if k.startswith("state.")])
    lang_key = "annotation.human.task_description"
    lang = obs.get(lang_key, "Open the drawer.")
    print(f"  Env OK. Images: {len(img_keys)}, States: {len(state_keys)}, Lang: {lang}")

    # Step 2: 加载模型
    print("\n[Step 2] Loading model...")
    model, tokenizer, om = setup_model(args.model_path, om_path)
    print("  Model OK")

    # Step 3: 运行评估
    print(f"\n[Step 3] Running {args.episodes} episode(s)...")
    results = []

    for ep in range(args.episodes):
        obs, info = env.reset()
        success = False
        ep_start = time.perf_counter()
        step = 0

        for step in range(args.max_steps):
            t0 = time.perf_counter()

            # 提取观测
            img = obs[img_keys[0]]  # [H,W,3] uint8
            from PIL import Image
            pil_img = Image.fromarray(img)
            lang = obs.get(lang_key, "Open the drawer.")
            if isinstance(lang, (list, tuple)): lang = lang[0] if lang else ""
            state = np.concatenate([obs[k].flatten() for k in state_keys])

            # Backbone 编码
            try:
                bb_feat = encode_obs(model, tokenizer, pil_img, lang)
            except Exception as e:
                print(f"  [Ep{ep+1} Step{step}] Backbone error: {e}")
                traceback.print_exc()
                break

            # Action 预测
            try:
                action_np = predict_action(model, om, bb_feat, state)
            except Exception as e:
                print(f"  [Ep{ep+1} Step{step}] Action error: {e}")
                traceback.print_exc()
                break

            # 环境步进
            try:
                obs, reward, terminated, truncated, info = step_env(env, action_np)
            except Exception as e:
                print(f"  [Ep{ep+1} Step{step}] Env error: {e}")
                break

            elapsed = time.perf_counter() - t0
            if step % 5 == 0:
                print(f"  [Ep{ep+1} Step{step}] {elapsed:.1f}s reward={reward:.3f}")

            if terminated:
                success = info.get("success", False)
                break

        ep_elapsed = time.perf_counter() - ep_start
        status = "✓ SUCCESS" if success else "✗ FAIL"
        print(f"  [Ep{ep+1}] {status} | steps={step+1} | {ep_elapsed:.0f}s")
        results.append({"episode": ep+1, "success": success, "steps": step+1, "time_s": ep_elapsed})

    env.close()
    success_count = sum(1 for r in results if r["success"])
    rate = success_count / args.episodes * 100 if args.episodes else 0
    print(f"\n{'='*70}\n  Results: {success_count}/{args.episodes} ({rate:.0f}%)\n{'='*70}")

    out_file = output_dir / f"eval_{args.task}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as f:
        json.dump({"task": args.task, "success_rate": rate, "episodes": args.episodes,
                   "results": results}, f, indent=2, default=str)
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    main()
