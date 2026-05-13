#!/bin/bash
# 一键修复 NPU 适配补丁 (被 linter 破坏后恢复)
# 包含: __future__ import, flash_attn→sdpa, bf16→fp16, VideoInput, collator等
set -e

ROOT=/home/wangzhe/Isaac-GR00T
EAGLE=$ROOT/gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2
CACHE1=/home/wangzhe/.cache/huggingface/modules/transformers_modules/Eagle-Block2A-2B-v2
CACHE2=/home/wangzhe/.cache/huggingface/modules/transformers_modules/Eagle_hyphen_Block2A_hyphen_2B_hyphen_v2

echo "=== 1. 批量恢复 __future__ import ==="
cd $ROOT
python3 << 'PYEOF'
import os, re
for root, dirs, files in os.walk('gr00t'):
    dirs[:] = [d for d in dirs if d not in ('__pycache__',)]
    for f in files:
        if not f.endswith('.py'): continue
        path = os.path.join(root, f)
        with open(path) as fh: content = fh.read()
        if 'from __future__ import annotations' in content: continue
        if not re.search(r':\s*\S+\s*\|', content): continue
        lines = content.split('\n')
        # 在第一个非注释非空行前插入
        ins = 0
        for i, l in enumerate(lines):
            s = l.strip()
            if not s or s.startswith('#') or s.startswith('#!'): continue
            ins = i; break
        lines.insert(ins, 'from __future__ import annotations')
        with open(path, 'w') as fh: fh.write('\n'.join(lines))
        print(f'  __future__: {path}')
PYEOF

echo ""
echo "=== 2. eagle_backbone.py — flash_attn/bf16 断言 ==="
python3 << 'PYEOF'
f = 'gr00t/model/modules/eagle_backbone.py'
with open(f) as fh: c = fh.read()
# Remove flash_attn assertion
c = c.replace(
    '            if use_flash_attention: (\n                "nvidia/Eagle-Block2A-2B-v2 requires flash attention by default"\n            )',
    '            # NPU: flash_attn assertion removed')
# Remove bf16 assertion
c = c.replace(
    '            assert load_bf16, "nvidia/Eagle-Block2A-2B-v2 requires bfloat16 by default"',
    '            # NPU: bf16 assertion removed')
# attn_implementation → sdpa
c = c.replace('"flash_attention_2"', '"sdpa"')
with open(f, 'w') as fh: fh.write(c)
print('  Fixed eagle_backbone.py')
PYEOF

echo ""
echo "=== 3. Eagle modeling — flash_attn/initializer_range ==="
python3 << 'PYEOF'
for src in ['gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2/modeling_eagle3_vl.py']:
    with open(src) as fh: lines = fh.readlines()
    out, i = [], 0
    while i < len(lines):
        if ('assert (' in lines[i] and i+2 < len(lines) and
            'flash_attention_2' in lines[i+1] and 'must use flash_attention_2' in lines[i+2]):
            i += 3; continue
        out.append(lines[i]); i += 1
    with open(src, 'w') as fh: fh.writelines(out)
    # Fix initializer_range
    with open(src) as fh: c = fh.read()
    c = c.replace('self.config.initializer_range', 'getattr(self.config, "initializer_range", 0.02)')
    with open(src, 'w') as fh: fh.write(c)
    print(f'  Fixed {src}')
PYEOF

echo ""
echo "=== 4. Eagle processing — VideoInput/BASE_IMAGE_PROCESSOR ==="
python3 << 'PYEOF'
EAGLE = 'gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2'

# processing_eagle3_vl.py
f = f'{EAGLE}/processing_eagle3_vl.py'
with open(f) as fh: lines = fh.readlines()
out = []
for l in lines:
    if l.strip() in ('VideoInput,', '    VideoInput,'):
        continue
    out.append(l)
with open(f, 'w') as fh: fh.writelines(out)
print(f'  Fixed processing_eagle3_vl.py')

# image_processing_eagle3_vl_fast.py
f = f'{EAGLE}/image_processing_eagle3_vl_fast.py'
with open(f) as fh: c = fh.read()
# comment out incompatible imports
for bad in ['    VideoInput,', '    make_batched_videos,',
            '    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,',
            '    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,']:
    c = c.replace(bad, f'    # NPU compat\n    # {bad.strip()}')
with open(f, 'w') as fh: fh.write(c)
print(f'  Fixed image_processing_eagle3_vl_fast.py')
PYEOF

echo ""
echo "=== 5. gr00t_n1d6.py — collator try/except ==="
python3 << 'PYEOF'
f = 'gr00t/model/gr00t_n1d6/gr00t_n1d6.py'
with open(f) as fh: c = fh.read()
old = '''        from .processing_gr00t_n1d6 import Gr00tN1d6DataCollator

        self.collator = Gr00tN1d6DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )'''
new = '''        try:
            from .processing_gr00t_n1d6 import Gr00tN1d6DataCollator
            self.collator = Gr00tN1d6DataCollator(
                model_name=config.model_name,
                model_type=config.backbone_model_type,
                transformers_loading_kwargs=transformers_loading_kwargs,
            )
        except (ImportError, Exception) as e:
            import logging
            logging.warning(f"Could not create data collator (not needed for inference): {e}")
            self.collator = None'''
if old in c:
    c = c.replace(old, new)
    with open(f, 'w') as fh: fh.write(c)
    print('  Fixed gr00t_n1d6.py')
else:
    print('  Already fixed or pattern not found')
PYEOF

echo ""
echo "=== 6. model/__init__.py — n1d6 + n1d7 可选导入 ==="
cat > $ROOT/gr00t/model/__init__.py << 'EOF'
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
try:
    from .gr00t_n1d6.setup import Gr00tN1d6Pipeline  # noqa: F401
except Exception:
    Gr00tN1d6Pipeline = None
try:
    from .gr00t_n1d7.setup import Gr00tN1d7Pipeline  # noqa: F401
except Exception:
    Gr00tN1d7Pipeline = None
from .registry import MODEL_REGISTRY
EOF
echo "  Fixed model/__init__.py"

echo ""
echo "=== 7. 清除 HF 缓存 ==="
rm -rf $CACHE1 $CACHE2 2>/dev/null
echo "  Cache cleared"

echo ""
echo "=== 全部修复完成 ==="
echo "运行测试: source /usr/local/Ascend/ascend-toolkit/set_env.sh"
echo "          python3 -c 'import gr00t.model; from transformers import AutoModel; m=AutoModel.from_pretrained(...)'"
