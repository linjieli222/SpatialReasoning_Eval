"""Visualize visual CoT predictions vs ground truth.

Generates an HTML report showing for each sample:
- Input topdown image
- GT sideview image
- Question, choices, GT answer
- Model prediction (thinking + answer)
- Correctness

Usage:
    python visualize_vcot_predictions.py \
        --prediction_xlsx <path_to_prediction.xlsx> \
        --result_xlsx <path_to_result.xlsx> \
        --subset dh_midpoint \
        --output_dir <output_dir>

    # Optional: specify directory of generated images
    python visualize_vcot_predictions.py \
        --prediction_xlsx <path> --result_xlsx <path> \
        --subset dh_midpoint --output_dir <output_dir> \
        --gen_image_dir <dir_with_generated_images>
"""

import argparse
import base64
import html
import io
import os
import re

import pandas as pd
from PIL import Image


def pil_to_data_uri(pil_img, max_size=512, fmt="JPEG"):
    """Convert PIL Image to inline data URI, resized for HTML display."""
    pil_img = pil_img.copy()
    pil_img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt, quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def extract_thinking_and_answer(prediction):
    """Split prediction into thinking and final answer parts."""
    pred = str(prediction)
    # Remove Round_0: prefix
    pred = re.sub(r'^Round_\d+:\s*', '', pred)

    thinking = ""
    answer_text = ""
    # Extract <think>...</think> block
    think_match = re.search(r'<think>(.*?)</think>', pred, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer_text = pred[think_match.end():].strip()
    else:
        answer_text = pred

    return thinking, answer_text


def build_html(samples, title, output_path):
    """Build HTML report from sample list."""
    n_correct = sum(1 for s in samples if s['hit'])
    n_total = len(samples)
    acc = n_correct / n_total * 100 if n_total > 0 else 0

    rows_html = []
    for i, s in enumerate(samples):
        hit_class = "correct" if s['hit'] else "wrong"
        hit_label = "Correct" if s['hit'] else "Wrong"

        thinking, answer_text = extract_thinking_and_answer(s['prediction'])

        # Truncate long thinking for display
        thinking_display = thinking
        if len(thinking) > 1500:
            thinking_display = thinking[:1500] + "... [truncated]"

        # Choices text
        choices_html = ""
        for letter in ['A', 'B', 'C', 'D']:
            if letter in s and s[letter]:
                gt_marker = " <b>[GT]</b>" if s['gt_answer'] == letter else ""
                choices_html += f"<div>{letter}. {html.escape(str(s[letter]))}{gt_marker}</div>"

        # Images
        img_html = ""
        if s.get('topdown_uri'):
            img_html += f'<div class="img-container"><div class="img-label">Input (Top-down)</div><img src="{s["topdown_uri"]}"></div>'
        for ei, ego_uri in enumerate(s.get('ego_uris', [])):
            img_html += f'<div class="img-container"><div class="img-label">Ego {ei+1}</div><img src="{ego_uri}"></div>'
        if s.get('sideview_uri'):
            sv_label = "GT Sideview"
            if s.get('sideview_desc'):
                sv_label += f'<br><span style="font-weight:normal;font-size:10px">{html.escape(s["sideview_desc"][:80])}</span>'
            img_html += f'<div class="img-container"><div class="img-label">{sv_label}</div><img src="{s["sideview_uri"]}"></div>'
        if s.get('gen_image_uri'):
            img_html += f'<div class="img-container"><div class="img-label">Generated Sideview</div><img src="{s["gen_image_uri"]}"></div>'

        row = f"""
        <div class="sample {hit_class}">
            <div class="sample-header">
                <span class="sample-idx">#{i}</span>
                <span class="badge {hit_class}">{hit_label}</span>
                <span class="gt-answer">GT: {html.escape(str(s['gt_answer']))}</span>
            </div>
            <div class="sample-body">
                <div class="images-row">{img_html}</div>
                <div class="text-col">
                    <div class="question"><b>Question:</b> {html.escape(str(s['question']))[:300]}</div>
                    <div class="choices">{choices_html}</div>
                    <details class="thinking">
                        <summary>Model Thinking ({len(thinking)} chars)</summary>
                        <pre>{html.escape(thinking_display)}</pre>
                    </details>
                    <div class="model-answer"><b>Model Answer:</b> {html.escape(answer_text[:500])}</div>
                </div>
            </div>
        </div>
        """
        rows_html.append(row)

    page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
    body {{ font-family: -apple-system, sans-serif; margin: 20px; background: #f5f5f5; }}
    h1 {{ color: #333; }}
    .summary {{ background: #fff; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .sample {{ background: #fff; padding: 15px; margin-bottom: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 4px solid #ccc; }}
    .sample.correct {{ border-left-color: #4caf50; }}
    .sample.wrong {{ border-left-color: #f44336; }}
    .sample-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }}
    .sample-idx {{ font-weight: bold; color: #666; }}
    .badge {{ padding: 2px 8px; border-radius: 4px; color: #fff; font-size: 12px; font-weight: bold; }}
    .badge.correct {{ background: #4caf50; }}
    .badge.wrong {{ background: #f44336; }}
    .gt-answer {{ color: #666; font-size: 14px; }}
    .sample-body {{ display: flex; gap: 15px; }}
    .images-row {{ display: flex; gap: 10px; flex-shrink: 0; }}
    .img-container {{ text-align: center; }}
    .img-label {{ font-size: 11px; color: #666; margin-bottom: 4px; font-weight: bold; }}
    .img-container img {{ max-width: 256px; max-height: 256px; border-radius: 4px; border: 1px solid #ddd; }}
    .text-col {{ flex: 1; min-width: 0; }}
    .question {{ margin-bottom: 8px; font-size: 13px; }}
    .choices {{ margin-bottom: 8px; font-size: 13px; color: #555; }}
    .thinking {{ margin-bottom: 8px; }}
    .thinking summary {{ cursor: pointer; font-size: 12px; color: #888; }}
    .thinking pre {{ font-size: 11px; white-space: pre-wrap; word-break: break-word; background: #f9f9f9; padding: 8px; border-radius: 4px; max-height: 300px; overflow-y: auto; }}
    .model-answer {{ font-size: 13px; padding: 8px; background: #f0f7ff; border-radius: 4px; }}
    .filter-bar {{ margin-bottom: 15px; }}
    .filter-bar button {{ padding: 6px 14px; margin-right: 5px; border: 1px solid #ccc; border-radius: 4px; cursor: pointer; background: #fff; }}
    .filter-bar button.active {{ background: #333; color: #fff; border-color: #333; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="summary">
    <b>Accuracy:</b> {acc:.1f}% ({n_correct}/{n_total})
</div>
<div class="filter-bar">
    <button class="active" onclick="filterSamples('all', this)">All ({n_total})</button>
    <button onclick="filterSamples('correct', this)">Correct ({n_correct})</button>
    <button onclick="filterSamples('wrong', this)">Wrong ({n_total - n_correct})</button>
</div>
<div id="samples">
{''.join(rows_html)}
</div>
<script>
function filterSamples(filter, btn) {{
    document.querySelectorAll('.filter-bar button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.sample').forEach(s => {{
        if (filter === 'all') s.style.display = '';
        else if (filter === 'correct') s.style.display = s.classList.contains('correct') ? '' : 'none';
        else s.style.display = s.classList.contains('wrong') ? '' : 'none';
    }});
}}
</script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(page)
    print(f"Saved HTML report to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_xlsx', required=True)
    parser.add_argument('--result_xlsx', required=True)
    parser.add_argument('--subset', required=True, help='Dataset subset name, e.g. dh_midpoint')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--gen_image_dir', default=None, help='Directory with generated sideview images')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to visualize')
    parser.add_argument('--img_size', type=int, default=384, help='Max image display size')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load predictions and results
    pred_df = pd.read_excel(args.prediction_xlsx)
    result_df = pd.read_excel(args.result_xlsx)

    # Merge hit column
    if 'hit' in result_df.columns:
        pred_df['hit'] = result_df['hit'].values

    # Load dataset from HuggingFace
    print(f"Loading dataset subset: {args.subset}")
    from datasets import load_dataset
    hf_ds = load_dataset(
        'linjieli222/ai2thor_path_tracing_2point_tifa_filtered_eval',
        args.subset, split='val'
    )
    print(f"Dataset loaded: {len(hf_ds)} samples")

    # Collect generated images if available
    gen_images = {}
    if args.gen_image_dir and os.path.isdir(args.gen_image_dir):
        for fname in os.listdir(args.gen_image_dir):
            if fname.endswith(('.jpg', '.png')):
                # Try to extract sample index from filename
                m = re.search(r'sample(\d+)', fname)
                if m:
                    gen_images[int(m.group(1))] = os.path.join(args.gen_image_dir, fname)

    # Build sample list
    samples = []
    n = min(len(pred_df), len(hf_ds))
    if args.max_samples:
        n = min(n, args.max_samples)

    for i in range(n):
        row = pred_df.iloc[i]
        ex = hf_ds[i]

        sample = {
            'question': row['question'],
            'A': row.get('A', ''),
            'B': row.get('B', ''),
            'C': row.get('C', ''),
            'D': row.get('D', ''),
            'gt_answer': row['answer'],
            'prediction': row['prediction'],
            'hit': bool(row.get('hit', 0)),
        }

        # Convert dataset images to data URIs
        if ex.get('topdown_image'):
            sample['topdown_uri'] = pil_to_data_uri(ex['topdown_image'], max_size=args.img_size)

        # Ego images (td_ego_side, td_ego_dir, etc.)
        ego_imgs = ex.get('ego_images') or []
        sample['ego_uris'] = []
        for ego_img in ego_imgs:
            sample['ego_uris'].append(pil_to_data_uri(ego_img, max_size=args.img_size))

        if ex.get('sideview_image'):
            sample['sideview_uri'] = pil_to_data_uri(ex['sideview_image'], max_size=args.img_size)

        # Sideview description (GT)
        sample['sideview_desc'] = ex.get('sideview_desc', '')

        # Generated image if available
        if i in gen_images:
            gen_img = Image.open(gen_images[i])
            sample['gen_image_uri'] = pil_to_data_uri(gen_img, max_size=args.img_size)

        samples.append(sample)

    title = f"Visual CoT Predictions — {args.subset}"
    output_path = os.path.join(args.output_dir, f"viz_{args.subset}.html")
    build_html(samples, title, output_path)
    print(f"Visualized {len(samples)} samples")


if __name__ == '__main__':
    main()
