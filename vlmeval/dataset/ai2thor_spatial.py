import base64
import io
import json
import os
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from .image_mcq import ImageMCQDataset


import re


def _score_mcq_prediction(pred, gt, item):
    """Score a MCQ prediction against ground truth.
    Handles both letter-based (A/B) and text-based ground truth."""
    # Extract answer from <answer> tags if present
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', pred, re.IGNORECASE)
    if answer_match:
        pred = answer_match.group(1)

    pred_letter = pred.strip().upper().replace('.', '').replace(')', '')
    gt_clean = gt.strip().upper()

    # If gt is a letter, compare letters directly
    if gt_clean in ['A', 'B', 'C', 'D']:
        return 1 if pred_letter == gt_clean else 0

    # Otherwise convert pred letter to text and compare
    if pred_letter == 'A':
        pred_answer = str(item.get('A', ''))
    elif pred_letter == 'B':
        pred_answer = str(item.get('B', ''))
    elif pred_letter == 'C':
        pred_answer = str(item.get('C', ''))
    elif pred_letter == 'D':
        pred_answer = str(item.get('D', ''))
    else:
        pred_answer = pred
    hit = 1 if pred_answer.strip().upper() == gt_clean else 0
    if hit == 0 and gt_clean in pred_answer.strip().upper():
        hit = 1
    return hit


def pil_to_base64(pil_image, format='PNG'):
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


class AI2ThorPathTracing(ImageMCQDataset):
    """
    AI2Thor Path Tracing QA Dataset.
    Source: linjieli222/ai2thor-path-tracing-qa-v7
    Uses topdown_image, optionally with sideview image.
    4-choice MCQ (A/B/C/D).
    """

    TYPE = 'MCQ'

    def __init__(self, dataset='AI2ThorPathTracing', use_sideview=False, nsamples=None, **kwargs):
        self.use_sideview = use_sideview
        self.nsamples = nsamples
        super().__init__(dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['AI2ThorPathTracing', 'AI2ThorPathTracing_sideview']

    def load_data(self, dataset):
        from datasets import load_dataset

        # Load from HuggingFace
        hf_ds = load_dataset('linjieli222/ai2thor-path-tracing-qa-v7', split='train')

        records = []
        for idx, ex in enumerate(hf_ds):
            # Early exit if we have enough samples
            if self.nsamples is not None and len(records) >= self.nsamples:
                break

            # Convert topdown_image to base64
            img_list = [pil_to_base64(ex['topdown_image'])]

            question = ex['question']

            if self.use_sideview:
                sv_img = ex.get('sideview_image') or ex.get('sideview_images')
                if sv_img is not None:
                    if isinstance(sv_img, list):
                        for img_item in sv_img:
                            img_list.append(pil_to_base64(img_item))
                    else:
                        img_list.append(pil_to_base64(sv_img))

            img = img_list if len(img_list) > 1 else img_list[0]

            # Map choices list to A/B/C/D
            choices = ex['choices']

            records.append({
                'index': len(records),
                'image': img,
                'question': question,
                'A': choices[0] if len(choices) > 0 else '',
                'B': choices[1] if len(choices) > 1 else '',
                'C': choices[2] if len(choices) > 2 else '',
                'D': choices[3] if len(choices) > 3 else '',
                'answer': ex['answer'],
            })

        return pd.DataFrame(records)

    def evaluate(self, eval_file, **judge_kwargs):
        import numpy as np
        from ..smp import load, dump

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', f'_result.{suffix}')

        data = load(eval_file)

        # Score each sample using rule-based matching
        if 'hit' not in data.columns:
            for i in range(len(data)):
                item = data.iloc[i]
                pred = str(item.get('prediction', ''))
                gt = str(item.get('answer', ''))
                hit = _score_mcq_prediction(pred, gt, item)
                data.loc[data.index[i], 'hit'] = hit
            dump(data, result_file)

        overall_acc = data['hit'].mean() * 100
        res = {'Category': ['Overall'], 'Accuracy': [overall_acc], 'Count': [len(data)]}
        return pd.DataFrame(res)


class AI2ThorPerspective_NoArrow(ImageMCQDataset):
    """
    AI2Thor Perspective QA Dataset (No Arrow version).
    Source: weikaih/ai2thor-perspective-v3-278-human-filter-with-obj-name
    Uses marked_image_no_arrow and question_no_arrow.
    2-choice MCQ (A/B).

    Categories (6 splits):
    - distance_change_closer
    - distance_change_further
    - relative_position_left_left
    - relative_position_left_right
    - relative_position_right_left
    - relative_position_right_right

    Overall is computed as unweighted average of 6 category accuracies.
    """

    TYPE = 'MCQ'

    def __init__(self, dataset='AI2ThorPerspective_NoArrow', nsamples=None, **kwargs):
        self.nsamples = nsamples
        super().__init__(dataset=dataset, **kwargs)

    def evaluate(self, eval_file, **judge_kwargs):
        """Custom evaluate that computes Overall as unweighted average of category accuracies."""
        from collections import defaultdict
        import numpy as np
        from ..smp import load, dump
        from .utils import build_judge

        # First run the parent evaluation to get per-sample results
        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', f'_result.{suffix}')

        data = load(eval_file)

        # Score each sample using rule-based matching
        if 'hit' not in data.columns:
            for i in range(len(data)):
                item = data.iloc[i]
                pred = str(item.get('prediction', ''))
                gt = str(item.get('answer', ''))
                hit = _score_mcq_prediction(pred, gt, item)
                data.loc[data.index[i], 'hit'] = hit
            dump(data, result_file)

        # Compute per-category accuracy
        category_acc = {}
        categories = data['category'].unique()

        for cat in categories:
            cat_data = data[data['category'] == cat]
            acc = cat_data['hit'].mean() * 100  # Convert to percentage
            category_acc[cat] = acc

        # Compute Overall as unweighted average of category accuracies
        overall_acc = np.mean(list(category_acc.values()))

        # Build result DataFrame
        res = {'Category': ['Overall'], 'Accuracy': [overall_acc], 'Count': [len(data)]}
        for cat in sorted(categories):
            cat_count = len(data[data['category'] == cat])
            res['Category'].append(cat)
            res['Accuracy'].append(category_acc[cat])
            res['Count'].append(cat_count)

        res_df = pd.DataFrame(res)

        # Save results
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(res_df, score_file)

        return res_df

    @classmethod
    def supported_datasets(cls):
        return ['AI2ThorPerspective_NoArrow']

    def load_data(self, dataset):
        from datasets import load_dataset

        # Load all splits from HuggingFace
        hf_ds = load_dataset('weikaih/ai2thor-perspective-v3-278-human-filter-with-obj-name')

        records = []
        done = False
        for split_name in hf_ds.keys():
            if done:
                break
            for ex in hf_ds[split_name]:
                # Early exit if we have enough samples
                if self.nsamples is not None and len(records) >= self.nsamples:
                    done = True
                    break

                # Convert marked_image_no_arrow to base64
                img_b64 = pil_to_base64(ex['marked_image_no_arrow'])

                # Map answer_choices to A/B (answer_choices is a JSON string)
                choices_raw = ex['answer_choices']
                if isinstance(choices_raw, str):
                    choices = json.loads(choices_raw)
                else:
                    choices = choices_raw

                records.append({
                    'index': len(records),
                    'image': img_b64,
                    'question': ex['question_no_arrow'],
                    'A': choices[0] if len(choices) > 0 else '',
                    'B': choices[1] if len(choices) > 1 else '',
                    'answer': ex['answer'],
                    'category': split_name,  # Use split name for 6-category breakdown
                })

        return pd.DataFrame(records)


class AI2ThorPerspective_Arrow(ImageMCQDataset):
    """
    AI2Thor Perspective QA Dataset (With Arrow version).
    Source: weikaih/ai2thor-perspective-v3-278-human-filter-with-obj-name
    Uses marked_image_with_arrow and question_with_arrow.
    2-choice MCQ (A/B).

    Categories (6 splits):
    - distance_change_closer
    - distance_change_further
    - relative_position_left_left
    - relative_position_left_right
    - relative_position_right_left
    - relative_position_right_right

    Overall is computed as unweighted average of 6 category accuracies.
    """

    TYPE = 'MCQ'

    @classmethod
    def supported_datasets(cls):
        return ['AI2ThorPerspective_Arrow']

    def evaluate(self, eval_file, **judge_kwargs):
        """Custom evaluate that computes Overall as unweighted average of category accuracies."""
        import numpy as np
        from ..smp import load, dump

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', f'_result.{suffix}')

        data = load(eval_file)

        # Score each sample using rule-based matching
        if 'hit' not in data.columns:
            for i in range(len(data)):
                item = data.iloc[i]
                pred = str(item.get('prediction', ''))
                gt = str(item.get('answer', ''))
                hit = _score_mcq_prediction(pred, gt, item)
                data.loc[data.index[i], 'hit'] = hit
            dump(data, result_file)

        # Compute per-category accuracy
        category_acc = {}
        categories = data['category'].unique()

        for cat in categories:
            cat_data = data[data['category'] == cat]
            acc = cat_data['hit'].mean() * 100
            category_acc[cat] = acc

        # Compute Overall as unweighted average of category accuracies
        overall_acc = np.mean(list(category_acc.values()))

        # Build result DataFrame
        res = {'Category': ['Overall'], 'Accuracy': [overall_acc], 'Count': [len(data)]}
        for cat in sorted(categories):
            cat_count = len(data[data['category'] == cat])
            res['Category'].append(cat)
            res['Accuracy'].append(category_acc[cat])
            res['Count'].append(cat_count)

        res_df = pd.DataFrame(res)
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(res_df, score_file)

        return res_df

    def load_data(self, dataset):
        from datasets import load_dataset

        # Load all splits from HuggingFace
        hf_ds = load_dataset('weikaih/ai2thor-perspective-v3-278-human-filter-with-obj-name')

        records = []
        global_idx = 0
        for split_name in hf_ds.keys():
            for ex in hf_ds[split_name]:
                # Convert marked_image_with_arrow to base64
                img_b64 = pil_to_base64(ex['marked_image_with_arrow'])

                # Map answer_choices to A/B (answer_choices is a JSON string)
                choices_raw = ex['answer_choices']
                if isinstance(choices_raw, str):
                    choices = json.loads(choices_raw)
                else:
                    choices = choices_raw

                records.append({
                    'index': global_idx,
                    'image': img_b64,
                    'question': ex['question_with_arrow'],
                    'A': choices[0] if len(choices) > 0 else '',
                    'B': choices[1] if len(choices) > 1 else '',
                    'answer': ex['answer'],
                    'category': split_name,  # Use split name for 6-category breakdown
                })
                global_idx += 1

        return pd.DataFrame(records)


class SideviewOverfit(ImageMCQDataset):
    """
    Sideview Overfit Test Dataset.
    Loads first N samples from sideview training parquet data to test if model can reproduce training samples.
    Uses original training instruction directly (no MCQ format).
    ThinkMorph.py has special handling to use the question field as-is.
    """

    TYPE = 'VQA'  # Not MCQ, but we still inherit from ImageMCQDataset for convenience

    # Path to sideview training parquet data
    PARQUET_DIR = '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/bagel_example/editing/ai2thor-path-tracing-train-sideview-only'

    def __init__(self, dataset='SideviewOverfit', nsamples=10, **kwargs):
        self.nsamples = nsamples
        super().__init__(dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['SideviewOverfit']

    def load_data(self, dataset):
        # Find parquet files
        parquet_files = sorted([f for f in os.listdir(self.PARQUET_DIR) if f.endswith('.parquet')])
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.PARQUET_DIR}")

        records = []
        samples_loaded = 0

        for pq_file in parquet_files:
            if samples_loaded >= self.nsamples:
                break

            pq_path = os.path.join(self.PARQUET_DIR, pq_file)
            table = pq.read_table(pq_path)
            df_pq = table.to_pandas()

            for idx, row in df_pq.iterrows():
                if samples_loaded >= self.nsamples:
                    break

                # Extract input image (first image in image_list)
                image_list = row['image_list']
                if len(image_list) == 0:
                    continue

                # Convert bytes to PIL Image then to base64
                input_img_bytes = image_list[0]
                pil_img = Image.open(io.BytesIO(input_img_bytes))
                img_b64 = pil_to_base64(pil_img)

                # Use full training instruction as question (includes system prompt)
                instruction = row['instruction_list'][0] if len(row['instruction_list']) > 0 else ""

                # Extract expected output text for reference
                output_texts = row['output_text_list']
                expected_text = ' '.join(output_texts) if len(output_texts) > 0 else ""

                # No MCQ options - ThinkMorph.py will use instruction directly
                records.append({
                    'index': samples_loaded,
                    'image': img_b64,
                    'question': instruction,  # Full training instruction
                    'answer': expected_text,  # Expected output for reference
                })
                samples_loaded += 1

        return pd.DataFrame(records)


class AI2ThorPathTracing2Point(ImageMCQDataset):
    """
    AI2Thor Path Tracing 2-Point Dataset.
    Source: linjieli222/ai2thor_path_tracing_2point_tifa_filtered_eval
    Supports 8 subsets (view configurations) and optional sideview image.
    4-choice MCQ (A/B/C/D).
    """

    TYPE = 'MCQ'
    HF_REPO = 'linjieli222/ai2thor_path_tracing_2point_tifa_filtered_eval'
    SUBSETS = [
        'dh_midpoint', 'td_ego_dir', 'td_ego_dir_arrow', 'td_ego_side',
        'td_ego_side_arrow', 'td_midpoint', 'td_path', 'td_path_arrow',
    ]

    def __init__(self, dataset='AI2ThorPathTracing2Point', subset='td_path',
                 use_sideview=False, vcot_prefill=False, nsamples=None, **kwargs):
        self.subset = subset
        self.use_sideview = use_sideview
        self.vcot_prefill = vcot_prefill
        self.nsamples = nsamples
        super().__init__(dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return [
            'AI2ThorPathTracing2Point',
            'AI2ThorPT2P_dh_midpoint', 'AI2ThorPT2P_dh_midpoint_sideview',
            'AI2ThorPT2P_td_ego_dir', 'AI2ThorPT2P_td_ego_dir_sideview',
            'AI2ThorPT2P_td_ego_dir_arrow', 'AI2ThorPT2P_td_ego_dir_arrow_sideview',
            'AI2ThorPT2P_td_ego_side', 'AI2ThorPT2P_td_ego_side_sideview',
            'AI2ThorPT2P_td_ego_side_arrow', 'AI2ThorPT2P_td_ego_side_arrow_sideview',
            'AI2ThorPT2P_td_midpoint', 'AI2ThorPT2P_td_midpoint_sideview',
            'AI2ThorPT2P_td_path', 'AI2ThorPT2P_td_path_sideview',
            'AI2ThorPT2P_td_path_arrow', 'AI2ThorPT2P_td_path_arrow_sideview',
            'AI2ThorPT2P_td_path_vcot_prefill',
            'AI2ThorPT2P_td_ego_dir_vcot_prefill',
        ]

    def build_prompt(self, line):
        """Build prompt with images interleaved at <image_N> positions in the question text."""
        import re
        import string

        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            from ..smp import toliststr
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        # Ensure tgt_path is a list
        if not isinstance(tgt_path, list):
            tgt_path = [tgt_path]

        question = line['question']

        # Build options text
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = ''
        if len(options):
            options_prompt = 'Options:\n'
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'
            options_prompt += 'Please select the correct answer from the options above. \n'

        # Split question at <image_N> tags and interleave with actual images
        # Images are ordered: image_1 → tgt_path[0], image_2 → tgt_path[1], etc.
        parts = re.split(r'(<image_\d+>)', question)

        msgs = []
        for part in parts:
            m = re.match(r'<image_(\d+)>', part)
            if m:
                img_idx = int(m.group(1)) - 1  # <image_1> → index 0
                if img_idx < len(tgt_path):
                    msgs.append(dict(type='image', value=tgt_path[img_idx]))
            else:
                text = part.strip()
                if text:
                    msgs.append(dict(type='text', value=text))

        # Append options at the end
        if options_prompt:
            msgs.append(dict(type='text', value=options_prompt))

        return msgs

    @staticmethod
    def _extract_answer(pred):
        """Extract answer letter from model prediction, handling <answer> tags."""
        import re
        pred = str(pred).strip()
        # Try to extract from <answer>...</answer> tags
        match = re.search(r'<answer>\s*([A-Da-d])\s*</answer>', pred)
        if match:
            return match.group(1).upper()
        # Try to extract from <answer>...(A)...</answer> or answer tag with text
        match = re.search(r'<answer>\s*\(?([A-Da-d])\)?\s*', pred)
        if match:
            return match.group(1).upper()
        # Fallback: exact single letter
        if pred.upper() in ('A', 'B', 'C', 'D'):
            return pred.upper()
        # Fallback: first letter match in the prediction
        match = re.search(r'\b([A-D])\b', pred)
        if match:
            return match.group(1)
        return pred.strip().upper()

    def evaluate(self, eval_file, **judge_kwargs):
        """Custom evaluate that computes accuracy for this subset."""
        import numpy as np
        from ..smp import load, dump

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', f'_result.{suffix}')

        data = load(eval_file)

        # Score each sample
        if 'hit' not in data.columns:
            for i in range(len(data)):
                item = data.iloc[i]
                pred = self._extract_answer(item.get('prediction', ''))
                gt = str(item.get('answer', '')).strip().upper()
                hit = 1 if pred == gt else 0
                data.loc[data.index[i], 'hit'] = hit
            dump(data, result_file)

        overall_acc = data['hit'].mean() * 100

        res = {
            'Category': ['Overall'],
            'Accuracy': [overall_acc],
            'Count': [len(data)],
        }

        # Per-category breakdown if category column exists
        if 'category' in data.columns:
            categories = data['category'].unique()
            for cat in sorted(categories):
                cat_data = data[data['category'] == cat]
                cat_acc = cat_data['hit'].mean() * 100
                res['Category'].append(cat)
                res['Accuracy'].append(cat_acc)
                res['Count'].append(len(cat_data))

        res_df = pd.DataFrame(res)
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(res_df, score_file)

        return res_df

    @staticmethod
    def _sideview_desc_to_second_person(desc):
        """Convert sideview_desc from first person to second person."""
        import re
        desc = re.sub(r'\bI\b', 'You', desc)
        desc = re.sub(r'\bmy\b', 'your', desc)
        desc = re.sub(r'\bMy\b', 'Your', desc)
        desc = re.sub(r'\bme\b', 'you', desc)
        return desc

    def load_data(self, dataset):
        from datasets import load_dataset

        hf_ds = load_dataset(self.HF_REPO, self.subset, split='val')

        records = []
        for idx, ex in enumerate(hf_ds):
            if self.nsamples is not None and len(records) >= self.nsamples:
                break

            # Build image list: topdown + ego images (if any) + sideview (if enabled)
            img_list = [pil_to_base64(ex['topdown_image'])]
            for ego_img in (ex.get('ego_images') or []):
                img_list.append(pil_to_base64(ego_img))

            question = ex['question']

            if self.use_sideview:
                sv_img = ex.get('sideview_image') or ex.get('sideview_images')
                sideview_b64 = pil_to_base64(sv_img)
                img_list.append(sideview_b64)
                # Append sideview description (converted to second person)
                sv_desc = ex.get('sideview_desc', '')
                if sv_desc:
                    sv_desc = self._sideview_desc_to_second_person(sv_desc)
                    question = question + ' ' + sv_desc

            # Single image → string, multiple images → list
            img = img_list if len(img_list) > 1 else img_list[0]

            # Map choices list to A/B/C/D
            choices = ex['choices']

            record = {
                'index': len(records),
                'image': img,
                'question': question,
                'A': choices[0] if len(choices) > 0 else '',
                'B': choices[1] if len(choices) > 1 else '',
                'C': choices[2] if len(choices) > 2 else '',
                'D': choices[3] if len(choices) > 3 else '',
                'answer': ex['answer'],
            }

            # For vcot_prefill, store GT sideview data as extra columns
            if self.vcot_prefill:
                sv_img = ex.get('sideview_image') or ex.get('sideview_images')
                record['gt_sideview_image'] = pil_to_base64(sv_img)
                # Clean sideview_desc to match training format
                sv_desc_raw = ex.get('sideview_desc', '')
                sv_desc_clean = sv_desc_raw.replace(' <image_2>', '').replace('<image_2>', '')
                sv_desc_clean = sv_desc_clean.replace(' ..', '.').replace(' .', '.')
                sv_desc_clean = sv_desc_clean.replace(':.', ':').replace('  ', ' ').strip()
                record['gt_sideview_desc'] = sv_desc_clean

            records.append(record)

        return pd.DataFrame(records)


class AI2ThorPathTracing2PointV2(AI2ThorPathTracing2Point):
    """
    AI2Thor Path Tracing 2-Point Dataset — Debiased Val V2.
    Source: linjieli222/ai2thor_path_tracing_2point_tifa_filtered_val_v3
    Same format as AI2ThorPathTracing2Point but with debiased validation samples.
    """

    HF_REPO = 'linjieli222/ai2thor_path_tracing_2point_tifa_filtered_val_v3'

    def __init__(self, dataset='AI2ThorPT2PV2', subset='td_path',
                 use_sideview=False, vcot_prefill=False, nsamples=None, **kwargs):
        super().__init__(dataset=dataset, subset=subset,
                         use_sideview=use_sideview, vcot_prefill=vcot_prefill,
                         nsamples=nsamples, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return [
            'AI2ThorPT2PV2',
            'AI2ThorPT2PV2_dh_midpoint', 'AI2ThorPT2PV2_dh_midpoint_sideview',
            'AI2ThorPT2PV2_td_ego_dir', 'AI2ThorPT2PV2_td_ego_dir_sideview',
            'AI2ThorPT2PV2_td_ego_dir_arrow', 'AI2ThorPT2PV2_td_ego_dir_arrow_sideview',
            'AI2ThorPT2PV2_td_ego_side', 'AI2ThorPT2PV2_td_ego_side_sideview',
            'AI2ThorPT2PV2_td_ego_side_arrow', 'AI2ThorPT2PV2_td_ego_side_arrow_sideview',
            'AI2ThorPT2PV2_td_midpoint', 'AI2ThorPT2PV2_td_midpoint_sideview',
            'AI2ThorPT2PV2_td_path', 'AI2ThorPT2PV2_td_path_sideview',
            'AI2ThorPT2PV2_td_path_arrow', 'AI2ThorPT2PV2_td_path_arrow_sideview',
            'AI2ThorPT2PV2_td_ego_dir_vcot_prefill',
            'AI2ThorPT2PV2_td_path_vcot_prefill',
            'AI2ThorPT2PV2_td_path_arrow_vcot_prefill',
        ]

    def load_data(self, dataset):
        from datasets import load_dataset

        # Try 'val' split first, fall back to 'train'
        try:
            hf_ds = load_dataset(self.HF_REPO, self.subset, split='val')
        except ValueError:
            hf_ds = load_dataset(self.HF_REPO, self.subset, split='train')

        records = []
        for idx, ex in enumerate(hf_ds):
            if self.nsamples is not None and len(records) >= self.nsamples:
                break

            img_list = [pil_to_base64(ex['topdown_image'])]
            for ego_img in (ex.get('ego_images') or []):
                img_list.append(pil_to_base64(ego_img))

            question = ex['question']

            if self.use_sideview:
                sv_img = ex.get('sideview_image') or ex.get('sideview_images')
                sideview_b64 = pil_to_base64(sv_img)
                img_list.append(sideview_b64)
                sv_desc = ex.get('sideview_desc', '')
                if sv_desc:
                    sv_desc = self._sideview_desc_to_second_person(sv_desc)
                    question = question + ' ' + sv_desc

            img = img_list if len(img_list) > 1 else img_list[0]
            choices = ex['choices']

            record = {
                'index': len(records),
                'image': img,
                'question': question,
                'A': choices[0] if len(choices) > 0 else '',
                'B': choices[1] if len(choices) > 1 else '',
                'C': choices[2] if len(choices) > 2 else '',
                'D': choices[3] if len(choices) > 3 else '',
                'answer': ex['answer'],
            }

            # For vcot_prefill, store GT sideview data as extra columns
            if self.vcot_prefill:
                sv_img = ex.get('sideview_image') or ex.get('sideview_images')
                record['gt_sideview_image'] = pil_to_base64(sv_img)
                sv_desc_raw = ex.get('sideview_desc', '')
                sv_desc_clean = sv_desc_raw.replace(' <image_2>', '').replace('<image_2>', '')
                sv_desc_clean = sv_desc_clean.replace(' ..', '.').replace(' .', '.')
                sv_desc_clean = sv_desc_clean.replace(':.', ':').replace('  ', ' ').strip()
                record['gt_sideview_desc'] = sv_desc_clean

            records.append(record)

        return pd.DataFrame(records)


class AI2ThorPathTracing2PointV2Test(AI2ThorPathTracing2PointV2):
    """
    AI2Thor Path Tracing 2-Point Dataset — Debiased Test V3.
    Source: linjieli222/ai2thor_path_tracing_2point_tifa_filtered_test_v3
    Same format as V2 val but test split.
    """

    HF_REPO = 'linjieli222/ai2thor_path_tracing_2point_tifa_filtered_test_v3'

    def __init__(self, dataset='AI2ThorPT2PV2Test', subset='td_path',
                 use_sideview=False, nsamples=None, **kwargs):
        super().__init__(dataset=dataset, subset=subset,
                         use_sideview=use_sideview, nsamples=nsamples, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return [
            'AI2ThorPT2PV2Test',
            'AI2ThorPT2PV2Test_dh_midpoint', 'AI2ThorPT2PV2Test_dh_midpoint_sideview',
            'AI2ThorPT2PV2Test_td_ego_dir', 'AI2ThorPT2PV2Test_td_ego_dir_sideview',
            'AI2ThorPT2PV2Test_td_ego_dir_arrow', 'AI2ThorPT2PV2Test_td_ego_dir_arrow_sideview',
            'AI2ThorPT2PV2Test_td_ego_side', 'AI2ThorPT2PV2Test_td_ego_side_sideview',
            'AI2ThorPT2PV2Test_td_ego_side_arrow', 'AI2ThorPT2PV2Test_td_ego_side_arrow_sideview',
            'AI2ThorPT2PV2Test_td_midpoint', 'AI2ThorPT2PV2Test_td_midpoint_sideview',
            'AI2ThorPT2PV2Test_td_path', 'AI2ThorPT2PV2Test_td_path_sideview',
            'AI2ThorPT2PV2Test_td_path_arrow', 'AI2ThorPT2PV2Test_td_path_arrow_sideview',
        ]


class AI2ThorPathTracing2PointV2Hard(AI2ThorPathTracing2PointV2):
    """
    AI2Thor Path Tracing 2-Point Dataset — Debiased Hard V3.
    Source: linjieli222/ai2thor_path_tracing_2point_tifa_filtered_val_v3_hard
    """

    HF_REPO = 'linjieli222/ai2thor_path_tracing_2point_tifa_filtered_val_v3_hard'

    def __init__(self, dataset='AI2ThorPT2PV2Hard', subset='td_path',
                 use_sideview=False, nsamples=None, **kwargs):
        super().__init__(dataset=dataset, subset=subset,
                         use_sideview=use_sideview, nsamples=nsamples, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return [
            'AI2ThorPT2PV2Hard',
            'AI2ThorPT2PV2Hard_dh_midpoint', 'AI2ThorPT2PV2Hard_dh_midpoint_sideview',
            'AI2ThorPT2PV2Hard_td_ego_dir', 'AI2ThorPT2PV2Hard_td_ego_dir_sideview',
            'AI2ThorPT2PV2Hard_td_ego_dir_arrow', 'AI2ThorPT2PV2Hard_td_ego_dir_arrow_sideview',
            'AI2ThorPT2PV2Hard_td_ego_side', 'AI2ThorPT2PV2Hard_td_ego_side_sideview',
            'AI2ThorPT2PV2Hard_td_ego_side_arrow', 'AI2ThorPT2PV2Hard_td_ego_side_arrow_sideview',
            'AI2ThorPT2PV2Hard_td_midpoint', 'AI2ThorPT2PV2Hard_td_midpoint_sideview',
            'AI2ThorPT2PV2Hard_td_path', 'AI2ThorPT2PV2Hard_td_path_sideview',
            'AI2ThorPT2PV2Hard_td_path_arrow', 'AI2ThorPT2PV2Hard_td_path_arrow_sideview',
        ]


class AI2ThorSpatialVerification(AI2ThorPathTracing2PointV2):
    """
    AI2Thor Spatial Verification Dataset — Val.
    Source: linjieli222/ai2thor_spatial_verification_val_v1
    Binary Yes/No spatial verification questions.
    Same 8 subsets as PT2P datasets.
    Mapped to A=Yes, B=No for MCQ evaluation.
    """

    HF_REPO = 'linjieli222/ai2thor_spatial_verification_val_v2'

    def __init__(self, dataset='AI2ThorSV', subset='td_path',
                 use_sideview=False, nsamples=None, **kwargs):
        super().__init__(dataset=dataset, subset=subset,
                         use_sideview=use_sideview, nsamples=nsamples, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return [
            'AI2ThorSV',
            'AI2ThorSV_dh_midpoint', 'AI2ThorSV_dh_midpoint_sideview',
            'AI2ThorSV_td_ego_dir', 'AI2ThorSV_td_ego_dir_sideview',
            'AI2ThorSV_td_ego_dir_arrow', 'AI2ThorSV_td_ego_dir_arrow_sideview',
            'AI2ThorSV_td_ego_side', 'AI2ThorSV_td_ego_side_sideview',
            'AI2ThorSV_td_ego_side_arrow', 'AI2ThorSV_td_ego_side_arrow_sideview',
            'AI2ThorSV_td_midpoint', 'AI2ThorSV_td_midpoint_sideview',
            'AI2ThorSV_td_path', 'AI2ThorSV_td_path_sideview',
            'AI2ThorSV_td_path_arrow', 'AI2ThorSV_td_path_arrow_sideview',
        ]

    def load_data(self, dataset):
        from datasets import load_dataset

        try:
            hf_ds = load_dataset(self.HF_REPO, self.subset, split='val')
        except ValueError:
            hf_ds = load_dataset(self.HF_REPO, self.subset, split='train')

        records = []
        for idx, ex in enumerate(hf_ds):
            if self.nsamples is not None and len(records) >= self.nsamples:
                break

            img_list = [pil_to_base64(ex['topdown_image'])]
            for ego_img in (ex.get('ego_images') or []):
                img_list.append(pil_to_base64(ego_img))

            question = ex['question']

            if self.use_sideview:
                sv_img = ex.get('sideview_image') or ex.get('sideview_images')
                sideview_b64 = pil_to_base64(sv_img)
                img_list.append(sideview_b64)
                sv_desc = ex.get('sideview_desc', '')
                if sv_desc:
                    sv_desc = self._sideview_desc_to_second_person(sv_desc)
                    question = question + ' ' + sv_desc

            img = img_list if len(img_list) > 1 else img_list[0]

            # Map Yes/No answer to A/B
            answer_text = str(ex['answer']).strip()
            answer_letter = 'A' if answer_text.lower() == 'yes' else 'B'

            record = {
                'index': len(records),
                'image': img,
                'question': question,
                'A': 'Yes',
                'B': 'No',
                'answer': answer_letter,
            }

            # Store verification_type as category for per-type breakdown
            vtype = ex.get('verification_type', '')
            if vtype:
                record['category'] = vtype

            records.append(record)

        return pd.DataFrame(records)

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate with accuracy, precision, recall, and F1 (Yes=positive class)."""
        import numpy as np
        from ..smp import load, dump

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', f'_result.{suffix}')

        data = load(eval_file)

        # Score each sample
        if 'hit' not in data.columns:
            for i in range(len(data)):
                item = data.iloc[i]
                pred = self._extract_answer(item.get('prediction', ''))
                gt = str(item.get('answer', '')).strip().upper()
                data.loc[data.index[i], 'hit'] = 1 if pred == gt else 0
                data.loc[data.index[i], 'pred_letter'] = pred
            dump(data, result_file)

        def _compute_metrics(df):
            """Compute accuracy, precision, recall, F1 treating A (Yes) as positive."""
            acc = df['hit'].mean() * 100
            # True positives: GT=A and pred=A
            tp = ((df['answer'] == 'A') & (df['pred_letter'] == 'A')).sum()
            fp = ((df['answer'] == 'B') & (df['pred_letter'] == 'A')).sum()
            fn = ((df['answer'] == 'A') & (df['pred_letter'] != 'A')).sum()
            prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            return acc, prec, rec, f1

        acc, prec, rec, f1 = _compute_metrics(data)
        res = {
            'Category': ['Overall'],
            'Accuracy': [acc],
            'Precision': [prec],
            'Recall': [rec],
            'F1': [f1],
            'Count': [len(data)],
        }

        if 'category' in data.columns:
            for cat in sorted(data['category'].unique()):
                cat_data = data[data['category'] == cat]
                c_acc, c_prec, c_rec, c_f1 = _compute_metrics(cat_data)
                res['Category'].append(cat)
                res['Accuracy'].append(c_acc)
                res['Precision'].append(c_prec)
                res['Recall'].append(c_rec)
                res['F1'].append(c_f1)
                res['Count'].append(len(cat_data))

        res_df = pd.DataFrame(res)
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(res_df, score_file)

        return res_df

    @staticmethod
    def _extract_answer(pred):
        """Extract answer letter, handling Yes/No text output."""
        import re
        pred = str(pred).strip()

        # Try <answer> tag with letter
        match = re.search(r'<answer>\s*([A-Ba-b])\s*</answer>', pred)
        if match:
            return match.group(1).upper()
        match = re.search(r'<answer>\s*\(?([A-Ba-b])\)?\s*', pred)
        if match:
            return match.group(1).upper()

        # Try <answer> tag with Yes/No text
        match = re.search(r'<answer>\s*(yes|no)\s*</answer>', pred, re.IGNORECASE)
        if match:
            return 'A' if match.group(1).lower() == 'yes' else 'B'

        # Exact single letter
        if pred.upper() in ('A', 'B'):
            return pred.upper()

        # Check for Yes/No text anywhere
        pred_lower = pred.lower().strip()
        if pred_lower == 'yes':
            return 'A'
        if pred_lower == 'no':
            return 'B'

        # Check for Yes/No as last word (common in CoT responses)
        match = re.search(r'\b(yes|no)\s*\.?\s*$', pred, re.IGNORECASE)
        if match:
            return 'A' if match.group(1).lower() == 'yes' else 'B'

        # Fallback: letter match
        match = re.search(r'\b([A-B])\b', pred)
        if match:
            return match.group(1)

        return pred.strip().upper()


class AI2ThorSpatialVerificationTest(AI2ThorSpatialVerification):
    """
    AI2Thor Spatial Verification Dataset — Test.
    Source: linjieli222/ai2thor_spatial_verification_test_v1
    Same format as AI2ThorSpatialVerification but test split.
    """

    HF_REPO = 'linjieli222/ai2thor_spatial_verification_test_v2'

    def __init__(self, dataset='AI2ThorSVTest', subset='td_path',
                 use_sideview=False, nsamples=None, **kwargs):
        super().__init__(dataset=dataset, subset=subset,
                         use_sideview=use_sideview, nsamples=nsamples, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return [
            'AI2ThorSVTest',
            'AI2ThorSVTest_dh_midpoint', 'AI2ThorSVTest_dh_midpoint_sideview',
            'AI2ThorSVTest_td_ego_dir', 'AI2ThorSVTest_td_ego_dir_sideview',
            'AI2ThorSVTest_td_ego_dir_arrow', 'AI2ThorSVTest_td_ego_dir_arrow_sideview',
            'AI2ThorSVTest_td_ego_side', 'AI2ThorSVTest_td_ego_side_sideview',
            'AI2ThorSVTest_td_ego_side_arrow', 'AI2ThorSVTest_td_ego_side_arrow_sideview',
            'AI2ThorSVTest_td_midpoint', 'AI2ThorSVTest_td_midpoint_sideview',
            'AI2ThorSVTest_td_path', 'AI2ThorSVTest_td_path_sideview',
            'AI2ThorSVTest_td_path_arrow', 'AI2ThorSVTest_td_path_arrow_sideview',
        ]


class AI2ThorMultiViewCounting(ImageMCQDataset):
    """
    AI2Thor Multi-View Counting Dataset.
    Source: weikaih/ai2thor-multiview-counting-val-800-v2-400
    Multi-image input (3-5 frames per sample).
    4-choice MCQ (A/B/C/D).

    Splits:
    - square: 200 samples with 4 fixed camera positions
    - rotation: 200 samples with 3-5 rotation frames
    """

    TYPE = 'MCQ'

    # Filter by trajectory type: None = all, 'square' or 'rotation'
    TRAJECTORY_FILTER = None

    def __init__(self, dataset='AI2ThorMultiViewCounting', nsamples=None, **kwargs):
        self.nsamples = nsamples
        super().__init__(dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['AI2ThorMultiViewCounting']

    def load_data(self, dataset):
        from datasets import load_dataset
        import re

        # Load from HuggingFace
        hf_ds = load_dataset('weikaih/ai2thor-multiview-counting-val-800-v2-400', split='train')

        records = []
        for idx, ex in enumerate(hf_ds):
            # Early exit if we have enough samples
            if self.nsamples is not None and len(records) >= self.nsamples:
                break

            # Filter by trajectory type if specified
            traj_id = ex.get('trajectory_id', '')
            if self.TRAJECTORY_FILTER is not None:
                if self.TRAJECTORY_FILTER == 'square' and 'square' not in traj_id.lower():
                    continue
                if self.TRAJECTORY_FILTER == 'rotation' and 'rotation' not in traj_id.lower():
                    continue

            # Collect all non-None frames
            frames = []
            for i in range(8):
                frame = ex.get(f'frame_{i}')
                if frame is not None:
                    frames.append(frame)

            if len(frames) == 0:
                continue

            # Convert frames to base64 (multi-image input)
            img_b64_list = [pil_to_base64(f) for f in frames]

            # Question already has choices formatted
            question = ex['question']

            # Parse choices from question if not separate
            # Format: "How many X? A) 1 B) 2 C) 3 D) 4"
            choices = ['', '', '', '']
            choice_pattern = r'([A-D])\)\s*(\d+)'
            matches = re.findall(choice_pattern, question)
            for letter, value in matches:
                idx_choice = ord(letter) - ord('A')
                if 0 <= idx_choice < 4:
                    choices[idx_choice] = value

            records.append({
                'index': len(records),
                'image': img_b64_list,  # List of base64 for multi-image
                'question': question,
                'A': choices[0],
                'B': choices[1],
                'C': choices[2],
                'D': choices[3],
                'answer': ex['answer'],
                'category': ex.get('movement_type', ''),
                'query_object': ex.get('query_object', ''),
            })

        return pd.DataFrame(records)

    def evaluate(self, eval_file, **judge_kwargs):
        import numpy as np
        from ..smp import load, dump

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', f'_result.{suffix}')

        data = load(eval_file)

        # Score each sample using rule-based matching
        if 'hit' not in data.columns:
            for i in range(len(data)):
                item = data.iloc[i]
                pred = str(item.get('prediction', ''))
                gt = str(item.get('answer', ''))
                hit = _score_mcq_prediction(pred, gt, item)
                data.loc[data.index[i], 'hit'] = hit
            dump(data, result_file)

        # Compute per-category accuracy
        category_acc = {}
        if 'category' in data.columns:
            categories = data['category'].unique()
            for cat in categories:
                cat_data = data[data['category'] == cat]
                acc = cat_data['hit'].mean() * 100
                category_acc[cat] = acc

        overall_acc = data['hit'].mean() * 100

        res = {'Category': ['Overall'], 'Accuracy': [overall_acc], 'Count': [len(data)]}
        for cat in sorted(category_acc.keys()):
            res['Category'].append(cat)
            res['Accuracy'].append(category_acc[cat])
            cat_data = data[data['category'] == cat]
            res['Count'].append(len(cat_data))

        return pd.DataFrame(res)


class AI2ThorMultiViewCounting_Square(AI2ThorMultiViewCounting):
    """Multi-View Counting - Square trajectory only (4 fixed camera positions)."""

    TRAJECTORY_FILTER = 'square'

    def __init__(self, dataset='AI2ThorMultiViewCounting_Square', nsamples=None, **kwargs):
        self.nsamples = nsamples
        ImageMCQDataset.__init__(self, dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['AI2ThorMultiViewCounting_Square']


class AI2ThorMultiViewCounting_Rotation(AI2ThorMultiViewCounting):
    """Multi-View Counting - Rotation trajectory only (3-5 rotation frames)."""

    TRAJECTORY_FILTER = 'rotation'

    def __init__(self, dataset='AI2ThorMultiViewCounting_Rotation', nsamples=None, **kwargs):
        self.nsamples = nsamples
        ImageMCQDataset.__init__(self, dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['AI2ThorMultiViewCounting_Rotation']


class AI2ThorMultiViewCounting_10(AI2ThorMultiViewCounting):
    """Quick test version with only 10 samples."""

    def __init__(self, dataset='AI2ThorMultiViewCounting_10', **kwargs):
        super().__init__(dataset=dataset, nsamples=10, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['AI2ThorMultiViewCounting_10']


class AI2ThorMultiViewCounting_Square_10(AI2ThorMultiViewCounting_Square):
    """Quick test version - Square only, 10 samples."""

    def __init__(self, dataset='AI2ThorMultiViewCounting_Square_10', **kwargs):
        super().__init__(dataset=dataset, nsamples=10, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['AI2ThorMultiViewCounting_Square_10']


class AI2ThorMultiViewCounting_Rotation_10(AI2ThorMultiViewCounting_Rotation):
    """Quick test version - Rotation only, 10 samples."""

    def __init__(self, dataset='AI2ThorMultiViewCounting_Rotation_10', **kwargs):
        super().__init__(dataset=dataset, nsamples=10, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['AI2ThorMultiViewCounting_Rotation_10']


class HabitatPerspective_NoArrow(ImageMCQDataset):
    """
    Habitat Perspective QA Dataset (No Arrow version).
    Source: weikaih/habitat-perspective-qa
    Uses marked_image_no_arrow and question_no_arrow.
    2-choice MCQ (A/B).

    Categories (6 splits):
    - distance_closer
    - distance_further
    - position_left_left
    - position_left_right
    - position_right_left
    - position_right_right

    Total: 900 samples (150 per split)

    Overall is computed as unweighted average of 6 category accuracies.
    Used for cross-dataset generalization testing (trained on AI2Thor, tested on Habitat).
    """

    TYPE = 'MCQ'

    def __init__(self, dataset='HabitatPerspective_NoArrow', nsamples=None, **kwargs):
        self.nsamples = nsamples
        super().__init__(dataset=dataset, **kwargs)

    def evaluate(self, eval_file, **judge_kwargs):
        """Custom evaluate that computes Overall as unweighted average of category accuracies."""
        import numpy as np
        from ..smp import load, dump

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', f'_result.{suffix}')

        data = load(eval_file)

        # Score each sample using rule-based matching
        if 'hit' not in data.columns:
            for i in range(len(data)):
                item = data.iloc[i]
                pred = str(item.get('prediction', ''))
                gt = str(item.get('answer', ''))
                hit = _score_mcq_prediction(pred, gt, item)
                data.loc[data.index[i], 'hit'] = hit
            dump(data, result_file)

        # Compute per-category accuracy
        category_acc = {}
        categories = data['category'].unique()

        for cat in categories:
            cat_data = data[data['category'] == cat]
            acc = cat_data['hit'].mean() * 100  # Convert to percentage
            category_acc[cat] = acc

        # Compute Overall as unweighted average of category accuracies
        overall_acc = np.mean(list(category_acc.values()))

        # Build result DataFrame
        res = {'Category': ['Overall'], 'Accuracy': [overall_acc], 'Count': [len(data)]}
        for cat in sorted(categories):
            cat_count = len(data[data['category'] == cat])
            res['Category'].append(cat)
            res['Accuracy'].append(category_acc[cat])
            res['Count'].append(cat_count)

        res_df = pd.DataFrame(res)

        # Save results
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(res_df, score_file)

        return res_df

    @classmethod
    def supported_datasets(cls):
        return ['HabitatPerspective_NoArrow']

    def load_data(self, dataset):
        from datasets import load_dataset

        # Load all splits from HuggingFace
        hf_ds = load_dataset('weikaih/habitat-perspective-qa')

        records = []
        done = False
        for split_name in hf_ds.keys():
            if done:
                break
            for ex in hf_ds[split_name]:
                # Early exit if we have enough samples
                if self.nsamples is not None and len(records) >= self.nsamples:
                    done = True
                    break

                # Convert marked_image_no_arrow to base64
                img_b64 = pil_to_base64(ex['marked_image_no_arrow'])

                # Map answer_choices to A/B (answer_choices is a JSON string)
                choices_raw = ex['answer_choices']
                if isinstance(choices_raw, str):
                    choices = json.loads(choices_raw)
                else:
                    choices = choices_raw

                records.append({
                    'index': len(records),
                    'image': img_b64,
                    'question': ex['question_no_arrow'],
                    'A': choices[0] if len(choices) > 0 else '',
                    'B': choices[1] if len(choices) > 1 else '',
                    'answer': ex['answer'],
                    'category': split_name,  # Use split name for 6-category breakdown
                })

        return pd.DataFrame(records)


class HabitatPerspective_Arrow(ImageMCQDataset):
    """
    Habitat Perspective QA Dataset (With Arrow version).
    Source: weikaih/habitat-perspective-qa
    Uses marked_image_with_arrow and question_with_arrow.
    2-choice MCQ (A/B).

    Categories (6 splits):
    - distance_closer
    - distance_further
    - position_left_left
    - position_left_right
    - position_right_left
    - position_right_right

    Total: 900 samples (150 per split)

    Overall is computed as unweighted average of 6 category accuracies.
    """

    TYPE = 'MCQ'

    def __init__(self, dataset='HabitatPerspective_Arrow', nsamples=None, **kwargs):
        self.nsamples = nsamples
        super().__init__(dataset=dataset, **kwargs)

    def evaluate(self, eval_file, **judge_kwargs):
        """Custom evaluate that computes Overall as unweighted average of category accuracies."""
        import numpy as np
        from ..smp import load, dump

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', f'_result.{suffix}')

        data = load(eval_file)

        # Score each sample using rule-based matching
        if 'hit' not in data.columns:
            for i in range(len(data)):
                item = data.iloc[i]
                pred = str(item.get('prediction', ''))
                gt = str(item.get('answer', ''))
                hit = _score_mcq_prediction(pred, gt, item)
                data.loc[data.index[i], 'hit'] = hit
            dump(data, result_file)

        # Compute per-category accuracy
        category_acc = {}
        categories = data['category'].unique()

        for cat in categories:
            cat_data = data[data['category'] == cat]
            acc = cat_data['hit'].mean() * 100
            category_acc[cat] = acc

        # Compute Overall as unweighted average of category accuracies
        overall_acc = np.mean(list(category_acc.values()))

        # Build result DataFrame
        res = {'Category': ['Overall'], 'Accuracy': [overall_acc], 'Count': [len(data)]}
        for cat in sorted(categories):
            cat_count = len(data[data['category'] == cat])
            res['Category'].append(cat)
            res['Accuracy'].append(category_acc[cat])
            res['Count'].append(cat_count)

        res_df = pd.DataFrame(res)
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(res_df, score_file)

        return res_df

    @classmethod
    def supported_datasets(cls):
        return ['HabitatPerspective_Arrow']

    def load_data(self, dataset):
        from datasets import load_dataset

        # Load all splits from HuggingFace
        hf_ds = load_dataset('weikaih/habitat-perspective-qa')

        records = []
        done = False
        for split_name in hf_ds.keys():
            if done:
                break
            for ex in hf_ds[split_name]:
                if self.nsamples is not None and len(records) >= self.nsamples:
                    done = True
                    break

                # Convert marked_image_with_arrow to base64
                img_b64 = pil_to_base64(ex['marked_image_with_arrow'])

                # Map answer_choices to A/B (answer_choices is a JSON string)
                choices_raw = ex['answer_choices']
                if isinstance(choices_raw, str):
                    choices = json.loads(choices_raw)
                else:
                    choices = choices_raw

                records.append({
                    'index': len(records),
                    'image': img_b64,
                    'question': ex['question_with_arrow'],
                    'A': choices[0] if len(choices) > 0 else '',
                    'B': choices[1] if len(choices) > 1 else '',
                    'answer': ex['answer'],
                    'category': split_name,  # Use split name for 6-category breakdown
                })

        return pd.DataFrame(records)


class HabitatPerspective_NoArrow_10(HabitatPerspective_NoArrow):
    """Quick test version with only 10 samples."""

    def __init__(self, dataset='HabitatPerspective_NoArrow_10', **kwargs):
        super().__init__(dataset=dataset, nsamples=10, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['HabitatPerspective_NoArrow_10']


class HabitatPerspective_Arrow_10(HabitatPerspective_Arrow):
    """Quick test version with only 10 samples."""

    def __init__(self, dataset='HabitatPerspective_Arrow_10', **kwargs):
        super().__init__(dataset=dataset, nsamples=10, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['HabitatPerspective_Arrow_10']


class HabitatPerspective_NoArrow_v2(HabitatPerspective_NoArrow):
    """
    Habitat Perspective QA Dataset v2 (No Arrow version).
    Source: weikaih/habitat-perspective-qa-val-v2
    Samples 40 per category = 240 total.
    Uses marked_image_no_arrow and question_no_arrow.
    2-choice MCQ (A/B).
    """

    SAMPLES_PER_CATEGORY = 40

    def __init__(self, dataset='HabitatPerspective_NoArrow_v2', **kwargs):
        super(HabitatPerspective_NoArrow, self).__init__(dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['HabitatPerspective_NoArrow_v2']

    def load_data(self, dataset):
        import random
        from datasets import load_dataset

        hf_ds = load_dataset('weikaih/habitat-perspective-qa-val-v2')

        records = []
        for split_name in sorted(hf_ds.keys()):
            split_data = list(hf_ds[split_name])
            # Sample N per category (or take all if fewer available)
            n = min(self.SAMPLES_PER_CATEGORY, len(split_data))
            random.seed(42)
            sampled = random.sample(split_data, n)

            for ex in sampled:
                img_b64 = pil_to_base64(ex['marked_image_no_arrow'])

                choices_raw = ex['answer_choices']
                if isinstance(choices_raw, str):
                    choices = json.loads(choices_raw)
                else:
                    choices = choices_raw

                records.append({
                    'index': len(records),
                    'image': img_b64,
                    'question': ex['question_no_arrow'],
                    'A': choices[0] if len(choices) > 0 else '',
                    'B': choices[1] if len(choices) > 1 else '',
                    'answer': ex['answer'],
                    'category': split_name,
                })

        return pd.DataFrame(records)


class AI2ThorMultiViewCounting_HumanVerified(AI2ThorMultiViewCounting):
    """
    AI2Thor Multi-View Counting Dataset (Human Verified Subset).
    Source: MahtabBg/multiview_eval
    260 human-verified samples, multi-image input (4-8 frames per sample).
    4-choice MCQ (A/B/C/D).

    Categories:
    - multi_camera: samples with multiple fixed camera positions
    - rotation: samples with rotation-based camera movement
    """

    TRAJECTORY_FILTER = None

    def __init__(self, dataset='AI2ThorMultiViewCounting_HumanVerified', nsamples=None, **kwargs):
        self.nsamples = nsamples
        ImageMCQDataset.__init__(self, dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['AI2ThorMultiViewCounting_HumanVerified']

    def load_data(self, dataset):
        from datasets import load_dataset
        import re

        hf_ds = load_dataset('MahtabBg/multiview_eval', split='train')

        records = []
        for idx, ex in enumerate(hf_ds):
            if self.nsamples is not None and len(records) >= self.nsamples:
                break

            # Filter by trajectory type if specified
            movement = ex.get('movement_type', '')
            if self.TRAJECTORY_FILTER is not None:
                if self.TRAJECTORY_FILTER not in movement.lower():
                    continue

            # Collect all non-None frames
            frames = []
            for i in range(8):
                frame = ex.get(f'frame_{i}')
                if frame is not None:
                    frames.append(frame)

            if len(frames) == 0:
                continue

            img_b64_list = [pil_to_base64(f) for f in frames]

            question = ex['question']

            choices = ['', '', '', '']
            choice_pattern = r'([A-D])\)\s*(\d+)'
            matches = re.findall(choice_pattern, question)
            for letter, value in matches:
                idx_choice = ord(letter) - ord('A')
                if 0 <= idx_choice < 4:
                    choices[idx_choice] = value

            records.append({
                'index': len(records),
                'image': img_b64_list,
                'question': question,
                'A': choices[0],
                'B': choices[1],
                'C': choices[2],
                'D': choices[3],
                'answer': ex['answer'],
                'category': movement,
                'query_object': ex.get('query_object', ''),
            })

        return pd.DataFrame(records)


class AI2ThorMultiViewCounting_HumanVerified_MultiCamera(AI2ThorMultiViewCounting_HumanVerified):
    """Human Verified Multi-View Counting - Multi-camera only."""

    TRAJECTORY_FILTER = 'multi_camera'

    def __init__(self, dataset='AI2ThorMultiViewCounting_HumanVerified_MultiCamera', nsamples=None, **kwargs):
        self.nsamples = nsamples
        ImageMCQDataset.__init__(self, dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['AI2ThorMultiViewCounting_HumanVerified_MultiCamera']


class AI2ThorMultiViewCounting_HumanVerified_Rotation(AI2ThorMultiViewCounting_HumanVerified):
    """Human Verified Multi-View Counting - Rotation only."""

    TRAJECTORY_FILTER = 'rotation'

    def __init__(self, dataset='AI2ThorMultiViewCounting_HumanVerified_Rotation', nsamples=None, **kwargs):
        self.nsamples = nsamples
        ImageMCQDataset.__init__(self, dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['AI2ThorMultiViewCounting_HumanVerified_Rotation']


class AI2ThorMultiViewCounting_HumanVerified_10(AI2ThorMultiViewCounting_HumanVerified):
    """Quick test version with only 10 samples."""

    def __init__(self, dataset='AI2ThorMultiViewCounting_HumanVerified_10', **kwargs):
        super().__init__(dataset=dataset, nsamples=10, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['AI2ThorMultiViewCounting_HumanVerified_10']


class MessyTableCounting(ImageMCQDataset):
    """
    MessyTable Multi-View Counting Evaluation Dataset.
    Source: leo66666/messytable (test split, 1861 samples)
    Multi-image input (2-7 views per sample).
    4-choice MCQ (A/B/C/D) with integer answer choices.
    """

    TYPE = 'MCQ'

    def __init__(self, dataset='MessyTableCounting', nsamples=None, **kwargs):
        self.nsamples = nsamples
        super().__init__(dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['MessyTableCounting']

    def load_data(self, dataset):
        import random as rng_module
        from datasets import load_dataset

        hf_ds = load_dataset('leo66666/messytable', split='test')

        records = []
        for idx, ex in enumerate(hf_ds):
            if self.nsamples is not None and len(records) >= self.nsamples:
                break

            gt = ex['gt_answer']
            question = ex['question'].strip()
            images = ex['images']

            if not images or gt is None or not question:
                continue

            # Generate 4 MCQ choices deterministically
            rng = rng_module.Random(42 + idx)
            pool = [x for x in range(max(1, gt - 3), min(8, gt + 3) + 1) if x != gt]
            if len(pool) < 3:
                pool = [x for x in range(1, 9) if x != gt]
            distractors = rng.sample(pool, 3)
            options = distractors + [gt]
            rng.shuffle(options)
            correct_letter = 'ABCD'[options.index(gt)]

            img_b64_list = [pil_to_base64(img) for img in images]

            records.append({
                'index': len(records),
                'image': img_b64_list,
                'question': question,
                'A': str(options[0]),
                'B': str(options[1]),
                'C': str(options[2]),
                'D': str(options[3]),
                'answer': correct_letter,
            })

        return pd.DataFrame(records)

    def evaluate(self, eval_file, **judge_kwargs):
        import numpy as np
        from ..smp import load, dump

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', f'_result.{suffix}')

        data = load(eval_file)

        # Score each sample using rule-based matching
        if 'hit' not in data.columns:
            for i in range(len(data)):
                item = data.iloc[i]
                pred = str(item.get('prediction', ''))
                gt = str(item.get('answer', ''))
                hit = _score_mcq_prediction(pred, gt, item)
                data.loc[data.index[i], 'hit'] = hit
            dump(data, result_file)

        overall_acc = data['hit'].mean() * 100
        res = {'Category': ['Overall'], 'Accuracy': [overall_acc], 'Count': [len(data)]}
        return pd.DataFrame(res)


class MessyTableCounting_10(MessyTableCounting):
    """Quick test version of MessyTableCounting with only 10 samples."""

    def __init__(self, dataset='MessyTableCounting_10', **kwargs):
        super().__init__(dataset=dataset, nsamples=10, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['MessyTableCounting_10']


class RealPathTracing(AI2ThorPathTracing2Point):
    """
    Real Indoor Path Tracing Dataset.
    Source: linjieli222/real_indoor_path_tracing
    2 subsets: td_path (174 samples), td_path_arrow (158 samples).
    Single image per sample, 4-choice MCQ (A/B/C/D).
    """

    HF_REPO = 'linjieli222/real_indoor_path_tracing'
    SUBSETS = ['td_path', 'td_path_arrow']

    def __init__(self, dataset='RealPT', subset='td_path', nsamples=None, **kwargs):
        super().__init__(dataset=dataset, subset=subset,
                         use_sideview=False, nsamples=nsamples, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return [
            'RealPT', 'RealPT_td_path', 'RealPT_td_path_arrow',
        ]

    def build_prompt(self, line):
        """Build prompt with single image prepended (no <image_N> tags in question)."""
        import string

        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            from ..smp import toliststr
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = ''
        if len(options):
            options_prompt = 'Options:\n'
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'
            options_prompt += 'Please select the correct answer from the options above. \n'

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs.append(dict(type='image', value=tgt_path))
        msgs.append(dict(type='text', value=question))
        if options_prompt:
            msgs.append(dict(type='text', value=options_prompt))

        return msgs

    def load_data(self, dataset):
        from datasets import load_dataset

        hf_ds = load_dataset(self.HF_REPO, split=self.subset)

        records = []
        for idx, ex in enumerate(hf_ds):
            if self.nsamples is not None and len(records) >= self.nsamples:
                break

            img_b64 = pil_to_base64(ex['image'])
            choices = ex['choices']

            records.append({
                'index': len(records),
                'image': img_b64,
                'question': ex['question'],
                'A': choices[0] if len(choices) > 0 else '',
                'B': choices[1] if len(choices) > 1 else '',
                'C': choices[2] if len(choices) > 2 else '',
                'D': choices[3] if len(choices) > 3 else '',
                'answer': ex['answer'],
            })

        return pd.DataFrame(records)
