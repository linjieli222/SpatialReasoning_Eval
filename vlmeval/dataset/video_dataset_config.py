from vlmeval.dataset import *
from functools import partial

vcrbench_dataset = {
    'VCRBench_8frame_nopack': partial(VCRBench, dataset='VCR-Bench', nframe=8, pack=False),
    'VCRBench_16frame_nopack': partial(VCRBench, dataset='VCR-Bench', nframe=16, pack=False),
    'VCRBench_32frame_nopack': partial(VCRBench, dataset='VCR-Bench', nframe=32, pack=False),
    'VCRBench_64frame_nopack': partial(VCRBench, dataset='VCR-Bench', nframe=64, pack=False),
    'VCRBench_1fps_nopack': partial(VCRBench, dataset='VCR-Bench', fps=1.0, pack=False)
}

mmbench_video_dataset = {
    'MMBench_Video_8frame_nopack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=8, pack=False),
    'MMBench_Video_8frame_pack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=8, pack=True),
    'MMBench_Video_16frame_nopack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=16, pack=False),
    'MMBench_Video_64frame_nopack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=64, pack=False),
    'MMBench_Video_64frame_pack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=64, pack=True),
    'MMBench_Video_1fps_nopack': partial(MMBenchVideo, dataset='MMBench-Video', fps=1.0, pack=False),
    'MMBench_Video_1fps_pack': partial(MMBenchVideo, dataset='MMBench-Video', fps=1.0, pack=True)
}

mvbench_dataset = {
    'MVBench_8frame': partial(MVBench, dataset='MVBench', nframe=8),
    'MVBench_64frame': partial(MVBench, dataset='MVBench', nframe=64),
    # MVBench not support fps, but MVBench_MP4 does
    'MVBench_MP4_8frame': partial(MVBench_MP4, dataset='MVBench_MP4', nframe=8),
    'MVBench_MP4_1fps': partial(MVBench_MP4, dataset='MVBench_MP4', fps=1.0),
}

tamperbench_dataset = {
    'MVTamperBench_8frame': partial(MVTamperBench, dataset='MVTamperBench', nframe=8),
    'MVTamperBenchStart_8frame': partial(MVTamperBench, dataset='MVTamperBenchStart', nframe=8),
    'MVTamperBenchEnd_8frame': partial(MVTamperBench, dataset='MVTamperBenchEnd', nframe=8),
}

videomme_dataset = {
    'Video-MME_8frame': partial(VideoMME, dataset='Video-MME', nframe=8),
    'Video-MME_64frame': partial(VideoMME, dataset='Video-MME', nframe=64),
    'Video-MME_8frame_subs': partial(VideoMME, dataset='Video-MME', nframe=8, use_subtitle=True),
    'Video-MME_1fps': partial(VideoMME, dataset='Video-MME', fps=1.0),
    'Video-MME_0.5fps': partial(VideoMME, dataset='Video-MME', fps=0.5),
    'Video-MME_0.5fps_subs': partial(VideoMME, dataset='Video-MME', fps=0.5, use_subtitle=True),
}

videommmu_dataset = {
    'VideoMMMU_8frame': partial(VideoMMMU, dataset='VideoMMMU', nframe=8),
    'VideoMMMU_64frame': partial(VideoMMMU, dataset='VideoMMMU', nframe=64),
    'VideoMMMU_1fps': partial(VideoMMMU, dataset='VideoMMMU', fps=1.0),
    'VideoMMMU_0.5fps': partial(VideoMMMU, dataset='VideoMMMU', fps=0.5),
}

longvideobench_dataset = {
    'LongVideoBench_8frame': partial(LongVideoBench, dataset='LongVideoBench', nframe=8),
    'LongVideoBench_8frame_subs': partial(LongVideoBench, dataset='LongVideoBench', nframe=8, use_subtitle=True),
    'LongVideoBench_64frame': partial(LongVideoBench, dataset='LongVideoBench', nframe=64),
    'LongVideoBench_1fps': partial(LongVideoBench, dataset='LongVideoBench', fps=1.0),
    'LongVideoBench_0.5fps': partial(LongVideoBench, dataset='LongVideoBench', fps=0.5),
    'LongVideoBench_0.5fps_subs': partial(LongVideoBench, dataset='LongVideoBench', fps=0.5, use_subtitle=True)
}

mlvu_dataset = {
    'MLVU_8frame': partial(MLVU, dataset='MLVU', nframe=8),
    'MLVU_64frame': partial(MLVU, dataset='MLVU', nframe=64),
    'MLVU_1fps': partial(MLVU, dataset='MLVU', fps=1.0)
}

tempcompass_dataset = {
    'TempCompass_8frame': partial(TempCompass, dataset='TempCompass', nframe=8),
    'TempCompass_64frame': partial(TempCompass, dataset='TempCompass', nframe=64),
    'TempCompass_1fps': partial(TempCompass, dataset='TempCompass', fps=1.0),
    'TempCompass_0.5fps': partial(TempCompass, dataset='TempCompass', fps=0.5)
}

# In order to reproduce the experimental results in CGbench paper,
# use_subtitle, use_subtitle_time and use_frame_time need to be set to True.
# When measuring clue-related results, if the number of frames used is greater
# than 32, the frame capture limit will be set to 32.
# We implement the metrics long_acc, clue_acc, miou, CRR, acc@iou and rec@iou
# in the CGBench_MCQ_Grounding_Mini and CGBench_MCQ_Grounding datasets;
# the metric open-ended is implemented in the CGBench_OpenEnded_Mini and CGBench_OpenEnded datasets.
cgbench_dataset = {
    'CGBench_MCQ_Grounding_Mini_8frame_subs_subt': partial(
        CGBench_MCQ_Grounding_Mini,
        dataset='CG-Bench_MCQ_Grounding_Mini',
        nframe=8,
        use_subtitle=True,
        use_subtitle_time=True
    ),
    'CGBench_OpenEnded_Mini_8frame_subs_subt_ft': partial(
        CGBench_OpenEnded_Mini,
        dataset='CG-Bench_OpenEnded_Mini',
        nframe=8,
        use_subtitle=True,
        use_subtitle_time=True,
        use_frame_time=True
    ),
    'CGBench_MCQ_Grounding_32frame_subs': partial(
        CGBench_MCQ_Grounding,
        dataset='CG-Bench_MCQ_Grounding',
        nframe=32,
        use_subtitle=True
    ),
    'CGBench_OpenEnded_8frame': partial(
        CGBench_OpenEnded,
        dataset='CG-Bench_OpenEnded',
        nframe=8
    ),
    'CGBench_MCQ_Grounding_16frame_subs_subt_ft': partial(
        CGBench_MCQ_Grounding,
        dataset='CG-Bench_MCQ_Grounding',
        nframe=16,
        use_subtitle=True,
        use_subtitle_time=True,
        use_frame_time=True
    ),
    'CGBench_OpenEnded_16frame_subs_subt_ft': partial(
        CGBench_OpenEnded,
        dataset='CG-Bench_OpenEnded',
        nframe=16,
        use_subtitle=True,
        use_subtitle_time=True,
        use_frame_time=True
    )
}

megabench_dataset = {
    'MEGABench_core_16frame': partial(MEGABench, dataset='MEGABench', nframe=16, subset_name="core"),
    'MEGABench_open_16frame': partial(MEGABench, dataset='MEGABench', nframe=16, subset_name="open"),
    'MEGABench_core_64frame': partial(MEGABench, dataset='MEGABench', nframe=64, subset_name="core"),
    'MEGABench_open_64frame': partial(MEGABench, dataset='MEGABench', nframe=64, subset_name="open")
}

moviechat1k_dataset = {
    'moviechat1k_breakpoint_8frame': partial(MovieChat1k, dataset='MovieChat1k', subset='breakpoint', nframe=8),
    'moviechat1k_global_14frame': partial(MovieChat1k, dataset='MovieChat1k', subset='global', nframe=14),
    'moviechat1k_global_8frame_limit0.01': partial(
        MovieChat1k, dataset='MovieChat1k', subset='global', nframe=8, limit=0.01
    )
}

vdc_dataset = {
    'VDC_8frame': partial(VDC, dataset='VDC', nframe=8),
    'VDC_1fps': partial(VDC, dataset='VDC', fps=1.0),
}

worldsense_dataset = {
    'WorldSense_8frame': partial(WorldSense, dataset='WorldSense', nframe=8),
    'WorldSense_8frame_subs': partial(WorldSense, dataset='WorldSense', nframe=8, use_subtitle=True),
    'WorldSense_8frame_audio': partial(WorldSense, dataset='WorldSense', nframe=8, use_audio=True),
    'WorldSense_32frame': partial(WorldSense, dataset='WorldSense', nframe=32),
    'WorldSense_32frame_subs': partial(WorldSense, dataset='WorldSense', nframe=32, use_subtitle=True),
    'WorldSense_32frame_audio': partial(WorldSense, dataset='WorldSense', nframe=32, use_audio=True),
    'WorldSense_1fps': partial(WorldSense, dataset='WorldSense', fps=1.0),
    'WorldSense_1fps_subs': partial(WorldSense, dataset='WorldSense', fps=1.0, use_subtitle=True),
    'WorldSense_1fps_audio': partial(WorldSense, dataset='WorldSense', fps=1.0, use_audio=True),
    'WorldSense_0.5fps': partial(WorldSense, dataset='WorldSense', fps=0.5),
    'WorldSense_0.5fps_subs': partial(WorldSense, dataset='WorldSense', fps=0.5, use_subtitle=True),
    'WorldSense_0.5fps_audio': partial(WorldSense, dataset='WorldSense', fps=0.5, use_audio=True)
}

qbench_video_dataset = {
    'QBench_Video_8frame': partial(QBench_Video, dataset='QBench_Video', nframe=8),
    'QBench_Video_16frame': partial(QBench_Video, dataset='QBench_Video', nframe=16),
}

video_mmlu_dataset = {
    'Video_MMLU_CAP_16frame': partial(Video_MMLU_CAP, dataset='Video_MMLU_CAP', nframe=16),
    'Video_MMLU_CAP_64frame': partial(Video_MMLU_CAP, dataset='Video_MMLU_CAP', nframe=64),
    'Video_MMLU_QA_16frame': partial(Video_MMLU_QA, dataset='Video_MMLU_QA', nframe=16),
    'Video_MMLU_QA_64frame': partial(Video_MMLU_QA, dataset='Video_MMLU_QA', nframe=64),
}

video_tt_dataset = {
    'Video_TT_16frame': partial(VideoTT, dataset='Video-TT', nframe=16),
    'Video_TT_32frame': partial(VideoTT, dataset='Video-TT', nframe=32),
    'Video_TT_64frame': partial(VideoTT, dataset='Video-TT', nframe=64),
}

video_holmes_dataset = {
    'Video_Holmes_32frame': partial(Video_Holmes, dataset='Video_Holmes', nframe=32),
    'Video_Holmes_64frame': partial(Video_Holmes, dataset='Video_Holmes', nframe=64),
}

cg_av_counting_dataset = {
    'CG-AV-Counting_32frame': partial(CGAVCounting, dataset='CG-AV-Counting', nframe=32, use_frame_time=False),
    'CG-AV-Counting_64frame': partial(CGAVCounting, dataset='CG-AV-Counting', nframe=64, use_frame_time=False)
}

egoexobench_dataset = {
    'EgoExoBench_64frame': partial(EgoExoBench_MCQ, dataset='EgoExoBench_MCQ', nframe=64, skip_EgoExo4D=False),  # noqa: E501
    'EgoExoBench_64frame_skip_EgoExo4D': partial(EgoExoBench_MCQ, dataset='EgoExoBench_MCQ', nframe=64, skip_EgoExo4D=True)  # noqa: E501

}

vsibench_dataset = {
    'vsibench_16frame': partial(VSIBench, dataset='VSIBench', nframe=16),
    'vsibench_32frame': partial(VSIBench, dataset='VSIBench', nframe=32),
    'vsibench_64frame': partial(VSIBench, dataset='VSIBench', nframe=64),
}

dream_1k_dataset = {
    'DREAM-1K_8frame': partial(DREAM, dataset='DREAM-1K', nframe=8),
    'DREAM-1K_64frame': partial(DREAM, dataset='DREAM-1K', nframe=64),
    'DREAM-1K_2fps': partial(DREAM, dataset='DREAM-1K', fps=2.0),
    'DREAM-1K_1fps': partial(DREAM, dataset='DREAM-1K', fps=1.0),
    'DREAM-1K_0.5fps': partial(DREAM, dataset='DREAM-1K', fps=0.5),
}

# AI2Thor spatial reasoning datasets with sample limits
ai2thor_dataset = {
    'AI2ThorPathTracing': partial(AI2ThorPathTracing, dataset='AI2ThorPathTracing'),
    'AI2ThorPathTracing_sideview': partial(AI2ThorPathTracing, dataset='AI2ThorPathTracing_sideview', use_sideview=True),
    'AI2ThorPathTracing_10': partial(AI2ThorPathTracing, dataset='AI2ThorPathTracing', nsamples=10),
    'AI2ThorPerspective_NoArrow': partial(AI2ThorPerspective_NoArrow, dataset='AI2ThorPerspective_NoArrow'),
    'AI2ThorPerspective_NoArrow_10': partial(AI2ThorPerspective_NoArrow, dataset='AI2ThorPerspective_NoArrow', nsamples=10),
    'AI2ThorPerspective_Arrow': partial(AI2ThorPerspective_Arrow, dataset='AI2ThorPerspective_Arrow'),
    'SideviewOverfit_10': partial(SideviewOverfit, dataset='SideviewOverfit', nsamples=10),
    # Path Tracing 2-Point: 8 subsets x 2 (with/without sideview) = 16 entries
    'AI2ThorPT2P_dh_midpoint': partial(AI2ThorPathTracing2Point, dataset='AI2ThorPT2P_dh_midpoint', subset='dh_midpoint', use_sideview=False),
    'AI2ThorPT2P_dh_midpoint_sideview': partial(AI2ThorPathTracing2Point, dataset='AI2ThorPT2P_dh_midpoint_sideview', subset='dh_midpoint', use_sideview=True),
    'AI2ThorPT2P_td_ego_dir': partial(AI2ThorPathTracing2Point, dataset='AI2ThorPT2P_td_ego_dir', subset='td_ego_dir', use_sideview=False),
    'AI2ThorPT2P_td_ego_dir_sideview': partial(AI2ThorPathTracing2Point, dataset='AI2ThorPT2P_td_ego_dir_sideview', subset='td_ego_dir', use_sideview=True),
    'AI2ThorPT2P_td_ego_dir_arrow': partial(AI2ThorPathTracing2Point, dataset='AI2ThorPT2P_td_ego_dir_arrow', subset='td_ego_dir_arrow', use_sideview=False),
    'AI2ThorPT2P_td_ego_dir_arrow_sideview': partial(AI2ThorPathTracing2Point, dataset='AI2ThorPT2P_td_ego_dir_arrow_sideview', subset='td_ego_dir_arrow', use_sideview=True),
    'AI2ThorPT2P_td_ego_side': partial(AI2ThorPathTracing2Point, dataset='AI2ThorPT2P_td_ego_side', subset='td_ego_side', use_sideview=False),
    'AI2ThorPT2P_td_ego_side_sideview': partial(AI2ThorPathTracing2Point, dataset='AI2ThorPT2P_td_ego_side_sideview', subset='td_ego_side', use_sideview=True),
    'AI2ThorPT2P_td_ego_side_arrow': partial(AI2ThorPathTracing2Point, dataset='AI2ThorPT2P_td_ego_side_arrow', subset='td_ego_side_arrow', use_sideview=False),
    'AI2ThorPT2P_td_ego_side_arrow_sideview': partial(AI2ThorPathTracing2Point, dataset='AI2ThorPT2P_td_ego_side_arrow_sideview', subset='td_ego_side_arrow', use_sideview=True),
    'AI2ThorPT2P_td_midpoint': partial(AI2ThorPathTracing2Point, dataset='AI2ThorPT2P_td_midpoint', subset='td_midpoint', use_sideview=False),
    'AI2ThorPT2P_td_midpoint_sideview': partial(AI2ThorPathTracing2Point, dataset='AI2ThorPT2P_td_midpoint_sideview', subset='td_midpoint', use_sideview=True),
    'AI2ThorPT2P_td_path': partial(AI2ThorPathTracing2Point, dataset='AI2ThorPT2P_td_path', subset='td_path', use_sideview=False),
    'AI2ThorPT2P_td_path_sideview': partial(AI2ThorPathTracing2Point, dataset='AI2ThorPT2P_td_path_sideview', subset='td_path', use_sideview=True),
    'AI2ThorPT2P_td_path_arrow': partial(AI2ThorPathTracing2Point, dataset='AI2ThorPT2P_td_path_arrow', subset='td_path_arrow', use_sideview=False),
    'AI2ThorPT2P_td_path_arrow_sideview': partial(AI2ThorPathTracing2Point, dataset='AI2ThorPT2P_td_path_arrow_sideview', subset='td_path_arrow', use_sideview=True),
    # Path Tracing 2-Point: VCoT GT prefill (topdown-only input, GT sideview injected as prefill)
    'AI2ThorPT2P_td_path_vcot_prefill': partial(AI2ThorPathTracing2Point, dataset='AI2ThorPT2P_td_path_vcot_prefill', subset='td_path', use_sideview=False, vcot_prefill=True),
    # Path Tracing 2-Point V2 (debiased): 8 subsets x 2 (with/without sideview) = 16 entries
    'AI2ThorPT2PV2_dh_midpoint': partial(AI2ThorPathTracing2PointV2, dataset='AI2ThorPT2PV2_dh_midpoint', subset='dh_midpoint', use_sideview=False),
    'AI2ThorPT2PV2_dh_midpoint_sideview': partial(AI2ThorPathTracing2PointV2, dataset='AI2ThorPT2PV2_dh_midpoint_sideview', subset='dh_midpoint', use_sideview=True),
    'AI2ThorPT2PV2_td_ego_dir': partial(AI2ThorPathTracing2PointV2, dataset='AI2ThorPT2PV2_td_ego_dir', subset='td_ego_dir', use_sideview=False),
    'AI2ThorPT2PV2_td_ego_dir_sideview': partial(AI2ThorPathTracing2PointV2, dataset='AI2ThorPT2PV2_td_ego_dir_sideview', subset='td_ego_dir', use_sideview=True),
    'AI2ThorPT2PV2_td_ego_dir_arrow': partial(AI2ThorPathTracing2PointV2, dataset='AI2ThorPT2PV2_td_ego_dir_arrow', subset='td_ego_dir_arrow', use_sideview=False),
    'AI2ThorPT2PV2_td_ego_dir_arrow_sideview': partial(AI2ThorPathTracing2PointV2, dataset='AI2ThorPT2PV2_td_ego_dir_arrow_sideview', subset='td_ego_dir_arrow', use_sideview=True),
    'AI2ThorPT2PV2_td_ego_side': partial(AI2ThorPathTracing2PointV2, dataset='AI2ThorPT2PV2_td_ego_side', subset='td_ego_side', use_sideview=False),
    'AI2ThorPT2PV2_td_ego_side_sideview': partial(AI2ThorPathTracing2PointV2, dataset='AI2ThorPT2PV2_td_ego_side_sideview', subset='td_ego_side', use_sideview=True),
    'AI2ThorPT2PV2_td_ego_side_arrow': partial(AI2ThorPathTracing2PointV2, dataset='AI2ThorPT2PV2_td_ego_side_arrow', subset='td_ego_side_arrow', use_sideview=False),
    'AI2ThorPT2PV2_td_ego_side_arrow_sideview': partial(AI2ThorPathTracing2PointV2, dataset='AI2ThorPT2PV2_td_ego_side_arrow_sideview', subset='td_ego_side_arrow', use_sideview=True),
    'AI2ThorPT2PV2_td_midpoint': partial(AI2ThorPathTracing2PointV2, dataset='AI2ThorPT2PV2_td_midpoint', subset='td_midpoint', use_sideview=False),
    'AI2ThorPT2PV2_td_midpoint_sideview': partial(AI2ThorPathTracing2PointV2, dataset='AI2ThorPT2PV2_td_midpoint_sideview', subset='td_midpoint', use_sideview=True),
    'AI2ThorPT2PV2_td_path': partial(AI2ThorPathTracing2PointV2, dataset='AI2ThorPT2PV2_td_path', subset='td_path', use_sideview=False),
    'AI2ThorPT2PV2_td_path_sideview': partial(AI2ThorPathTracing2PointV2, dataset='AI2ThorPT2PV2_td_path_sideview', subset='td_path', use_sideview=True),
    'AI2ThorPT2PV2_td_path_arrow': partial(AI2ThorPathTracing2PointV2, dataset='AI2ThorPT2PV2_td_path_arrow', subset='td_path_arrow', use_sideview=False),
    'AI2ThorPT2PV2_td_path_arrow_sideview': partial(AI2ThorPathTracing2PointV2, dataset='AI2ThorPT2PV2_td_path_arrow_sideview', subset='td_path_arrow', use_sideview=True),
    # Path Tracing 2-Point V2 Test (debiased): 8 subsets x 2 (with/without sideview) = 16 entries
    'AI2ThorPT2PV2Test_dh_midpoint': partial(AI2ThorPathTracing2PointV2Test, dataset='AI2ThorPT2PV2Test_dh_midpoint', subset='dh_midpoint', use_sideview=False),
    'AI2ThorPT2PV2Test_dh_midpoint_sideview': partial(AI2ThorPathTracing2PointV2Test, dataset='AI2ThorPT2PV2Test_dh_midpoint_sideview', subset='dh_midpoint', use_sideview=True),
    'AI2ThorPT2PV2Test_td_ego_dir': partial(AI2ThorPathTracing2PointV2Test, dataset='AI2ThorPT2PV2Test_td_ego_dir', subset='td_ego_dir', use_sideview=False),
    'AI2ThorPT2PV2Test_td_ego_dir_sideview': partial(AI2ThorPathTracing2PointV2Test, dataset='AI2ThorPT2PV2Test_td_ego_dir_sideview', subset='td_ego_dir', use_sideview=True),
    'AI2ThorPT2PV2Test_td_ego_dir_arrow': partial(AI2ThorPathTracing2PointV2Test, dataset='AI2ThorPT2PV2Test_td_ego_dir_arrow', subset='td_ego_dir_arrow', use_sideview=False),
    'AI2ThorPT2PV2Test_td_ego_dir_arrow_sideview': partial(AI2ThorPathTracing2PointV2Test, dataset='AI2ThorPT2PV2Test_td_ego_dir_arrow_sideview', subset='td_ego_dir_arrow', use_sideview=True),
    'AI2ThorPT2PV2Test_td_ego_side': partial(AI2ThorPathTracing2PointV2Test, dataset='AI2ThorPT2PV2Test_td_ego_side', subset='td_ego_side', use_sideview=False),
    'AI2ThorPT2PV2Test_td_ego_side_sideview': partial(AI2ThorPathTracing2PointV2Test, dataset='AI2ThorPT2PV2Test_td_ego_side_sideview', subset='td_ego_side', use_sideview=True),
    'AI2ThorPT2PV2Test_td_ego_side_arrow': partial(AI2ThorPathTracing2PointV2Test, dataset='AI2ThorPT2PV2Test_td_ego_side_arrow', subset='td_ego_side_arrow', use_sideview=False),
    'AI2ThorPT2PV2Test_td_ego_side_arrow_sideview': partial(AI2ThorPathTracing2PointV2Test, dataset='AI2ThorPT2PV2Test_td_ego_side_arrow_sideview', subset='td_ego_side_arrow', use_sideview=True),
    'AI2ThorPT2PV2Test_td_midpoint': partial(AI2ThorPathTracing2PointV2Test, dataset='AI2ThorPT2PV2Test_td_midpoint', subset='td_midpoint', use_sideview=False),
    'AI2ThorPT2PV2Test_td_midpoint_sideview': partial(AI2ThorPathTracing2PointV2Test, dataset='AI2ThorPT2PV2Test_td_midpoint_sideview', subset='td_midpoint', use_sideview=True),
    'AI2ThorPT2PV2Test_td_path': partial(AI2ThorPathTracing2PointV2Test, dataset='AI2ThorPT2PV2Test_td_path', subset='td_path', use_sideview=False),
    'AI2ThorPT2PV2Test_td_path_sideview': partial(AI2ThorPathTracing2PointV2Test, dataset='AI2ThorPT2PV2Test_td_path_sideview', subset='td_path', use_sideview=True),
    'AI2ThorPT2PV2Test_td_path_arrow': partial(AI2ThorPathTracing2PointV2Test, dataset='AI2ThorPT2PV2Test_td_path_arrow', subset='td_path_arrow', use_sideview=False),
    'AI2ThorPT2PV2Test_td_path_arrow_sideview': partial(AI2ThorPathTracing2PointV2Test, dataset='AI2ThorPT2PV2Test_td_path_arrow_sideview', subset='td_path_arrow', use_sideview=True),
    # Path Tracing 2-Point V2 Hard (debiased hard): 8 subsets x 2 (with/without sideview) = 16 entries
    'AI2ThorPT2PV2Hard_dh_midpoint': partial(AI2ThorPathTracing2PointV2Hard, dataset='AI2ThorPT2PV2Hard_dh_midpoint', subset='dh_midpoint', use_sideview=False),
    'AI2ThorPT2PV2Hard_dh_midpoint_sideview': partial(AI2ThorPathTracing2PointV2Hard, dataset='AI2ThorPT2PV2Hard_dh_midpoint_sideview', subset='dh_midpoint', use_sideview=True),
    'AI2ThorPT2PV2Hard_td_ego_dir': partial(AI2ThorPathTracing2PointV2Hard, dataset='AI2ThorPT2PV2Hard_td_ego_dir', subset='td_ego_dir', use_sideview=False),
    'AI2ThorPT2PV2Hard_td_ego_dir_sideview': partial(AI2ThorPathTracing2PointV2Hard, dataset='AI2ThorPT2PV2Hard_td_ego_dir_sideview', subset='td_ego_dir', use_sideview=True),
    'AI2ThorPT2PV2Hard_td_ego_dir_arrow': partial(AI2ThorPathTracing2PointV2Hard, dataset='AI2ThorPT2PV2Hard_td_ego_dir_arrow', subset='td_ego_dir_arrow', use_sideview=False),
    'AI2ThorPT2PV2Hard_td_ego_dir_arrow_sideview': partial(AI2ThorPathTracing2PointV2Hard, dataset='AI2ThorPT2PV2Hard_td_ego_dir_arrow_sideview', subset='td_ego_dir_arrow', use_sideview=True),
    'AI2ThorPT2PV2Hard_td_ego_side': partial(AI2ThorPathTracing2PointV2Hard, dataset='AI2ThorPT2PV2Hard_td_ego_side', subset='td_ego_side', use_sideview=False),
    'AI2ThorPT2PV2Hard_td_ego_side_sideview': partial(AI2ThorPathTracing2PointV2Hard, dataset='AI2ThorPT2PV2Hard_td_ego_side_sideview', subset='td_ego_side', use_sideview=True),
    'AI2ThorPT2PV2Hard_td_ego_side_arrow': partial(AI2ThorPathTracing2PointV2Hard, dataset='AI2ThorPT2PV2Hard_td_ego_side_arrow', subset='td_ego_side_arrow', use_sideview=False),
    'AI2ThorPT2PV2Hard_td_ego_side_arrow_sideview': partial(AI2ThorPathTracing2PointV2Hard, dataset='AI2ThorPT2PV2Hard_td_ego_side_arrow_sideview', subset='td_ego_side_arrow', use_sideview=True),
    'AI2ThorPT2PV2Hard_td_midpoint': partial(AI2ThorPathTracing2PointV2Hard, dataset='AI2ThorPT2PV2Hard_td_midpoint', subset='td_midpoint', use_sideview=False),
    'AI2ThorPT2PV2Hard_td_midpoint_sideview': partial(AI2ThorPathTracing2PointV2Hard, dataset='AI2ThorPT2PV2Hard_td_midpoint_sideview', subset='td_midpoint', use_sideview=True),
    'AI2ThorPT2PV2Hard_td_path': partial(AI2ThorPathTracing2PointV2Hard, dataset='AI2ThorPT2PV2Hard_td_path', subset='td_path', use_sideview=False),
    'AI2ThorPT2PV2Hard_td_path_sideview': partial(AI2ThorPathTracing2PointV2Hard, dataset='AI2ThorPT2PV2Hard_td_path_sideview', subset='td_path', use_sideview=True),
    'AI2ThorPT2PV2Hard_td_path_arrow': partial(AI2ThorPathTracing2PointV2Hard, dataset='AI2ThorPT2PV2Hard_td_path_arrow', subset='td_path_arrow', use_sideview=False),
    'AI2ThorPT2PV2Hard_td_path_arrow_sideview': partial(AI2ThorPathTracing2PointV2Hard, dataset='AI2ThorPT2PV2Hard_td_path_arrow_sideview', subset='td_path_arrow', use_sideview=True),
    # Spatial Verification Val: 8 subsets x 2 (with/without sideview) = 16 entries
    'AI2ThorSV_dh_midpoint': partial(AI2ThorSpatialVerification, dataset='AI2ThorSV_dh_midpoint', subset='dh_midpoint', use_sideview=False),
    'AI2ThorSV_dh_midpoint_sideview': partial(AI2ThorSpatialVerification, dataset='AI2ThorSV_dh_midpoint_sideview', subset='dh_midpoint', use_sideview=True),
    'AI2ThorSV_td_ego_dir': partial(AI2ThorSpatialVerification, dataset='AI2ThorSV_td_ego_dir', subset='td_ego_dir', use_sideview=False),
    'AI2ThorSV_td_ego_dir_sideview': partial(AI2ThorSpatialVerification, dataset='AI2ThorSV_td_ego_dir_sideview', subset='td_ego_dir', use_sideview=True),
    'AI2ThorSV_td_ego_dir_arrow': partial(AI2ThorSpatialVerification, dataset='AI2ThorSV_td_ego_dir_arrow', subset='td_ego_dir_arrow', use_sideview=False),
    'AI2ThorSV_td_ego_dir_arrow_sideview': partial(AI2ThorSpatialVerification, dataset='AI2ThorSV_td_ego_dir_arrow_sideview', subset='td_ego_dir_arrow', use_sideview=True),
    'AI2ThorSV_td_ego_side': partial(AI2ThorSpatialVerification, dataset='AI2ThorSV_td_ego_side', subset='td_ego_side', use_sideview=False),
    'AI2ThorSV_td_ego_side_sideview': partial(AI2ThorSpatialVerification, dataset='AI2ThorSV_td_ego_side_sideview', subset='td_ego_side', use_sideview=True),
    'AI2ThorSV_td_ego_side_arrow': partial(AI2ThorSpatialVerification, dataset='AI2ThorSV_td_ego_side_arrow', subset='td_ego_side_arrow', use_sideview=False),
    'AI2ThorSV_td_ego_side_arrow_sideview': partial(AI2ThorSpatialVerification, dataset='AI2ThorSV_td_ego_side_arrow_sideview', subset='td_ego_side_arrow', use_sideview=True),
    'AI2ThorSV_td_midpoint': partial(AI2ThorSpatialVerification, dataset='AI2ThorSV_td_midpoint', subset='td_midpoint', use_sideview=False),
    'AI2ThorSV_td_midpoint_sideview': partial(AI2ThorSpatialVerification, dataset='AI2ThorSV_td_midpoint_sideview', subset='td_midpoint', use_sideview=True),
    'AI2ThorSV_td_path': partial(AI2ThorSpatialVerification, dataset='AI2ThorSV_td_path', subset='td_path', use_sideview=False),
    'AI2ThorSV_td_path_sideview': partial(AI2ThorSpatialVerification, dataset='AI2ThorSV_td_path_sideview', subset='td_path', use_sideview=True),
    'AI2ThorSV_td_path_arrow': partial(AI2ThorSpatialVerification, dataset='AI2ThorSV_td_path_arrow', subset='td_path_arrow', use_sideview=False),
    'AI2ThorSV_td_path_arrow_sideview': partial(AI2ThorSpatialVerification, dataset='AI2ThorSV_td_path_arrow_sideview', subset='td_path_arrow', use_sideview=True),
    # Spatial Verification Test: 8 subsets x 2 (with/without sideview) = 16 entries
    'AI2ThorSVTest_dh_midpoint': partial(AI2ThorSpatialVerificationTest, dataset='AI2ThorSVTest_dh_midpoint', subset='dh_midpoint', use_sideview=False),
    'AI2ThorSVTest_dh_midpoint_sideview': partial(AI2ThorSpatialVerificationTest, dataset='AI2ThorSVTest_dh_midpoint_sideview', subset='dh_midpoint', use_sideview=True),
    'AI2ThorSVTest_td_ego_dir': partial(AI2ThorSpatialVerificationTest, dataset='AI2ThorSVTest_td_ego_dir', subset='td_ego_dir', use_sideview=False),
    'AI2ThorSVTest_td_ego_dir_sideview': partial(AI2ThorSpatialVerificationTest, dataset='AI2ThorSVTest_td_ego_dir_sideview', subset='td_ego_dir', use_sideview=True),
    'AI2ThorSVTest_td_ego_dir_arrow': partial(AI2ThorSpatialVerificationTest, dataset='AI2ThorSVTest_td_ego_dir_arrow', subset='td_ego_dir_arrow', use_sideview=False),
    'AI2ThorSVTest_td_ego_dir_arrow_sideview': partial(AI2ThorSpatialVerificationTest, dataset='AI2ThorSVTest_td_ego_dir_arrow_sideview', subset='td_ego_dir_arrow', use_sideview=True),
    'AI2ThorSVTest_td_ego_side': partial(AI2ThorSpatialVerificationTest, dataset='AI2ThorSVTest_td_ego_side', subset='td_ego_side', use_sideview=False),
    'AI2ThorSVTest_td_ego_side_sideview': partial(AI2ThorSpatialVerificationTest, dataset='AI2ThorSVTest_td_ego_side_sideview', subset='td_ego_side', use_sideview=True),
    'AI2ThorSVTest_td_ego_side_arrow': partial(AI2ThorSpatialVerificationTest, dataset='AI2ThorSVTest_td_ego_side_arrow', subset='td_ego_side_arrow', use_sideview=False),
    'AI2ThorSVTest_td_ego_side_arrow_sideview': partial(AI2ThorSpatialVerificationTest, dataset='AI2ThorSVTest_td_ego_side_arrow_sideview', subset='td_ego_side_arrow', use_sideview=True),
    'AI2ThorSVTest_td_midpoint': partial(AI2ThorSpatialVerificationTest, dataset='AI2ThorSVTest_td_midpoint', subset='td_midpoint', use_sideview=False),
    'AI2ThorSVTest_td_midpoint_sideview': partial(AI2ThorSpatialVerificationTest, dataset='AI2ThorSVTest_td_midpoint_sideview', subset='td_midpoint', use_sideview=True),
    'AI2ThorSVTest_td_path': partial(AI2ThorSpatialVerificationTest, dataset='AI2ThorSVTest_td_path', subset='td_path', use_sideview=False),
    'AI2ThorSVTest_td_path_sideview': partial(AI2ThorSpatialVerificationTest, dataset='AI2ThorSVTest_td_path_sideview', subset='td_path', use_sideview=True),
    'AI2ThorSVTest_td_path_arrow': partial(AI2ThorSpatialVerificationTest, dataset='AI2ThorSVTest_td_path_arrow', subset='td_path_arrow', use_sideview=False),
    'AI2ThorSVTest_td_path_arrow_sideview': partial(AI2ThorSpatialVerificationTest, dataset='AI2ThorSVTest_td_path_arrow_sideview', subset='td_path_arrow', use_sideview=True),
}

# SAT Circular dataset with sample limits
from .sat_circular_limited import SATCircularLimited, SATPerspectiveTaking
sat_circular_dataset = {
    'SAT_circular_10': partial(SATCircularLimited, dataset='SAT_circular', nsamples=10),
    'SAT_perspective_10': partial(SATPerspectiveTaking, dataset='SAT_perspective', nsamples=10),
}

supported_video_datasets = {}

dataset_groups = [
    mmbench_video_dataset, mvbench_dataset, videomme_dataset, videommmu_dataset, longvideobench_dataset,
    mlvu_dataset, tempcompass_dataset, cgbench_dataset, worldsense_dataset, tamperbench_dataset,
    megabench_dataset, qbench_video_dataset, moviechat1k_dataset, vdc_dataset, video_holmes_dataset, vcrbench_dataset,
    cg_av_counting_dataset, video_mmlu_dataset, egoexobench_dataset, dream_1k_dataset, video_tt_dataset,
    vsibench_dataset, ai2thor_dataset, sat_circular_dataset
]

for grp in dataset_groups:
    supported_video_datasets.update(grp)
