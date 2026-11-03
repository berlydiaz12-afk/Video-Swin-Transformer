"""Video Swin Transformer for Colab, without mmcv / mmaction.

Typical use::

    from vswin import build_model, index_dataset, make_loaders, FineTuneConfig, FineTuner

    model = build_model("swin_b_k400_22k", num_classes=2, use_checkpoint=True, device="cuda")
"""
from .checkpoints import LoadReport, download_checkpoint, load_pretrained, load_weights, read_state_dict
from .colab_setup import amp_dtype, cache_dir, default_device, describe_environment, gpu_info, mount_drive
from .dataset import (MiniclipDataset, VideoRecord, collate_video, describe_records, get_or_create_split,
                      index_dataset, label_from_path, load_split, make_clip_sampler, make_eval_loader,
                      make_loaders, save_split, select_records, split_records, split_train_val_test)
from .feature_cache import (FeatureSet, HeadTrainConfig, HeadTrainer, evaluate_features, extract_record_features,
                            get_or_extract_features, load_features, save_features)
from .fine_tuning import (FineTuneConfig, FineTuner, classification_metrics, confusion_matrix, evaluate,
                          load_checkpoint, make_criterion, save_checkpoint, set_seed)
from .inference import (VideoPrediction, extract_features, load_kinetics400_labels, predict_clips,
                        predict_video)
from .miniclip_reading import read_miniclip, read_miniclips
from .model import BinaryBCELoss, I3DHead, VideoSwin, build_model, count_parameters, logits_to_probs
from .swin_input import (CLIP_LEN, CROP_SIZE, IMAGENET_MEAN, IMAGENET_STD, check_input, dummy_input,
                         normalize, prepare_clips, to_cthw, to_model_input)
from .variants import DEFAULT_VARIANT, VARIANTS, SwinVariant, get_variant, list_variants

__version__ = "0.1.0"
