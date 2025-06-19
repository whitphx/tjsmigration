# Copied from https://github.com/huggingface/optimum/blob/be76dde503d125b6a21cf1d3bec99e1185f86ba8/optimum/exporters/tasks.py#L1704

from huggingface_hub.hf_api import ModelInfo


_SYNONYM_TASK_MAP = {
    "audio-ctc": "automatic-speech-recognition",
    "causal-lm": "text-generation",
    "causal-lm-with-past": "text-generation-with-past",
    "default": "feature-extraction",
    "default-with-past": "feature-extraction-with-past",
    "masked-lm": "fill-mask",
    "mask-generation": "feature-extraction",
    "sentence-similarity": "feature-extraction",
    "seq2seq-lm": "text2text-generation",
    "seq2seq-lm-with-past": "text2text-generation-with-past",
    "sequence-classification": "text-classification",
    "speech2seq-lm": "automatic-speech-recognition",
    "speech2seq-lm-with-past": "automatic-speech-recognition-with-past",
    "summarization": "text2text-generation",
    "text-to-speech": "text-to-audio",
    "translation": "text2text-generation",
    "vision2seq-lm": "image-to-text",
    "zero-shot-classification": "text-classification",
    "image-feature-extraction": "feature-extraction",
    "pretraining": "feature-extraction",
    # for backward compatibility and testing (where
    # model task and model type are still the same)
    "stable-diffusion": "text-to-image",
    "stable-diffusion-xl": "text-to-image",
    "latent-consistency": "text-to-image",
}
_TRANSFORMERS_TASKS_TO_MODEL_LOADERS = {
    "audio-classification": "AutoModelForAudioClassification",
    "audio-frame-classification": "AutoModelForAudioFrameClassification",
    "audio-xvector": "AutoModelForAudioXVector",
    "automatic-speech-recognition": ("AutoModelForSpeechSeq2Seq", "AutoModelForCTC"),
    "depth-estimation": "AutoModelForDepthEstimation",
    "feature-extraction": "AutoModel",
    "fill-mask": "AutoModelForMaskedLM",
    "image-classification": "AutoModelForImageClassification",
    "image-segmentation": (
        "AutoModelForImageSegmentation",
        "AutoModelForSemanticSegmentation",
        "AutoModelForInstanceSegmentation",
        "AutoModelForUniversalSegmentation",
    ),
    "image-to-image": "AutoModelForImageToImage",
    "image-to-text": ("AutoModelForVision2Seq", "AutoModel"),
    "mask-generation": "AutoModel",
    "masked-im": "AutoModelForMaskedImageModeling",
    "multiple-choice": "AutoModelForMultipleChoice",
    "object-detection": "AutoModelForObjectDetection",
    "question-answering": "AutoModelForQuestionAnswering",
    "reinforcement-learning": "AutoModel",
    "semantic-segmentation": "AutoModelForSemanticSegmentation",
    "text-to-audio": ("AutoModelForTextToSpectrogram", "AutoModelForTextToWaveform"),
    "text-generation": "AutoModelForCausalLM",
    "text2text-generation": "AutoModelForSeq2SeqLM",
    "text-classification": "AutoModelForSequenceClassification",
    "token-classification": "AutoModelForTokenClassification",
    "visual-question-answering": "AutoModelForVisualQuestionAnswering",
    "zero-shot-image-classification": "AutoModelForZeroShotImageClassification",
    "zero-shot-object-detection": "AutoModelForZeroShotObjectDetection",
}

def map_from_synonym(pipeline_tag: str) -> str:
    return _SYNONYM_TASK_MAP.get(pipeline_tag, pipeline_tag)


def infer_transformers_task_type(model_info: ModelInfo) -> str:
    pipeline_tag = model_info.pipeline_tag
    transformers_info = model_info.transformersInfo
    if pipeline_tag is not None:
        inferred_task_name = map_from_synonym(model_info.pipeline_tag)
    elif transformers_info is not None:
        transformers_pipeline_tag = transformers_info.get("pipeline_tag", None)
        transformers_auto_model = transformers_info.get("auto_model", None)
        if transformers_pipeline_tag is not None:
            pipeline_tag = transformers_info["pipeline_tag"]
            inferred_task_name = map_from_synonym(pipeline_tag)
        elif transformers_auto_model is not None:
            transformers_auto_model = transformers_auto_model.replace("TF", "")
            for task_name, model_loaders in _TRANSFORMERS_TASKS_TO_MODEL_LOADERS.items():
                if isinstance(model_loaders, str):
                    model_loaders = (model_loaders,)
                for model_loader_class_name in model_loaders:
                    if transformers_auto_model == model_loader_class_name:
                        inferred_task_name = task_name
                        break
                if inferred_task_name is not None:
                    break
    else:
        raise ValueError(f"No task type found for {model_info}")
    return inferred_task_name
