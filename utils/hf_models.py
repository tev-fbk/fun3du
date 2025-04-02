import math
import re
from typing import List, Optional, Tuple

import numpy as np
import torch
from numpy import array
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskGeneration,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    GenerationConfig,
    Owlv2ForObjectDetection,
    SamModel,
    SamProcessor,
)

from utils.io import pad_detection_predictions
from utils.misc import mask_to_box


def init_molmo():

    processor = AutoProcessor.from_pretrained(
        "allenai/Molmo-7B-D-0924",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # load the model
    model = AutoModelForCausalLM.from_pretrained(
        "allenai/Molmo-7B-D-0924",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, processor


def inference_molmo(molmo_model, molmo_processor, img, prompt) -> str:
    """
    Only supports single batch inference for now
    """

    # process the image and text
    inputs = molmo_processor.process(images=[img], text=prompt)
    # move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(molmo_model.device).unsqueeze(0) for k, v in inputs.items()}
    # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
    inputs["images"] = inputs["images"].to(torch.bfloat16)
    output = molmo_model.generate_from_batch(
        inputs,
        GenerationConfig(
            max_new_tokens=200,
            stop_strings="<|endoftext|>",
            temperature=1.0,
            do_sample=True,
        ),
        tokenizer=molmo_processor.tokenizer,
    )

    generated_tokens = output[0, inputs["input_ids"].size(1) :]
    generated_text = molmo_processor.tokenizer.decode(
        generated_tokens, skip_special_tokens=True
    )

    return generated_text


def _inference_molmo_batched(
    molmo_model, molmo_processor, imgs: List[array], prompt: str
) -> List[str]:
    """
    Batch inference (N images, one prompt), used as wrapper for inference_molmo_batched
    """
    # process the image and text
    BS = len(imgs)
    text_inputs = molmo_processor.process(images=[imgs[0]], text=prompt)
    text_list = list()

    all_inputs = {
        "input_ids": text_inputs["input_ids"]
        .unsqueeze(0)
        .repeat(BS, 1)
        .to(molmo_model.device),
        "images": list(),
        "image_input_idx": list(),
        "image_masks": list(),
    }

    for img in imgs:
        img_inputs = molmo_processor.process(images=[img], text=prompt)
        all_inputs["images"].append(img_inputs["images"])
        all_inputs["image_input_idx"].append(img_inputs["image_input_idx"])
        all_inputs["image_masks"].append(img_inputs["image_masks"])

    for k in all_inputs.keys():
        if "image" in k:
            all_inputs[k] = torch.stack(all_inputs[k], dim=0).to(molmo_model.device)

    # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
    output = molmo_model.generate_from_batch(
        all_inputs,
        GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
        tokenizer=molmo_processor.tokenizer,
    )

    for i in range(len(imgs)):
        generated_tokens = output[i, all_inputs["input_ids"].size(1) :]
        generated_text = molmo_processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )
        text_list.append(generated_text)

    return text_list


def inference_molmo_batched(
    molmo_model, molmo_processor, imgs: List[array], prompt: str, batch: int
) -> str:
    """
    Batch inference (N images, one prompt)
    """
    BS = len(imgs)

    # batching needed!
    if BS > batch:

        NB = math.ceil(BS / batch)
        text_list = list()
        for i_b in range(NB):

            imgs_i = imgs[i_b * NB : (i_b + 1) * NB]
            text_list.extend(
                _inference_molmo_batched(molmo_model, molmo_processor, imgs_i, prompt)
            )
    else:
        text_list = _inference_molmo_batched(molmo_model, molmo_processor, imgs, prompt)

    return text_list


def extract_points(molmo_output: str, size: Tuple[int, int]) -> np.array:
    """
    Obtained from https://huggingface.co/allenai/Molmo-7B-O-0924/discussions/1
    """
    image_w, image_h = size
    all_points = []
    for match in re.finditer(
        r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"',
        molmo_output,
    ):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            point = np.array(point)
            if np.max(point) > 100:
                # Treat as an invalid output
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            points = all_points.append(point)
    if len(all_points) > 0:
        points = np.stack(all_points, axis=0)
    else:
        points = None
    return points


def init_sam_model(device: str):

    device = torch.device(device)
    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    return model, processor


def process_sam_features(sam_model, sam_processor, imgs, batch: Optional[int] = 50):
    """
    Return SAM features by given mask.
    If all_masks = True, returns all masks hierarchy, otherwise only the first
    """

    # must have fake prompt to extract features. Does not influence output
    fake_boxes = [[[100, 100, 200, 200]] for _ in imgs]

    if len(imgs) > batch:

        emb_list = list()
        NB = math.ceil(len(imgs) / batch)

        for i_b in range(NB):

            inputs = sam_processor(
                imgs[i_b * batch : (i_b + 1) * batch],
                input_boxes=fake_boxes[i_b * batch : (i_b + 1) * batch],
                return_tensors="pt",
            ).to(sam_model.device)
            with torch.no_grad():
                image_embeddings = sam_model.get_image_embeddings(
                    inputs["pixel_values"]
                )
            emb_list.append(image_embeddings.cpu().to(torch.float16))

        emb_list = torch.cat(emb_list, dim=0)

    else:
        inputs = sam_processor(imgs, input_boxes=fake_boxes, return_tensors="pt").to(
            sam_model.device
        )
        with torch.no_grad():
            emb_list = sam_model.get_image_embeddings(inputs["pixel_values"])
        emb_list = emb_list.cpu().to(torch.float16)

    return emb_list


def process_sam_prompts(
    sam_model,
    sam_processor,
    img,
    points: np.array,
    batch: Optional[int] = 50,
):
    """
    Return SAM masks by given prompts, given as a list of 2D points (XY).
    If all_masks = True, returns all masks hierarchy, otherwise only the first
    """

    input_points = points.tolist()
    inputs = sam_processor(img, input_points=input_points, return_tensors="pt").to(
        sam_model.device
    )
    image_embeddings = sam_model.get_image_embeddings(inputs["pixel_values"])

    # pop the pixel_values as they are not neded
    inputs.pop("pixel_values", None)
    inputs.update({"image_embeddings": image_embeddings})
    inputs["input_points"] = inputs["input_points"].unsqueeze(0)

    n_points = inputs["input_points"].shape[1]

    if n_points > batch:

        masks = list()
        scores = list()
        NB = math.ceil(n_points / batch)
        copied_points = inputs["input_points"].clone()
        for i_b in range(NB):
            inputs["input_points"] = copied_points[
                :, i_b * batch : (i_b + 1) * batch, :, :
            ]
            with torch.no_grad():
                outputs = sam_model(**inputs)

            masks_i = sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )[0]
            scores_i = outputs.iou_scores[0]
            top_scores_idxs = torch.argmax(scores_i, dim=1).to(masks_i.device)
            idxs = torch.arange(0, masks_i.shape[0]).to(masks_i.device)
            masks.append(masks_i[idxs, top_scores_idxs])

        masks = torch.concatenate(masks, dim=0)

    else:

        with torch.no_grad():
            outputs = sam_model(**inputs)

        masks = sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]
        scores = outputs.iou_scores[0]
        top_scores_idxs = torch.argmax(scores, dim=1).to(masks.device)
        idxs = torch.arange(0, masks.shape[0]).to(masks.device)
        masks = masks[idxs, top_scores_idxs]

    return masks


def init_detection(device):

    # text_config = {'max_position_embeddings':128}
    owl_model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble",  # ,config=Owlv2Config(text_config)
    ).to(device)

    owl_processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
    sam_model = SamModel.from_pretrained("jadechoghari/robustsam-vit-large").to(device)
    sam_processor = SamProcessor.from_pretrained("jadechoghari/robustsam-vit-large")
    return owl_model, owl_processor, sam_model, sam_processor


def init_gsam_detection(device):

    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-tiny"
    ).to(device)
    gdino_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")

    sam_model = AutoModelForMaskGeneration.from_pretrained("facebook/sam-vit-base").to(
        device
    )
    sam_proc = AutoProcessor.from_pretrained("facebook/sam-vit-base")

    return gdino_model, gdino_proc, sam_model, sam_proc


def process_gsam_detection(gdino_m, gdino_p, sam_m, sam_p, images, prompts) -> dict:
    """
    Expects a list of PIL images and a list of prompts. Each prompt list is multiplied and applied to all images!
    Batch size shold be decided a priori.
    """

    sizes = [image.size[::-1] for image in images]
    form_prompts = " . ".join(prompts) + "."
    all_prompts = [form_prompts for i in range(len(images))]
    # owl_prompts = tuple([prompts for _ in range(len(images))])
    inputs = gdino_p(
        text=all_prompts, images=images, return_tensors="pt", padding=True
    ).to(gdino_m.device)

    with torch.no_grad():
        outputs = gdino_m(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.pixel_values,
        )

    # Convert outputs (bounding boxes and class logits) to final bounding boxes and scores
    results = gdino_p.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=sizes,
    )

    all_scores, all_boxes, all_labels = list(), list(), list()
    for res in results:
        all_scores.append(res["scores"].cpu().numpy())
        all_boxes.append(res["boxes"].cpu().numpy())
        all_labels.append(res["labels"])

    all_boxes, all_scores, all_labels = pad_detection_predictions(
        all_boxes, all_scores, all_labels
    )

    # this may happen in empty frames
    if all_boxes.shape[1] > 0:
        all_boxes = all_boxes.tolist()

        inputs = sam_p(images, input_boxes=all_boxes, return_tensors="pt").to(
            sam_m.device
        )

        with torch.no_grad():
            outputs = sam_m(**inputs)

        masks = sam_p.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )

        mask_data = list()
        # iterate on batch
        for i, (mask_i, labs_i, scores_i) in enumerate(
            zip(masks, all_labels, all_scores)
        ):
            # iterate on masks within an image
            mask_list, label_list, score_list = list(), list(), list()
            for j, (mask_j, lab_j, score_j) in enumerate(zip(mask_i, labs_i, scores_i)):
                if lab_j != "empty":
                    mask_list.append(mask_j[0].numpy().astype(np.uint8))
                    score_list.append(score_j)
                    label_list.append(lab_j)
                    # viz_2d_mask(np.asarray(images[i]), mask_j[0].numpy(), lab_j, f'viz/img_{i}_{j}.png')
            mask_list = np.stack(mask_list, axis=0) if len(mask_list) > 0 else []
            score_list = np.stack(score_list, axis=0) if len(score_list) > 0 else []
            label_list = np.stack(label_list, axis=0) if len(label_list) > 0 else []
            mask_data.append(
                {
                    "masks": mask_list,
                    "scores": score_list,
                    "labels": label_list,
                }
            )

    else:
        # append empty list!
        mask_data = list()
        for res in results:
            mask_data.append({"masks": [], "scores": [], "labels": []})

    return mask_data


def process_detection(owl_m, owl_p, sam_m, sam_p, images, prompts) -> dict:
    """
    Expects a list of PIL images and a list of prompts. Each prompt list is multiplied and applied to all images!
    Batch size shold be decided a priori.
    """

    sizes = [image.size[::-1] for image in images]
    owl_prompts = [[p for p in prompts] for _ in range(len(images))]
    inputs = owl_p(text=owl_prompts, images=images, return_tensors="pt").to(
        owl_m.device
    )
    if inputs.input_ids.shape[1] > 16:
        new_ids = list()
        for input_id in inputs.input_ids:
            # enough to cut it
            if input_id.argmax(dim=0) >= 16:
                input_id[15] = 49407
            new_ids.append(input_id[:16])

        new_ids = torch.stack(new_ids, dim=0).to(inputs.input_ids.device)
        inputs.input_ids = new_ids
        inputs.attention_mask = inputs.attention_mask[:, :16]

    # forward pass
    with torch.no_grad():
        outputs = owl_m(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.pixel_values,
        )

    # Convert outputs (bounding boxes and class logits) to final bounding boxes and scores
    results = owl_p.post_process_object_detection(
        outputs=outputs, threshold=0.2, target_sizes=sizes
    )

    all_scores, all_boxes, all_labels = list(), list(), list()
    for res in results:
        all_scores.append(res["scores"].cpu().numpy())
        all_boxes.append(res["boxes"].cpu().numpy())
        all_labels.append([prompts[i.cpu().item()] for i in res["labels"]])

    all_boxes, all_scores, all_labels = pad_detection_predictions(
        all_boxes, all_scores, all_labels
    )

    # this may happen in empty frames
    if all_boxes.shape[1] > 0:
        all_boxes = all_boxes.tolist()

        inputs = sam_p(images, input_boxes=all_boxes, return_tensors="pt").to(
            sam_m.device
        )

        with torch.no_grad():
            outputs = sam_m(**inputs)

        masks = sam_p.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )

        mask_data = list()
        # iterate on batch
        for i, (mask_i, labs_i, scores_i) in enumerate(
            zip(masks, all_labels, all_scores)
        ):
            # iterate on masks within an image
            mask_list, label_list, score_list = list(), list(), list()
            for j, (mask_j, lab_j, score_j) in enumerate(zip(mask_i, labs_i, scores_i)):
                if lab_j != "empty":
                    mask_list.append(mask_j[0].numpy().astype(np.uint8))
                    score_list.append(score_j)
                    label_list.append(lab_j)
                    # viz_2d_mask(np.asarray(images[i]), mask_j[0].numpy(), lab_j, f'viz/img_{i}_{j}.png')
            mask_list = np.stack(mask_list, axis=0) if len(mask_list) > 0 else []
            score_list = np.stack(score_list, axis=0) if len(score_list) > 0 else []
            label_list = np.stack(label_list, axis=0) if len(label_list) > 0 else []
            mask_data.append(
                {
                    "masks": mask_list,
                    "scores": score_list,
                    "labels": label_list,
                }
            )

    else:
        # append empty list!
        mask_data = list()
        for res in results:
            mask_data.append({"masks": [], "scores": [], "labels": []})

    return mask_data
