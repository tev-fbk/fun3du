# Based on https://github.com/apple/ARKitScenes/blob/main/download_data.py
# Adapted by Ayca Takmaz, July 2023

import argparse
import math
import os
import subprocess

import pandas as pd

ARkitscense_url = (
    "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1"
)
TRAINING = "Training"
VALIDATION = "Validation"
HIGRES_DEPTH_ASSET_NAME = "highres_depth"

default_raw_dataset_assets = [
    "mov",
    "annotation",
    "mesh",
    "confidence",
    "lowres_depth",
    "lowres_wide.traj",
    "lowres_wide",
    "lowres_wide_intrinsics",
    "ultrawide",
    "ultrawide_intrinsics",
    "vga_wide",
    "vga_wide_intrinsics",
    "wide",
    "wide_intrinsics",
]

missing_3dod_assets_video_ids = [
    "47334522",
    "47334523",
    "42897421",
    "45261582",
    "47333152",
    "47333155",
    "48458535",
    "48018733",
    "47429677",
    "48458541",
    "42897848",
    "47895482",
    "47333960",
    "47430089",
    "42899148",
    "42897612",
    "42899153",
    "42446164",
    "48018149",
    "47332198",
    "47334515",
    "45663223",
    "45663226",
    "45663227",
]


def raw_files(video_id, assets, metadata):
    file_names = []
    for asset in assets:
        if HIGRES_DEPTH_ASSET_NAME == asset:
            in_upsampling = metadata.loc[
                metadata["video_id"] == float(video_id), ["is_in_upsampling"]
            ].iat[0, 0]
            if not in_upsampling:
                print(
                    f"Skipping asset {asset} for video_id {video_id} - Video not in upsampling dataset"
                )
                continue  # highres_depth asset only available for video ids from upsampling dataset

        if asset in [
            "confidence",
            "highres_depth",
            "lowres_depth",
            "lowres_wide",
            "lowres_wide_intrinsics",
            "ultrawide",
            "ultrawide_intrinsics",
            "vga_wide",
            "vga_wide_intrinsics",
            "wide",
            "wide_intrinsics",
        ]:
            file_names.append(asset + ".zip")
        elif asset == "mov":
            file_names.append(f"{video_id}.mov")
        elif asset == "mesh":
            if video_id not in missing_3dod_assets_video_ids:
                file_names.append(f"{video_id}_3dod_mesh.ply")
        elif asset == "annotation":
            if video_id not in missing_3dod_assets_video_ids:
                file_names.append(f"{video_id}_3dod_annotation.json")
        elif asset == "lowres_wide.traj":
            if video_id not in missing_3dod_assets_video_ids:
                file_names.append("lowres_wide.traj")
        else:
            raise Exception(f"No asset = {asset} in raw dataset")
    return file_names


def download_file(url, file_name, dst):
    os.makedirs(dst, exist_ok=True)
    filepath = os.path.join(dst, file_name)

    if not os.path.isfile(filepath):
        command = f"curl {url} -o {file_name}.tmp --fail"
        print(f"Downloading file {filepath}")
        try:
            subprocess.check_call(command, shell=True, cwd=dst)
        except Exception as error:
            print(f"Error downloading {url}, error: {error}")
            return False
        os.rename(filepath + ".tmp", filepath)
    else:
        print(f"WARNING: skipping download of existing file: {filepath}")
    return True


def unzip_file(file_name, dst):
    filepath = os.path.join(dst, file_name)
    print(f"Unzipping zip file {filepath}")
    command = f"unzip -oq {filepath} -d {dst}"
    try:
        subprocess.check_call(command, shell=True)
    except Exception as error:
        print(f"Error unzipping {filepath}, error: {error}")
        return False
    os.remove(filepath)
    return True


def get_metadata(dataset, download_dir, destination_folder):
    filename = "metadata.csv"
    url = (
        f"{ARkitscense_url}/threedod/{filename}"
        if "3dod" == dataset
        else f"{ARkitscense_url}/{dataset}/{filename}"
    )
    dst_folder = os.path.join(download_dir, destination_folder)
    dst_file = os.path.join(dst_folder, filename)

    if not download_file(url, filename, dst_folder):
        return

    metadata = pd.read_csv(dst_file)
    return metadata


def download_data(
    dataset,
    destination_folder,
    video_ids,
    dataset_splits,
    download_dir,
    raw_dataset_assets,
):
    metadata = get_metadata(dataset, download_dir, destination_folder)
    if None is metadata:
        print(f"Error retrieving metadata for dataset {dataset}")
        return

    download_dir = os.path.abspath(download_dir)
    for video_id in sorted(set(video_ids)):
        split = dataset_splits[video_ids.index(video_id)]
        dst_dir = os.path.join(download_dir, destination_folder)
        if dataset == "raw":
            url_prefix = ""
            file_names = []
            if not raw_dataset_assets:
                print(f"Warning: No raw assets given for video id {video_id}")
            else:
                dst_dir = os.path.join(dst_dir, str(video_id))
                url_prefix = f"{ARkitscense_url}/raw/{split}/{video_id}" + "/{}"
                file_names = raw_files(video_id, raw_dataset_assets, metadata)
        elif dataset == "3dod":
            url_prefix = f"{ARkitscense_url}/threedod/{split}" + "/{}"
            file_names = [
                f"{video_id}.zip",
            ]
        elif dataset == "upsampling":
            url_prefix = f"{ARkitscense_url}/upsampling/{split}" + "/{}"
            file_names = [
                f"{video_id}.zip",
            ]
        else:
            raise Exception(f"No such dataset = {dataset}")

        for file_name in file_names:
            dst_path = os.path.join(dst_dir, file_name)
            url = url_prefix.format(file_name)

            if not file_name.endswith(".zip") or not os.path.isdir(
                dst_path[: -len(".zip")]
            ):
                download_file(url, dst_path, dst_dir)
            else:
                print(f"WARNING: skipping download of existing zip file: {dst_path}")
            if file_name.endswith(".zip") and os.path.isfile(dst_path):
                unzip_file(file_name, dst_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_type",
        choices=[
            "challenge_development_set",
            "challenge_test_set",
            "full_training_set",
            "full_training_set_w_wide_assets",
        ],
        default="challenge_development_set",
    )

    parser.add_argument(
        "--download_dir",
        default="data",
    )

    args = parser.parse_args()

    dataset = "raw"
    if args.data_type == "challenge_development_set":
        video_id_csv = "benchmark_file_lists/challenge_development_scenes.csv"
        destination_folder = "ChallengeDevelopmentSet"
    elif args.data_type == "challenge_test_set":
        video_id_csv = "benchmark_file_lists/challenge_test_scenes.csv"
        destination_folder = "ChallengeTestSet"
    elif args.data_type == "full_training_set":
        video_id_csv = "benchmark_file_lists/full_training_scenes.csv"
        destination_folder = "FullTrainingSet"
    elif args.data_type == "full_training_set_w_wide_assets":
        video_id_csv = "benchmark_file_lists/full_training_scenes_w_wide_assets.csv"
        destination_folder = "FullTrainingSetWithWideAssets"
    else:
        raise Exception(f"No such data_type = {args.data_type}")

    raw_dataset_assets = [
        "lowres_depth",
        "lowres_wide.traj",
        "lowres_wide",
        "lowres_wide_intrinsics",
        "wide",
        "wide_intrinsics",
        "mesh",
    ]

    df = pd.read_csv(video_id_csv)
    video_ids_ = df["video_id"].to_list()
    video_ids_ = list(map(str, video_ids_))  # Expecting video id to be a string
    splits_ = df["fold"].to_list()

    download_data(
        dataset,
        destination_folder,
        video_ids_,
        splits_,
        args.download_dir,
        raw_dataset_assets,
    )
