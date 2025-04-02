# Fun3DU : Functional Understanding and Segmentation in 3D Scenes
Official implementation and [website](https://tev-fbk.github.io/fun3du/) of Fun3DU, a novel method for functional understanding and segmentation in 3D scenes.
The technical report is available [on arxiv](https://arxiv.org/abs/2411.16310).

## News
- 02/04/25: Code for Fun3DU has been released.
- 27/02/25: Fun3DU has been accepted at CVPR25!


## Dataset preparation
NB: the split0 and split1 splits mentioned in the paper correspond to the train and validation split of SceneFun3D, respectively.

1. Create dataset root folder `$ROOT` (the dataset scripts assume it to be `data/scenefun3d/`)
2. Download the original [dataset folder](https://github.com/SceneFun3D/scenefun3d/tree/main/benchmark_file_lists) and put in the `$ROOT`.
3. Create the two splits (SceneFun3D provides them as a single file), by running the following scripts:
```
python scripts/make_video_list.py train
python scripts/make_video_list.py val
```
4. Download the splits with the following scripts:
```
python scripts/sun3d/data_asset_download.py --split custom --video_id_csv $ROOT/benchmark_file_lists/val_set.csv --download_dir $ROOT/val --dataset_asset laser_scan_5mm crop_mask annotations descriptions hires_wide hires_wide_intrinsics hires_depth hires_poses

python scripts/sun3d/data_asset_download.py --split custom --video_id_csv $ROOT/benchmark_file_lists/train_set.csv --download_dir $ROOT/train --dataset_asset laser_scan_5mm crop_mask annotations descriptions hires_wide hires_wide_intrinsics hires_depth hires_poses
```

## Setup environment

TODO.
`requirements.txt` show a partial list of the required packages.

## Running Fun3DU

Fun3DU is divided in 4 scripts, each executing one of the 4 steps mentioned in the paper (LLM preprocessing, Context object segmentation, Functional object segmentation, and multi-view agreement).
The intermediate results will be saved in the dataset folder for the LLM processing and the contextual object segmentation steps, and in a specific experiment folder for the remaining two steps. This allows to try the method with various configurations without rerunning all steps.

### LLM preprocessing (LLama3.1)
This will generate a .json file for each visit, containing the results of the LLM reasoning on each description.
1. Download ollama from here https://ollama.com
2. install the python library `pip install ollama`
3. Run `ollama serve` to start the ollama backend (it should start the backend on `localhost:11434`)
4. In another window, run `ollama pull llama3.1` to download Llama
5. If the port is not 11434, update `OLLAMA_PORT` to the correct port in `run_llm.py` (line 8).
6. Run `python run_llm.py dataset.root=$ROOT dataset.split=val llm_type=llama`

### Contextual object segmentation (Owl2+RSAM)
Run the following script to save the predicted masks for all contextual object in a scene:
```
python run_detection.py dataset.root=$ROOT dataset.split=val llm_type=llama mask_type=standard
```

### Functional object segmentation (Molmo)
Choose an exp_root to save the experiments intermediate results (`exps` by default).
Run the following script to save the predicted masks for functional objects in a scene:
```
python run_molmo.py dataset.root=$ROOT dataset.split=val llm_type=llama mask_type=standard exp_name=$EXP
```
By default, intermediate results will be saved in exps/$EXP/frames. This can be changed in the config.

### Multi-view agreement
Run the following script to lift the predicted masks from the previous step and produce a point cloud:
```
python run_lifting.py dataset.root=$ROOT dataset.split=val mask_type=standard exp_name=$EXP
```
By default, point cloud data will be saved in exps/$EXP/pcds. This can be changed in the config.

### Evaluation
Run the following script to evaluate an experiment with a fixed threshold:
```
python evaluate.py dataset.root=$ROOT dataset.split=val exp_name=$EXP threshold=0.7
```
By default, this will evaulate with point cloud data in exps/$EXP/pcds. This can be changed in the config.

## Citing Fun3DU

If you find Fun3DU useful for your work, consider citing it:
```
  @inproceedings{corsetti2025fun3du,
    title={Functionality understanding and segmentation in 3D scenes},
    author={Corsetti, Jaime and Giuliari, Francesco and Fasoli, Alice and Boscaini, Davide and Poiesi, Fabio},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2025}
  }
```


## Acknowledgements

We thank the authors of [SceneFun3D](https://scenefun3d.github.io/documentation/) for the dataset toolkit, on which our implementation is based.
This work was supported by the European Union’s Horizon Europe research and innovation programme under grant agreement No 101058589 (AI-PRISM). We also acknowledge ISCRA for awarding this project access to the LEONARDO supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CINECA (Italy).


## Website License
The website template is from [Nerfies](https://github.com/nerfies/nerfies.github.io).

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
