# The VoxBlink2 Dataset

The VoxBlink2 dataset is a Large Scale speaker recognition dataset with 100K+ speakers obtained from YouTube platform. This repository provides guidelines to build the corpus and relative resources to reproduce the results in our article . For more introduction, please see [cite](https://VoxBlink2.github.io). 

## Resource 
Let's start with obtaining the [resource](https://drive.google.com/drive/folders/1lzumPsnl5yEaMP9g2bFbSKINLZ-QRJVP?usp=sharing) files and decompressing tar-files.
```bash
tar -zxvf spk_info.tar.gz
tar -zxvf vb2_meta.tar.gz 
```

## File structure


```
% The file structure is summarized as follows: 
|---- data               
|     |---- ossi    # [Folder]evaluation protocols for open-set speaker identification
|     |---- test_vox # [Folder] evaluation protocols for speaker verification
|     |---- spk2videos	# [spk,video1,video2,...]
|---- ckpt #checkpoints for evaluation
|     |---- ecapatdnn # [Folder]
|     |---- resnet34 # [Folder]
|     |---- resnet100 # [Folder]
|     |---- resnet293 # [Folder]
|---- spk_info             # video'tags of speakers：
|     |---- id000000	
|     |---- id000001	
|     |---- ...
|---- meta		# timestamps for video/audio cropping
|     |---- id000000	# spkid
|           |---- DwgYRqnQZHM	#videoid
|                 |---- 00000.txt	#uttid
|                 |---- ...
|           |---- ... 
|     |---- ...	
|---- ossi            # video'tags of speakers：
|     |---- eval.py # recipe for evaluate openset speaker identification
|     |---- utils.py 
|     |---- example.npy # eg. Resnet34-based embedding for evaluate OSSI 
|---- audio_cropper.py	# extract audio-only segments by timestamps from downloaded audios
|---- video_cropper.py	# extract audio-visual segments by timestamps from downloaded videos
|---- downloader.py	# scripts for download videos
|---- LICENSE		# license
|---- README.md	
|---- requirement.txt			

```
## Download
The following procedures show how to construct your VoxBlink2
### Pre-requisites
1. Install **ffmpeg**:
```bash
sudo apt-get update && sudo apt-get upgrade
sudo apt-get install ffmpeg
```
2. Install Python library:
```bash
pip install -r requirements.txt
```

3. Download videos

We provide two alternatives for you to download video or audio-only segments. We Also leverage multi-thread to facilate download process.

* For Audio-Visual
```python
python downloader.py --base_dir ${BASE_DIR} --num_workers 4 --mode audio
```
* For Audio-Only
```python
python downloader.py --base_dir ${BASE_DIR} --num_workers 4 --mode audio
```

4. Crop Audio/Videos
* For Audio-Visual
```python
python cropper_video.py --save_dir_audio ${SAVE_PATH_AUDIO} --save_dir_video ${SAVE_PATH_VIDEO} --timestamp_path meta --video_root=${BASE_DIR} --num_workers 4
```
* For Audio-Only
```python
python cropper_audio.py --save_dir ${SAVE_PATH_AUDIO} --timestamp_path meta --audio_root=${BASE_DIR} --num_workers 4
```

## SV Evaluation

We provide simple scripts for model evaluation of ASV, just execute `run_eval.sh` in `asv` folder. For more, please look at [asv](https://github.com/VoxBlink2/ScriptsForVoxBlink2/tree/main/asv).

## Open-Set Speaker Identification Evaluation
We provide simple scripts for model evaluation of our proposed task: Open-Set Speaker-Identification(OSSI). just execute `run_eval_ossi.sh` in `ossi` folder. For more, please look at [ossi](https://github.com/VoxBlink2/ScriptsForVoxBlink2/tree/main/ossi).

## License

The dataset is licensed under the **CC BY-NC-SA 4.0** license. This means that you can share and adapt the dataset for non-commercial purposes as long as you provide appropriate attribution and distribute your contributions under the same license. Detailed terms can be found [here](LICENSE).

Important Note: Our released dataset only contains annotation data, including the YouTube links, time stamps and speaker labels. We do not release audio or visual data and it is the user's responsibility to decide whether and how to download the video data and whether their intended purpose with the downloaded data is legal in their country. For YouTube users with concerns regarding their videos' inclusion in our dataset, please contact us via E-mail: yuke.lin@dukekunshan.edu.cn or ming.li369@dukekunshan.edu.cn.




## Citation

Please cite the paper below if you make use of the dataset:

```
@INPROCEEDINGS{10446780,
  author={Lin, Yuke and Qin, Xiaoyi and Zhao, Guoqing and Cheng, Ming and Jiang, Ning and Wu, Haiying and Li, Ming},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Voxblink: A Large Scale Speaker Verification Dataset on Camera}, 
  year={2024},
  volume={},
  number={},
  pages={10271-10275},
  keywords={Training;Video on demand;Purification;Pipelines;Signal processing;Web sites;Noise measurement;Speaker Verification;Dataset;Large-scale;Multi-modal},
  doi={10.1109/ICASSP48485.2024.10446780}}
```