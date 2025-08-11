## 0. Downloading Datasets

- [MELD: Multimodal EmotionLines Dataset](https://affective-meld.github.io/)
- [IEMOCAP: interactive emotional dyadic motion capture database](https://sail.usc.edu/iemocap/)
- [MEDIC: A Multimodal Empathy Dataset in Counseling](https://ustc-ac.github.io/datasets/medic/)

## 1. Data Pre-processing

- For each video, use [FFmpeg](https://ffmpeg.org/) to extract the full audio track with `ffmpeg -i input_video.mp4 -vn -ar 16000 -ac 1 -c:a pcm_s16le output_audio.wav`.
- Utilized the [MFA](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html) tool to generate precise temporal boundaries. Input the audio track and its corresponding text to perform a forced alignment with `mfa align /path/to/your_corpus english_us_arpa english_us_arpa /path/to/output_folder`. Then, you will start and end timestamps for every word in the text.
- Use FFmpeg again, to segment the video file according to the alignment timestamps with `ffmpeg -i input_video.mp4 -ss time_start -to time_end -c copy output_clip.mp4`.
- Now, for evert utterance, you should get a strictly aligned (text, video_clip, audio_clip) triplet.
- Finally, extract key frames for each video clip with `ffmpeg -i input_video.mp4 -vf "select='eq(pict_type,I)'" -vsync vfr output_folder/keyframe-%03d.jpg` for later use.

## 2. Feature Extraction

```bash
cd ERGM

# install all required packages
pip install -r requirements.txt

python data_process/feature_extraction.py

# You may need to adjust the settings in the code to fit your environment.
```

## 3. Load Data


Use the `load_data.sh` script.
```shell
sh load_data.sh
```



After loading the data, several files will be generated in the data directory, with a structure like this:

```bash
data
├--gpt2
│   ├── train_utters.pickle
│   ├── train_ids.pickle
│   ├── valid_utters.pickle
│   └── valid_ids.pickle
```

## 4. Run

1. Use `train.sh` to train the model, you may change the settings in the script to fit your environment.

   ```bash
   sh train.sh
   ```

2. Use `infer.sh` to conduct inference with a trained model, you should specify  a trained checkpoint.

   ```bash
   sh infer.sh checkpoint_name
   ```

   