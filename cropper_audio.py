import subprocess
import os,cv2
import argparse, os, random
import multiprocessing as mp
from multiprocessing import Manager
import logging,sys
from itertools import islice
logging.basicConfig(level=logging.INFO)  # configure logging level to INFO
parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num_workers', 
                    default=1, 
                    type=int,
                    help="Multi-thread to facilate cropping process")
parser.add_argument('--save_dir', 
                    type=str,
                    help="where to save files")
parser.add_argument('--timestamp_path', 
                    type=str,
                    help="save path of timestamps")
parser.add_argument('--audio_root', 
                    type=str,
                    help="save path of audios")
args = parser.parse_args()


def split_dict(data, num_splits):
    keys = list(data.keys())
    random.shuffle(keys)
    split_keys = [keys[i::num_splits] for i in range(num_splits)]
    return [{k: data[k] for k in subset} for subset in split_keys]
def frame2second(frame,fps=25):
    return frame / fps
def prepare_timestamp(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        start_line = lines[7] if len(lines) >= 8 else None
        end_line = lines[-1] if len(lines) > 0 else None
        start_frame = start_line.strip().split("\t")[0]
        end_frame = end_line.strip().split("\t")[0]
        
    return frame2second(int(start_frame)),frame2second(int(end_frame))

def float_to_timecode(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:06.3f}"

def cut_audio(input_path, output_path, start_time, end_time):
    try:
        start_timecode = float_to_timecode(start_time)
        end_timecode = float_to_timecode(end_time)
        # exec ffmpeg command
        command = [
            'ffmpeg',
            '-y',
            '-i', input_path,
            '-ss', start_timecode,
            '-to', end_timecode,
            '-c', 'copy',  # 
            output_path
        ]
        # exec
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"failed extraction: {e}")

def crop_by_spks(spk2audios):
    for spk, audios in spk2audios.items():
        tsf_spk = os.path.join(args.timestamp_path,spk)
        if not os.path.exists(tsf_spk):
            logging.info(f"Spk dir {tsf_spk} of timestamp does not exist. Skipping...")
            continue
        for audio in audios:
            tsf_audio = os.path.join(tsf_spk, audio)
            audio_path = os.path.join(args.audio_root,spk,audio+'.wav')
            if not os.path.exists(tsf_audio):
                logging.info(f"audio dir {tsf_audio} of timestamp does not exist. Skipping...")
                continue
            if not os.path.exists(audio_path):
                logging.info(f"audio {audio_path} does not exist. Skipping...")
                continue
            os.makedirs(os.path.join(args.save_dir,spk,audio),exist_ok=True)
            for tsf in os.listdir(tsf_audio):
                num = tsf.replace(".txt","")
                ts = os.path.join(tsf_audio,tsf)
                start_time, end_time= prepare_timestamp(ts)
                save_audio_path = os.path.join(args.save_dir,spk,audio,num+".wav")
                cut_audio(audio_path,save_audio_path,start_time,end_time)                

if __name__ == '__main__':
    print("*"*15)
    print("* Cropping Starts *")
    print("*"*15)
    os.makedirs(args.save_dir,exist_ok=True)
    os.makedirs(os.path.join(args.save_dir,"audio"),exist_ok=True)
    spk2audios_loc = "resource/data/spk2videos"
    assert os.path.exists(spk2audios_loc)
    spk2audios = {line.split()[0]:line.strip().split()[1:] for line in open(spk2audios_loc)}
    workers = min(args.num_workers,len(spk2audios))
    spk2audios_slices = split_dict(spk2audios,workers)
    pool = mp.Pool(processes=workers)
    pool.map(crop_by_spks, spk2audios_slices)
    pool.close()
    pool.join() 

    