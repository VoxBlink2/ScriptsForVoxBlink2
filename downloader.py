import yt_dlp
from yt_dlp.utils import download_range_func
# -------------
import time, argparse, os, json
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Manager
import random,logging,sys
logging.basicConfig(level=logging.INFO)  # configure logging level to INFO

parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--base_dir', 
                    default='videos', 
                    type=str,
                    help="where to save video files")
parser.add_argument('--num_workers', 
                    default=1, 
                    type=int,
                    help="Multi-Process to facilate download process")
parser.add_argument('--mode', 
                    default='video', 
                    type=str,
                    help="Please Select your Download Mode, video or audio")  
args = parser.parse_args()
# Download from Ytb
def job_video(urls,spk):
    ydl_opts_video = {
        'format': 'bestvideo[height<=720]+bestaudio',
        'outtmpl':os.path.join(args.base_dir,spk,'%(id)s.%(ext)s'),
        'noplaylist': True,
        'ignoreerrors': True,
        'max_sleep_interval': 0.2,
        'verbose':True,
        'quiet':True,
        'download_ranges': download_range_func(None, [(0, 60)]),
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        },],
        'postprocessor_args': [
            '-ar', '16000',
            '-strict', '-2',
            '-async','1', '-r' ,'25'
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts_video) as ydl:
        error_code = ydl.download(urls)
    return error_code

def job_audio(urls,spk):
    ydl_opts_video = {
        'format': 'bestaudio/best',
        'outtmpl':os.path.join(args.base_dir,spk,'%(id)s.%(ext)s'),
        'noplaylist': True,
        'ignoreerrors': True,
        'max_sleep_interval': 0.2,
        'verbose':True,
        'quiet':True,
        'download_ranges': download_range_func(None, [(0,60)]),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192'
        },],
        'postprocessor_args': [
            '-ar', '16000',
        ],
        'prefer_ffmpeg': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts_video) as ydl:
        error_code = ydl.download(urls)
    return error_code




def split_dict(data, num_splits):
    keys = list(data.keys())
    random.shuffle(keys)
    split_keys = [keys[i::num_splits] for i in range(num_splits)]
    return [{k: data[k] for k in subset} for subset in split_keys]

def download_audios(spk2videos):
    for spk, videos in spk2videos.items():
        os.makedirs(os.path.join(args.base_dir,spk),exist_ok=True)
        err_codes = job_audio(videos,spk)

def download_videos(spk2videos):
    for spk, videos in spk2videos.items():
        os.makedirs(os.path.join(args.base_dir,spk),exist_ok=True)
        err_codes = job_video(videos,spk)

if __name__ == '__main__':
    
    print("*"*15)
    print("* Download Starts *")
    print("*"*15)
    os.makedirs(args.base_dir,exist_ok=True)
    spk2videos_loc = "data/spk2videos"
    if not os.path.exists(spk2videos_loc):
        logging.error("Video list not exist!!")
        sys.exit()
    # Load Videos
    spk2videos = {line.split()[0]:line.strip().split()[1:] for line in open(spk2videos_loc)}
    workers = min(args.num_workers,len(spk2videos))
    spk2videos_slices = split_dict(spk2videos,workers)
    pool = mp.Pool(processes=workers)
    assert args.mode in set(['audio','video'])
    if args.mode == 'audio':
        pool.map(download_audios, spk2videos_slices)
    elif args.mode == 'video':
        pool.map(download_videos,spk2videos_slices)
    else:
        raise TypeError()
    pool.close()
    pool.join() 
