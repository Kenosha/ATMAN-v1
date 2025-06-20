

import os
import subprocess

from ATMANpy.funcs.util_funcs import get_folder_struct, maybe_make_folder

SAMPLE_RATE = 44100
FRAME_RATE = 25

PROJECT_NAME = 'atman_capabilities'
VIDEOS = ['short_intro_20250620_024215', 'short_raster_20250607_022044']

PROJECT_NAME = 'atman_capabilities'
VIDEOS = ['lines_20250609_173938', 'players_20250606_013734', 'balls_20250606_014742', 
          'sound_20250609_132907', 'vision_20250606_021653', 'trajectory_20250607_183929', 
          'filtering_20250607_232416', 'visual_engine_20250616_233029']

PROJECT_NAME = 'return_chip_topspin'
VIDEOS = ['intro_20250618_143225', 'descriptive_20250618_144448', 'inference_20250618_223539']

PROJECT_NAME = 'return_chip_topspin'
VIDEOS = ['short_20250615_000557']

PROJECT_NAME = 'serve_pressure_new'
VIDEOS = ['first_part_20250617_151722', 'ad_side_example_20250616_135617', 'ad_side_cases_20250616_144314',
          'transition_ad_to_deuce_20250616_145410', 'deuce_side_cases_20250616_152224']

PROJECT_NAME = 'serve_pressure_new'
VIDEOS = ['short_20250617_141905']

IDX = 0
VIDEO = VIDEOS[IDX]

### Paths
result_path = os.path.join(get_folder_struct('results_video')[PROJECT_NAME], VIDEO)
segments_joint_path = os.path.join(result_path, 'segments_joint')
all_joint_path = maybe_make_folder(result_path, 'out')
txt_file = os.path.join(all_joint_path, 'joinlist.txt')
out_file = os.path.join(all_joint_path, 'out.mp4')

### sort segments
def sort_key(segStr):
    leading_int = int(segStr.replace('.mp4', '').split('_')[1])
    try:
        segStr.replace('.mp4', '').split('_')[2]
    except IndexError:
        offset = 0
    else:
        offset = 0.5
    return leading_int + offset
    
seg_files = sorted(os.listdir(segments_joint_path), key = sort_key)

### write the text file, used for joining
text_lines = []
for seg_file in seg_files:
    full_file_path = os.path.join(segments_joint_path, seg_file)
    line = f"file \'{full_file_path}\'\n"
    text_lines.append(line)

with open(txt_file, 'w') as f:  
    f.writelines(text_lines)
    
### join videos
subprocess.run(f'ffmpeg -f concat -safe 0 -i {txt_file} -c copy {out_file} -r {FRAME_RATE} -ar {SAMPLE_RATE}')

