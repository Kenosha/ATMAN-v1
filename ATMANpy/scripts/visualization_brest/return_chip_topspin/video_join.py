

import os
import subprocess

from ATMANpy.funcs.util_funcs import get_folder_struct, maybe_make_folder

PROJECT_NAME = 'return_chip_topspin'
VIDEOS = ['intro_20241103_165952', 'descriptive_20241105_200855', 'inference_20241109_191055']

IDX = 2
VIDEO = VIDEOS[IDX]

### Paths
result_path = os.path.join(get_folder_struct('results_video')[PROJECT_NAME], VIDEO)
segments_joint_path = os.path.join(result_path, 'segments_joint')
all_joint_path = maybe_make_folder(result_path, 'out')
txt_file = os.path.join(all_joint_path, 'joinlist.txt')
out_file = os.path.join(all_joint_path, 'out.mp4')

### sort segments
def sort_key(segStr):
    return int(segStr.replace('.mp4', '').split('_')[1])
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
subprocess.run(f'ffmpeg -f concat -safe 0 -i {txt_file} -c copy {out_file}')

