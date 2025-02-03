

import os
import subprocess

from ATMANpy.funcs.util_funcs import get_folder_struct, maybe_make_folder, make_timestamp

PROJECT_NAME = 'return_chip_topspin'
VIDEOS = ['intro_20241103_165952', 'descriptive_20241105_200855', 'inference_20241109_191055']

### Paths
proj_path = get_folder_struct('results_video')[PROJECT_NAME]
save_path = maybe_make_folder(proj_path, f'all_{make_timestamp()}')
txt_file = os.path.join(save_path, 'joinlist.txt')
out_file = os.path.join(save_path, 'out.mp4')

### write the text file, used for joining
text_lines = []
for vid in VIDEOS:
    result_path = os.path.join(proj_path, vid)
    all_joint_path = os.path.join(result_path, 'out')
    full_file_path = os.path.join(all_joint_path, 'out.mp4')
    line = f"file \'{full_file_path}\'\n"
    text_lines.append(line)

with open(txt_file, 'w') as f:
    f.writelines(text_lines)
    
### join videos
subprocess.run(f'ffmpeg -f concat -safe 0 -i {txt_file} -c copy {out_file}')

