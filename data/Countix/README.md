To generate final annotations run in the following order:
1. get_countix.py
2. download_videos.py
3. parse_orig_annotations.py
4. build_final_annotations.py
5. add_final_columns.py

To extract audio tracks, run:
2. extract_countix_audio.py

The number of videos claimed in the paper for train, val, and test is 4588, 1450, 2719. After preprocessing and filtering we have 4118, 1432, 2564 videos.

NOTE: Videos are heavily untrimmed in a way similar to OVR

NOTE: Repetition intervals are created dividing uniformerly the interval