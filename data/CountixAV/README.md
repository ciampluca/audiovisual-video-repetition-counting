Run in the following order:
1. get_countixav.py
2. download_missing_videos.py
3. parse_orig_annotations.py
4. build_final_annotations.py
5. add_final_columns.py

Audios are alredy provided.

The number of videos claimed in the paper for train, val, and test is 987, 311, 565. After preprocessing and filtering we have 934, 310, 551 videos.

NOTE: Videos are heavily untrimmed in a way similar to OVR

NOTE: Repetition intervals are created dividing uniformerly the interval