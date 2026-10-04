Starting from intermediate_anns and intermediate_videos alredy processed, to generate final annotations run in the following order:
1. build_ovr_kinetics_annotations.py

To extract audio tracks, run:
2. extract_valid_audio.py


The number of annotations claimed in the paper should be: train+val -> 2230 among 1111 videos, test -> 7563 among 3760 videos.
The number of final annotations considered here, after filtering not found videos and exact duplicate annotations, is: train+val -> 2100 among 1058 videos, test -> 7110 among 3577 videos

NOTE: I found that 95% of the video in val split are also in the train split

NOTE: Videos are heavily untrimmed

NOTE: Repetition intervals are created dividing uniformerly the interval