Starting from intermediate_anns and intermediate_videos alredy processed, to generate final annotations run in the following order:
1. build_ovr_ego4d_annotations.py

To extract audio tracks, run:
2. extract_ovr_ego4d_audio.py


The number of annotations claimed in the paper should be: train+val -> 70179 among 33370 videos, test -> 20409 among 8692 videos.
The number of final annotations considered here, after filtering not found videos and exact duplicate annotations, is: train+val -> 68835 among 32838 videos, test -> 18019 among 8580 videos

NOTE: I found that 95% of the video in val split are also in the train split

NOTE: There are a lot of videos without audio track

NOTE: Videos are heavily untrimmed

NOTE: Repetition intervals are created dividing uniformerly the interval