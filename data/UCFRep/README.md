To generate final annotations run in the following order:
1. build_ucfrep_annotations.py

To extract audio tracks, run:
2. extract_ucfrep_audio.py


The number of videos claimed in the paper is 526. After preprocessing and filtering we have 526 videos.

NOTE: Videos are untrimmed, even if in general the repetition sequence covers most of the video (not like OVR for instance). Another difference compared to OVR is that here in any case there is just one action in the whole video.

NOTE: Repetition are annotated precisely, i.e., it is not a uniform division of the time interval but they are manually annotated