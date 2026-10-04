To generate final annotations run in the following order:
1. build_repcount_annotations.py

To extract audio tracks, run:
2. extract_repcount_audio.py


The number of videos claimed in the paper is 1041. After preprocessing and filtering we have 1037 videos.

NOTE: Only about 5% of videos have also audios

NOTE: Videos are untrimmed, even if in general the repetition sequence covers most of the video (not like OVR for instance); furthermore, sometimes the same actions is interrupted and then is considered again: this is also because repetitions are not created dividing uniformely the interval time (see below). Another difference compared to OVR is that here in any case there is just one action in the whole video.

NOTE: Repetition are annotated precisely, i.e., it is not a uniform division of the time interval but they are manually annotated