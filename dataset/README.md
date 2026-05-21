# Datasets

GS3-ICL is evaluated on four speech emotion recognition benchmarks:

| Dataset | Size | Emotion Classes | Modalities |
| --- | ---: | ---: | --- |
| MELD | 13,708 utterances | 7 | Text, audio, visual |
| IEMOCAP | 5,531 conversational utterances | 4 | Text, audio, visual |
| RAVDESS | 4,800 acted speech and song recordings | 8 | Audio, visual |
| SAVEE | 480 acted speech samples | 7 | Audio, visual |

MELD and IEMOCAP provide three modalities: text, audio, and visual signals. RAVDESS and SAVEE contain audio and visual modalities.

All datasets are split into training, validation, and test sets with an 8:1:1 ratio. Speaker independence is preserved during splitting to avoid identity leakage between training and evaluation.
