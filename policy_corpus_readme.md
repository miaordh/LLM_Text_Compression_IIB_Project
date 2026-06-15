# The Policy Corpus
The Policy Corpus contains 5 text (```.txt``` ) files which are all policy and white papers published by the UK Government.
| File name                                     | On-disk size (bytes) | Date first published |
|-----------------------------------------------|---------------------:|----------------------|
| a_new_vision_for_water.txt                    | 97312                | 20/1/2026            |
| enduring_relationships.txt                    | 48465                | 4/6/2026             |
| every_child_achieving_and_thriving.txt        | 291067               | 23/2/2026            |
| greater_cambridge_development_corporation.txt | 105008               | 4/2/2026             |
| a_fairer_end_to_relationships.txt             | 171149               | 5/6/2026             |

The original documents were cleaned so that the following items were removed:
1. Footnotes and footnote labels;
2. Case studies that contain real names or real-life cases;
3. Images, infographics, and their captions; the captions might be kept if there was text detailing contents of the image that could fully replace the presence of the image.

The files were chosen as they were newly publish (and therefore less likely included in pretraining datasets for newer LLMs), have suitable lengths comparable to classic corpora such as the Canterbury Corpus, and can be freely used under the Open Government Licence [Open Government Licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).