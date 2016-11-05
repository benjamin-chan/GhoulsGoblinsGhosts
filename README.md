# Kaggle Ghouls, Goblins, and Ghosts... Boo!

[Link](https://www.kaggle.com/c/ghouls-goblins-and-ghosts-boo)


## Competition Details

Can you classify monsters haunting Kaggle?

Get out your dowsing rods, electromagnetic sensors, … and gradient boosting
machines. Kaggle is haunted and we need your help. After a month of making
scientific observations and taking careful measurements, we've determined that
900 ghouls, ghosts, and goblins are infesting our halls and frightening our
data scientists. When trying garlic, asking politely, and using reverse
psychology didn't work, it became clear that machine learning is the only
answer to banishing our unwanted guests.

![halloween-660x.png](https://kaggle2.blob.core.windows.net/competitions/kaggle/5708/media/halloween-660x.png)

So now the hour has come to put the data we've collected in your hands. We've
managed to identify 371 of the ghastly creatures, but need your help to
vanquish the rest. And only an accurate classification algorithm can thwart
them. Use bone length measurements, severity of rot, extent of soullessness,
and other characteristics to distinguish (and extinguish) the intruders. Are
you ghost-busters up for the challenge?


## Data Files

File Name| Available Formats | Description
:---|:---|:---
`sample_submission.csv` | .zip (1.29 kb)  | a sample submission file in the correct format
`test.csv`              | .zip (20.98 kb) | the test set
`train.csv`             | .zip (15.27 kb) | the training set

## Data fields

Field | Description
:---|:---
`id` | id of the creature
`bone_length` | average length of bone in the creature, normalized between `0` and `1`
`rotting_flesh` | percentage of rotting flesh in the creature
`hair_length` | average hair length, normalized between `0` and `1`
`has_soul` | percentage of soul in the creature
`color` | dominant color of the creature: `white`, `black`, `clear`, `blue`, `green`, `blood`


## My submission

File | Description
:---|:---
[data/processed/submission.csv](data/processed/submission.csv) | Submission data
[docs/index.md](docs/index.md) | Analysis
