import pandas as pd

from src.utils.paths import data_path, labels_path

# Load unified file
df = pd.read_csv(data_path("perfume_level", "Unified_Emotion_Profile_Perfume.csv"))

COARSE = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral']

LABEL_MAP = {
    # JOY
    'joy':'joy','love':'joy','optimism':'joy','admiration':'joy',
    'approval':'joy','amusement':'joy','excitement':'joy',
    'gratitude':'joy','relief':'joy','pride':'joy','desire':'joy',
    'trust':'joy','positive':'joy',

    # SADNESS
    'sadness':'sadness','disappointment':'sadness',
    'grief':'sadness','remorse':'sadness',

    # ANGER
    'anger':'anger','annoyance':'anger',
    'disapproval':'anger','disgust':'anger',

    # FEAR
    'fear':'fear','nervousness':'fear',
    'anxiety':'fear','embarrassment':'fear',

    # SURPRISE
    'surprise':'surprise','realization':'surprise',
    'confusion':'surprise',

    # NEUTRAL
    'neutral':'neutral'
}

def coarsen(label):
    if not isinstance(label, str):
        return 'neutral'
    label = label.strip().lower()
    return LABEL_MAP.get(label, 'neutral')

silver_labels = []

for _, row in df.iterrows():
    votes = []

    for col in ['JHartmann_Label', 'Savani_Label',
                'GoEmotions_Label', 'Cardiff_Label']:
        if col in df.columns and pd.notna(row[col]):
            votes.append(coarsen(row[col]))

    if not votes:
        silver_labels.append('neutral')
        continue

    vote_counts = pd.Series(votes).value_counts()
    silver_labels.append(vote_counts.idxmax())

df['Silver_Label'] = silver_labels

df.to_csv(labels_path("silver", "Unified_With_Silver_Labels.csv"), index=False)

print("Silver labels created successfully.")
print(df['Silver_Label'].value_counts())
