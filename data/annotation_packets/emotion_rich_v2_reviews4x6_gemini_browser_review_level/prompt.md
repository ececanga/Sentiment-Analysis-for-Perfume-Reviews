You are labeling perfume review text for an academic thesis.

Task: For each review in the batch, assign emotion-tone labels using ONLY the review text provided.

Rules:
- Label each review independently.
- Do NOT infer from brand, perfume name, reputation, note list, marketing concept, or niche status.
- Labels reflect the reviewer's emotional tone expressed in the text.
- Set an emotion to 1 only when there is clear linguistic evidence. Otherwise set it to 0.
- Multiple emotions can be 1.
- Surprise=1 only when the text explicitly expresses unexpectedness, such as "surprised", "did not expect", "caught me off guard", or similar language.
- Anticipation=1 when the text clearly expresses expectation, looking forward, curiosity, desire to keep testing, or waiting for something to unfold.
- Trust=1 when the text clearly expresses comfort, safety, reliability, attachment, reassurance, or confidence in wearing the perfume.
- Arousal means emotional activation or calmness, not perfume strength or projection.
- Dominance means confidence/control/assertiveness, not perfume loudness unless framed as personal power/control.
- If emotional evidence is weak or neutral, use all emotions=0 and valence=5, arousal=5, dominance=5.

Emotion columns, binary 0/1:
joy,sadness,anger,fear,trust,surprise,anticipation,disgust

VAD columns, integer 1-9:
valence: 1 very negative, 5 neutral, 9 very positive
arousal: 1 very calm, 5 neutral, 9 highly activated
dominance: 1 passive/gentle, 5 neutral, 9 commanding/confident

Return no markdown, no explanation, and no code fences.
Apart from the batch tracking line requested below, return only the CSV header and rows.
Use exactly this CSV header:
synthetic_review_id,source_type,perfume_id,perfume_name,review_index,joy,sadness,anger,fear,trust,surprise,anticipation,disgust,valence,arousal,dominance

Return exactly one output row per input review.
