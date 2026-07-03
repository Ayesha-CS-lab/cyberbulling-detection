# ABSTRACT

Cyberbullying is one of the darker sides of social media, and it becomes even harder to
catch when people switch between Urdu, Roman Urdu and English in the same conversation.
Most existing tools were built for English alone, and they tend to treat a single rude
message as bullying, ignoring the repetition and harmful intent that actually define it.
This thesis takes a different route. A first stage uses multilingual transformers
(m-BERT and MuRIL) to flag aggressive messages, and a second stage looks at a whole
conversation between two users, weighing how often the aggression repeats, how harmful the
intent is, and how the two people relate, before deciding whether it counts as
cyberbullying. The aggression model reached about 88% accuracy, and the relationship-level
model reached roughly 92%, correctly identifying 196 of 198 bullying pairs. A separate
CLIP-based model was also built for image-and-text memes. In short, the system judges
behaviour over time rather than one message on its own.

**Keywords:** cyberbullying detection, multilingual NLP, Roman Urdu, code-switching,
transformers, m-BERT, MuRIL, repetition, intent to harm.
