# Qwen3-TTS émotif — protocole de test pour Codex

## Objectif

Tester jusqu’où **Qwen3-TTS** peut être utilisé comme un véritable **comédien vocal pilotable**, et pas seulement comme un moteur de synthèse vocale.

Le but est de pouvoir conserver une voix cohérente pour un personnage tout en contrôlant, phrase par phrase :

- l’émotion ;
- l’intensité ;
- le rythme ;
- la hauteur perçue ;
- la tension ;
- la respiration ;
- les silences ;
- le sourire dans la voix ;
- la retenue ;
- les changements émotionnels progressifs ;
- l’intention dramatique.

Exemples recherchés :

- colère contenue puis explosion ;
- joie chaleureuse avec sourire audible ;
- peur avec respiration courte et hésitations ;
- tristesse contenue sans surjeu ;
- ironie ;
- soulagement ;
- tendresse ;
- fatigue ;
- colère mêlée d’affection ;
- inquiétude masquée.

L’objectif final est de déterminer si Qwen3-TTS peut servir dans une application narrative où chaque personnage possède une identité vocale stable et une véritable direction d’acteur.

---

# 1. Modèles Qwen3-TTS à comparer

Tester en priorité :

## A. Qwen3-TTS 1.7B CustomVoice

C’est le modèle à tester en premier pour le contrôle émotionnel.

Il accepte une instruction de style du type :

```python
instruct="Speak with restrained anger..."
```

Il faut déterminer avec précision jusqu’où ce paramètre influence réellement :

- prosodie ;
- intensité ;
- hauteur ;
- rythme ;
- sourire ;
- respiration ;
- pauses ;
- articulation ;
- émotion.

---

## B. Qwen3-TTS 0.6B CustomVoice

Même protocole que pour le 1.7B.

Objectif :

Comparer :

- expressivité ;
- fidélité aux instructions ;
- stabilité ;
- vitesse ;
- consommation mémoire.

---

## C. Qwen3-TTS Base / Voice Clone

Tester séparément le modèle utilisé pour le clonage vocal.

Question centrale :

> Peut-on conserver une voix clonée tout en lui imposant une direction émotionnelle comparable à CustomVoice ?

Si `instruct` n’est pas directement disponible ou efficace, tester :

- influence de la ponctuation ;
- influence de l’audio de référence ;
- plusieurs références émotionnelles ;
- segmentation du texte ;
- seeds différentes ;
- éventuels mécanismes communautaires de voice embedding / expression steering.

---

# 2. Principe général

Ne pas traiter une émotion comme une simple étiquette.

Éviter :

```text
angry
happy
sad
afraid
```

Préférer une **direction d’acteur descriptive**.

Exemple :

```text
Speak with restrained anger.
Start quietly, with a tense and controlled voice.
Keep the words clipped and precise.
Use short pauses.
Gradually increase the intensity.
Stress the accusations.
Do not shout at first.
Lose control only on the final sentence.
```

Le but du test est de vérifier si Qwen comprend mieux :

```text
"anger"
```

ou :

```text
"restrained anger + tense voice + short pauses + growing intensity"
```

---

# 3. Architecture cible

L’architecture envisagée est :

```text
TEXTE
  |
  v
Analyse dramaturgique
  |
  +--> émotion
  +--> intensité
  +--> intention
  +--> rythme
  +--> tension
  +--> respiration
  +--> sourire
  +--> évolution émotionnelle
  |
  v
Direction d’acteur
  |
  v
Qwen3-TTS
  |
  +--> prise 1
  +--> prise 2
  +--> prise 3
  +--> prise 4
  |
  v
Sélection de la meilleure prise
```

---

# 4. Première batterie de tests

Utiliser toujours exactement le même texte.

Phrase de test :

```text
Tu étais où ? Je t’ai appelé trois fois. Trois fois, Rakoto. Et toi... rien.
```

Produire plusieurs versions.

---

## Test 1 — neutre

```text
Speak naturally in a calm, neutral voice.
No strong emotion.
Medium speaking rate.
```

---

## Test 2 — colère contenue

```text
Speak with restrained anger.
The voice is tense but controlled.
Keep the pitch slightly low at first.
Use clipped articulation.
Short pauses.
Stress "trois fois" and "rien".
The speaker is trying very hard not to shout.
Let the anger become more audible toward the end.
```

---

## Test 3 — colère explosive

```text
Speak with explosive anger.
Strong projection.
Sharp consonants.
Fast breathing.
Increase vocal intensity rapidly.
Stress the accusations strongly.
Raise the voice significantly on the final words.
The speaker has just lost control.
```

---

## Test 4 — peur

```text
Speak with growing fear.
The voice is slightly breathless.
Use irregular rhythm.
Add small hesitations.
The pitch rises involuntarily on alarming words.
The speaker tries to remain quiet because someone may be listening.
```

---

## Test 5 — tristesse contenue

```text
Speak with restrained sadness.
Low energy.
Slow rhythm.
Slightly lower pitch.
Longer pauses before emotionally difficult words.
The speaker is trying not to cry.
Do not exaggerate.
```

---

## Test 6 — joie chaleureuse

```text
Speak with genuine warm happiness.
Smile naturally while speaking.
The smile should be audible in the voice.
Use relaxed articulation.
Light melodic pitch variations.
Slightly quicker rhythm.
The speaker is deeply pleased, not excited.
Do not sound childish.
```

---

## Test 7 — ironie

```text
Speak with dry irony.
The speaker does not openly laugh.
Use subtle emphasis on key words.
Slightly slow the most sarcastic phrases.
Keep the voice controlled and intelligent.
The contempt should remain understated.
```

---

## Test 8 — soulagement

```text
Speak with strong relief after intense fear.
Start with residual tension.
Gradually release the voice.
Breathing becomes easier while speaking.
The end of the sentence should sound softer and warmer.
```

---

# 5. Tester les émotions complexes

Une émotion seule est insuffisante pour du dialogue dramatique.

Tester des mélanges.

---

## Colère + peur

```text
The speaker is angry because they are frightened.
Speak with restrained anger hiding fear.
The voice is tense.
Keep the rhythm controlled at first.
Occasional slight breathlessness should betray the fear.
Do not sound openly scared.
```

---

## Colère + affection

```text
The speaker is angry because they care deeply about the other person.
Start sharply and reproachfully.
The anger is real, but affection should gradually become audible.
Soften unexpectedly on the final words.
Do not become sentimental.
```

---

## Soulagement + colère

```text
The speaker thought the other person might be dead.
Now they are alive.
Speak with anger caused by intense relief.
Start aggressively.
The voice should briefly crack with emotion.
Let warmth appear underneath the reproach.
```

---

## Joie + fatigue

```text
The speaker is exhausted but genuinely happy.
Keep the energy physically low.
The voice should still carry a warm smile.
Avoid excitement.
Use slow, relaxed phrasing.
```

---

# 6. Tester le contexte dramatique

Comparer deux méthodes.

## Méthode A

Instruction purement acoustique :

```text
Speak with restrained anger.
Short pauses.
Tense voice.
Gradually increase intensity.
```

## Méthode B

Instruction avec contexte dramatique :

```text
Jeanne has been terrified that Rakoto might die.
He has just returned alive.
She is angry because she was afraid for him.
She wants to scold him, but relief and affection are breaking through.

Speak with restrained anger mixed with relief.
Start sharply.
Keep the first words tense.
Gradually soften.
A slight smile should become audible toward the end.
```

Texte :

```text
Tu es vraiment un imbécile.
```

Comparer les deux résultats.

Objectif :

Déterminer si le contexte narratif améliore réellement l’interprétation ou si seules les consignes acoustiques sont prises en compte.

---

# 7. Tester le sourire dans la voix

Phrase :

```text
Je suis vraiment contente de te voir.
```

Variantes :

## Sans instruction

Aucune instruction émotionnelle.

## Happy

```text
Very happy.
```

## Sourire détaillé

```text
Speak with genuine warm happiness.
Smile naturally while speaking.
The smile should be clearly audible in the voice.
Keep the voice warm and relaxed.
Do not raise the pitch excessively.
Do not sound childish or overexcited.
```

Critères :

- changement spectral audible ;
- articulation plus ouverte ;
- impression de sourire ;
- variation mélodique ;
- naturel ;
- absence d’effet caricatural.

---

# 8. Tester la montée en colère

Phrase :

```text
Je t’ai demandé de partir. Je te l’ai demandé deux fois. Maintenant, sors.
```

Instruction :

```text
Start with very restrained irritation.
The first sentence should remain controlled.
Increase tension on the second sentence.
The final sentence must be significantly louder and more forceful.
The emotional progression must be clearly audible.
Do not start angry.
```

Critère principal :

Qwen sait-il produire une **évolution émotionnelle à l’intérieur d’une même génération** ?

---

# 9. Tester la segmentation

Comparer :

## Version A — phrase complète

```text
Tu étais où ? Je t’ai appelé trois fois. Trois fois, Rakoto. Et toi... rien.
```

Une seule génération.

## Version B — segmentation

Segment 1 :

```text
Tu étais où ?
```

Direction :

```text
Worried irritation. Controlled.
```

Segment 2 :

```text
Je t’ai appelé trois fois.
```

Direction :

```text
Growing anger.
```

Segment 3 :

```text
Trois fois, Rakoto.
```

Direction :

```text
Restrained anger. Strong emphasis on "trois fois".
```

Segment 4 :

```text
Et toi... rien.
```

Direction :

```text
Cold disappointment. Lower intensity. Hurt underneath the anger.
```

Assembler ensuite les WAV.

Comparer :

- naturel ;
- continuité de voix ;
- continuité émotionnelle ;
- respiration ;
- cohérence des niveaux ;
- qualité dramatique.

Hypothèse :

La segmentation peut donner un meilleur contrôle que la génération d’un long paragraphe.

---

# 10. Tester la ponctuation

Créer plusieurs versions du même texte.

## Version 1

```text
Tu étais où ? Je t’ai appelé trois fois. Trois fois, Rakoto. Et toi... rien.
```

## Version 2

```text
Tu étais où ?

Je t’ai appelé trois fois.

Trois fois, Rakoto.

Et toi... rien.
```

## Version 3

```text
Tu étais où ?...
Je t’ai appelé trois fois.
Trois fois, Rakoto...
Et toi...
rien.
```

Mesurer l’impact sur :

- pauses ;
- intensité ;
- respiration ;
- rythme ;
- interprétation.

---

# 11. Gestion de la ponctuation spéciale `;;`

Dans les récits, `;;` représente une pause narrative ou un changement de situation.

Exemple :

```text
Tu étais où ? ;;
Je t’ai appelé trois fois. ;;
Trois fois, Rakoto. ;;
Et toi... rien.
```

Tester deux méthodes.

## Méthode 1 — remplacer par silence écrit

```text
Tu étais où ?

Je t’ai appelé trois fois.

Trois fois, Rakoto.

Et toi... rien.
```

## Méthode 2 — générer chaque bloc séparément

Parser :

```python
segments = text.split(";;")
```

Puis générer chaque segment individuellement et ajouter un silence WAV configurable.

Exemple :

```python
pause_ms = 450
```

Tester plusieurs valeurs :

- 250 ms ;
- 400 ms ;
- 600 ms ;
- 900 ms.

---

# 12. Plusieurs prises / seeds

Ne pas considérer une génération comme déterministe.

Pour chaque phrase importante, produire plusieurs prises.

Pseudo-code :

```python
seeds = [11, 27, 42, 81]

for seed in seeds:
    torch.manual_seed(seed)

    wavs, sr = model.generate_custom_voice(
        text=text,
        language="French",
        speaker=speaker,
        instruct=acting_direction,
    )
```

Conserver :

```text
take_11.wav
take_27.wav
take_42.wav
take_81.wav
```

Comparer la variance de :

- émotion ;
- hauteur ;
- rythme ;
- pauses ;
- intensité ;
- articulation.

---

# 13. Paramètres de personnage

Créer une fiche par personnage.

Exemple :

```json
{
  "name": "Rakoto",
  "voice": "speaker_x",
  "default_style": {
    "pitch": "low",
    "rate": "slow",
    "energy": "restrained",
    "articulation": "precise",
    "emotional_display": "low"
  }
}
```

Puis définir comment ce personnage exprime les émotions.

```json
{
  "anger": {
    "style": "cold restrained anger",
    "volume_change": "moderate",
    "pitch_change": "small",
    "rate_change": "slightly faster",
    "explosion_probability": "low"
  },

  "joy": {
    "style": "warm but discreet",
    "smile": "subtle",
    "pitch_change": "small",
    "rate_change": "slightly faster"
  }
}
```

Le but est que :

```text
colère de Rakoto
```

ne ressemble pas à :

```text
colère de Jeanne
```

---

# 14. Générateur automatique de direction d’acteur

Créer une fonction :

```python
build_acting_direction(
    character,
    emotion,
    intensity,
    context,
    evolution=None
)
```

Exemple d’entrée :

```json
{
  "character": "Jeanne",
  "emotion": "anger",
  "intensity": 0.65,
  "context": "Rakoto vient de revenir vivant après avoir disparu.",
  "subtext": "Elle a eu peur de le perdre.",
  "evolution": "anger_to_relief"
}
```

Sortie :

```text
Speak with restrained anger mixed with relief.
The voice is tense and reproachful.
The anger comes from fear for the other person.
Start sharply.
Keep the first half controlled.
Gradually soften.
A slight warmth should appear toward the end.
Do not become sentimental.
```

---

# 15. Intensité émotionnelle

Tester une échelle normalisée :

```text
0.0 = neutre
0.2 = émotion à peine perceptible
0.4 = émotion claire mais contenue
0.6 = émotion forte
0.8 = émotion très forte
1.0 = maximum / perte de contrôle
```

Exemple pour colère :

## 0.2

```text
Very slight irritation.
Remain almost neutral.
Only subtle tension should be audible.
```

## 0.5

```text
Clear restrained anger.
Tense articulation.
Moderate emphasis.
Do not shout.
```

## 0.8

```text
Strong anger.
High vocal intensity.
Sharp articulation.
The speaker is close to shouting.
```

## 1.0

```text
Explosive uncontrolled anger.
The speaker has completely lost control.
Strong projection.
Very intense delivery.
```

Objectif :

Vérifier si l’intensité est réellement graduelle.

---

# 16. Voice cloning + émotion

C’est le test le plus important à long terme.

Objectif :

```text
IDENTITÉ VOCALE FIXE
+
DIRECTION D’ACTEUR VARIABLE
```

Tester :

1. une référence vocale neutre ;
2. une référence vocale joyeuse ;
3. une référence vocale triste ;
4. une référence vocale en colère.

Puis synthétiser exactement la même phrase.

Comparer :

- identité vocale ;
- émotion transférée ;
- stabilité ;
- ressemblance avec la référence.

---

# 17. Références multiples par personnage

Si le modèle ou l’implémentation le permet, créer pour chaque personnage une petite bibliothèque :

```text
rakoto_neutral.wav
rakoto_angry.wav
rakoto_tired.wav
rakoto_happy.wav
rakoto_afraid.wav
```

Tester si sélectionner la référence adaptée améliore davantage l’émotion qu’une référence neutre.

---

# 18. Expression embeddings / steering

Explorer les implémentations communautaires Qwen3-TTS qui expérimentent :

- steering vectors ;
- emotion embeddings ;
- voice embedding transfer ;
- expression vectors ;
- mélange de styles.

Objectif :

Tester si une représentation peut être séparée en :

```text
speaker identity
+
emotion
+
prosody
```

Idéalement :

```python
voice = rakoto_embedding
emotion = anger_vector
intensity = 0.65

output = synthesize(
    voice=voice,
    emotion=emotion,
    emotion_strength=intensity
)
```

Ne pas considérer ces fonctions comme officielles tant qu’elles ne sont pas présentes dans Qwen3-TTS officiel.

---

# 19. Test français

Tous les tests doivent être réalisés en français.

Créer également un corpus spécifique.

## Neutre

```text
Le train arrivera dans dix minutes.
```

## Colère

```text
Je t’avais pourtant dit de ne pas y toucher.
```

## Joie

```text
Tu es là ! Je n’arrive pas à y croire.
```

## Peur

```text
Il y a quelqu’un derrière la porte.
```

## Tristesse

```text
Je pensais vraiment qu’il reviendrait.
```

## Soulagement

```text
Mon Dieu... tu es vivant.
```

## Ironie

```text
Évidemment. Quelle excellente idée.
```

## Tendresse

```text
Viens là. Tout va bien maintenant.
```

---

# 20. Critères d’évaluation

Créer une note de 0 à 5 pour chaque génération.

```text
Emotion correctness
Emotion intensity
Naturalness
Voice identity
Prosody
Rhythm
Pauses
Pitch behavior
Breathing
Articulation
Smile / warmth
Emotional progression
French pronunciation
Character consistency
```

Exemple CSV :

```csv
model,speaker,emotion,intensity,seed,naturalness,emotion_accuracy,voice_consistency,prosody,notes
1.7B,SpeakerA,anger,0.6,11,4,5,5,4,"bonne montée"
1.7B,SpeakerA,anger,0.6,27,5,3,5,4,"trop calme"
```

---

# 21. Tests AB

Créer systématiquement des comparaisons :

```text
A = simple emotion label
B = detailed acting direction
```

Puis :

```text
A = one-shot paragraph
B = segmented dialogue
```

Puis :

```text
A = neutral voice reference
B = emotional voice reference
```

Puis :

```text
A = one generation
B = four seeds
```

---

# 22. Analyse acoustique facultative

Pour objectiver certains résultats, mesurer :

- F0 moyen ;
- variation de F0 ;
- RMS ;
- dynamique ;
- durée ;
- pauses ;
- débit syllabique ;
- spectral centroid ;
- respiration détectable.

Mais l’évaluation humaine reste prioritaire.

Une émotion convaincante n’est pas réductible à :

```text
pitch +20 %
speed +10 %
volume +15 %
```

---

# 23. Application cible

Le système final doit pouvoir recevoir quelque chose comme :

```json
{
  "character": "Jeanne",
  "text": "Tu es vraiment un imbécile.",
  "emotion": "anger",
  "intensity": 0.55,
  "subtext": "Elle a eu très peur pour Rakoto.",
  "evolution": "anger_to_relief"
}
```

Puis produire automatiquement :

```text
Speak with restrained anger mixed with relief.
She is angry because she was frightened for him.
Start sharply and reproachfully.
Keep the voice tense but controlled.
Gradually soften.
A slight warmth should appear toward the end.
Do not become sentimental.
```

et enfin générer plusieurs prises.

---

# 24. API interne proposée

Créer une abstraction indépendante du moteur :

```python
class EmotionalTTS:

    def synthesize(
        self,
        text: str,
        character: str,
        emotion: str = "neutral",
        intensity: float = 0.5,
        context: str | None = None,
        subtext: str | None = None,
        evolution: str | None = None,
        seed: int | None = None,
    ):
        ...
```

Cela permettra de remplacer plus tard Qwen par :

- Seed Audio ;
- un autre modèle local ;
- un service cloud ;
- un futur modèle expressif.

---

# 25. Livrable demandé à Codex

Créer un prototype Python permettant :

1. de sélectionner un modèle Qwen3-TTS disponible localement ;
2. de saisir un texte ;
3. de choisir une émotion ;
4. de choisir une intensité entre 0 et 1 ;
5. d’ajouter un contexte dramatique ;
6. de choisir un personnage / speaker ;
7. de générer automatiquement la direction d’acteur ;
8. de générer 1 à 4 prises avec seeds différentes ;
9. d’écouter les WAV produits ;
10. de comparer les versions ;
11. de sauvegarder les paramètres dans un JSON ;
12. de supporter la ponctuation spéciale `;;` ;
13. de permettre une génération segmentée ;
14. de créer un CSV de notation des prises.

---

# 26. Interface minimale souhaitée

Prototype possible avec PySide6.

Panneau :

```text
TEXT
[................................]

CHARACTER
[Jeanne ▼]

EMOTION
[anger ▼]

INTENSITY
[0.65 --------]

CONTEXT
[Rakoto vient de revenir vivant...]

SUBTEXT
[Elle a eu peur de le perdre.]

EVOLUTION
[anger_to_relief ▼]

TAKES
[4]

[GENERATE]
```

Puis :

```text
TAKE 1     ▶    ★★★★☆
TAKE 2     ▶    ★★★★★
TAKE 3     ▶    ★★★☆☆
TAKE 4     ▶    ★★★★☆
```

---

# 27. Priorités

Ordre conseillé :

## Phase 1

Tester uniquement :

```text
Qwen3-TTS 1.7B CustomVoice
```

avec :

- français ;
- 8 émotions ;
- instructions détaillées ;
- 4 seeds.

## Phase 2

Ajouter :

- intensité ;
- segmentation ;
- ponctuation `;;` ;
- contexte dramatique.

## Phase 3

Tester :

```text
voice cloning + émotion
```

## Phase 4

Explorer :

- emotion embeddings ;
- steering vectors ;
- références émotionnelles multiples.

---

# 28. Critère de réussite

Le prototype est considéré comme réussi si un auditeur peut reconnaître de façon fiable, sans voir le prompt :

```text
colère
joie
peur
tristesse
soulagement
ironie
tendresse
```

et distinguer :

```text
colère contenue
```

de :

```text
colère explosive
```

tout en conservant une identité vocale cohérente.

Le niveau supérieur de réussite est atteint si le système arrive à produire :

```text
colère + affection
colère + peur
soulagement + colère
joie + fatigue
```

sans tomber dans la caricature.

---

# 29. Hypothèse centrale à vérifier

L’hypothèse de travail est la suivante :

> Qwen3-TTS devient nettement plus expressif lorsqu’on remplace les simples étiquettes émotionnelles par des directions d’acteur décrivant concrètement la prosodie, le rythme, la tension, la respiration, l’intention et l’évolution émotionnelle.

Deuxième hypothèse :

> Pour les dialogues dramatiques, la segmentation en petites unités émotionnelles produit probablement un résultat plus contrôlable qu’une longue génération unique.

Troisième hypothèse :

> Plusieurs générations avec seeds différentes sont nécessaires pour obtenir une véritable qualité de jeu, comme plusieurs prises d’un comédien.

---

# 30. Résultat attendu

À terme :

```text
SCÉNARIO
   |
   v
analyse dramatique automatique
   |
   v
direction d’acteur
   |
   v
voix du personnage
   |
   v
Qwen3-TTS
   |
   v
plusieurs prises
   |
   v
sélection
   |
   v
dialogue joué
```

L’objectif n’est donc pas seulement de faire parler Qwen3-TTS.

L’objectif est de déterminer s’il peut devenir un **moteur de jeu vocal pilotable**, capable de faire vivre de véritables personnages.
