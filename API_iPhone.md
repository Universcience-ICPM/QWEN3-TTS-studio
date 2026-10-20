# Faire parler Voix depuis un iPhone

Voix (botqwenClaude) tourne sur le Mac ou sur le PC. Un iPhone branché sur le **même Wi-Fi** peut lui demander de dire un texte, avec la voix de votre choix, et entendre la réponse. Il y a trois façons de faire, de la plus simple à la plus pratique au quotidien :

1. ouvrir l'interface complète dans Safari ;
2. ouvrir un simple lien qui fait parler la voix ;
3. créer un Raccourci iPhone, qu'on peut aussi lancer en disant « Dis Siri… ».


## Avant tout : lancer Voix en mode réseau

Par défaut, Voix ne répond qu'à la machine sur laquelle il tourne. Pour qu'un iPhone puisse s'en servir :

- **sur le Mac** : double-clic sur `botqwenClaude_reseau.command` (dans le dossier QWEN) ;
- **sur le PC** : double-clic sur `LANCER_BOTQWENCLAUDE_RESEAU.bat`.

La première fois, l'ordinateur demande d'autoriser les connexions entrantes pour Python : acceptez. Sur Windows, cochez « Réseaux privés ».

### Trouver l'adresse

L'adresse s'affiche dans la fenêtre du terminal au lancement, et dans l'onglet **À distance** de Voix. Elle est recalculée à chaque fois qu'on ouvre l'onglet : si l'ordinateur change de Wi-Fi ou si la box lui donne une autre adresse, c'est la nouvelle qui s'affiche.

Il y en a deux sortes :

- une **adresse IP**, par exemple `http://192.168.1.20:7860`. Elle peut changer d'un jour à l'autre ;
- un **nom**, par exemple `http://Mac-de-Dario.local:7860`. Il ne change pas, même quand l'adresse IP change. C'est le plus pratique pour un Raccourci. L'iPhone le reconnaît tout seul sur un Mac ; sur un PC Windows 10 ou 11, cela marche le plus souvent, sinon prenez l'adresse IP.

Dans la suite, remplacez `ADRESSE` par l'une ou l'autre, sans le `http://` ni le `:7860` (par exemple `Mac-de-Dario.local` ou `192.168.1.20`).


## 1. L'interface complète dans Safari

Dans Safari, sur l'iPhone, tapez :

```
http://ADRESSE:7860
```

C'est la même interface que sur l'ordinateur : lire un texte, discuter, régler les voix…

Pour l'avoir comme une application : bouton Partager, puis **« Sur l'écran d'accueil »**.


## 2. Un lien qui fait parler la voix

Tapez dans Safari, ou enregistrez dans vos signets :

```
http://ADRESSE:7860/dire?voix=Femme centenaire&texte=Bonjour, je suis prête.
```

Safari affiche un petit lecteur et la voix dit la phrase. Comptez quelques secondes, et un peu plus la toute première fois (la voix se charge).

### Ce qu'on peut mettre dans le lien

| Réglage | Exemple | Ce qu'il fait |
|---|---|---|
| `texte` | `texte=Bonjour Louise` | ce qu'il faut dire (obligatoire) |
| `voix` | `voix=Femme centenaire` | une voix clonée de la bibliothèque, ou une voix standard (`Ryan`, `Vivian`…). Sans `voix`, c'est la voix clonée active |
| `emotion` | `emotion=sad` | `neutral`, `happy`, `sad`, `excited`, `curious` ou `angry` (avec une voix clonée, il faut que sa version émotion existe) |
| `langue` | `langue=French` | la langue parlée (`French` par défaut) |
| `mdp` | `mdp=…` | le mot de passe, seulement si vous en avez mis un |

On sépare les réglages par `&`. Les espaces et les accents peuvent s'écrire tels quels dans Safari ; ailleurs, un espace s'écrit `%20` ou `+`.

Le texte accepte aussi les marqueurs de Voix : `[triste]`, `[voix:Marielle]`, et `::` ou `;;` pour une pause.

### La liste des voix

```
http://ADRESSE:7860/voix.json
```

Elle donne le nom exact de chaque voix, ses versions émotion et son réglage.


## 3. Un Raccourci iPhone (et Siri)

L'application **Raccourcis** est déjà sur l'iPhone. Les noms des actions sont donnés en français, avec le nom anglais entre parenthèses si votre iPhone est en anglais.

### Le raccourci « Fais parler la voix »

1. Ouvrez **Raccourcis**, touchez **+** pour en créer un nouveau, et nommez-le, par exemple « Fais parler Madeleine ».
2. Ajoutez l'action **Dicter le texte** (*Dictate Text*), pour parler, ou **Demander une entrée** (*Ask for Input*), pour taper.
3. Ajoutez **Encoder l'URL** (*URL Encode*). Elle prend le texte de l'étape 2 et le rend utilisable dans un lien.
4. Ajoutez **URL** et écrivez :
   `http://ADRESSE:7860/dire?voix=jeune Femme&texte=`
   puis, juste après le `=`, insérez la variable **Texte encodé** de l'étape 3. Remplacez `jeune Femme` par la voix voulue, avec `%20` à la place de chaque espace (`jeune%20Femme`).
5. Ajoutez **Obtenir le contenu de l'URL** (*Get Contents of URL*).
6. Ajoutez **Lire le son** (*Play Sound*). Si vous ne la trouvez pas, **Coup d'œil** (*Quick Look*) fait aussi l'affaire.

Touchez ▶ pour essayer. Ensuite, il suffit de dire **« Dis Siri, fais parler Madeleine »**, puis de dicter la phrase.

### Variante : choisir la voix à chaque fois

Entre les étapes 3 et 4, ajoutez **Choisir dans la liste** (*Choose from List*) avec les noms de vos voix, puis utilisez la variable **Élément choisi**, passée par **Encoder l'URL**, à la place du nom de la voix dans l'URL.

### Pour les textes longs

Un lien a ses limites (environ 2 000 caractères). Pour un long texte, à l'étape 5 :

1. touchez **Afficher plus** ;
2. réglez **Méthode** sur **POST** et **Corps de la requête** sur **JSON** ;
3. ajoutez les champs `texte`, `voix` (et `emotion`, `mdp` au besoin) ;
4. l'URL devient simplement `http://ADRESSE:7860/dire`, et l'étape 3 (Encoder l'URL) n'est plus nécessaire.


## Et sur Android

Tout ce qui précède marche aussi sur un téléphone Android, avec Chrome à la place de Safari :

- l'interface : `http://ADRESSE:7860`, puis menu ⋮ et « Ajouter à l'écran d'accueil » ;
- le lien qui parle : le même lien `…/dire?voix=…&texte=…`.

Chaque son téléchargé porte un nom différent : la voix, le début du texte, la date et l'heure. Par exemple `Femme_centenaire_Bonjour_Louise_2026-10-15_17h32m05.wav`. On les retrouve dans le dossier « Téléchargements ».

Pour un bouton ou une commande vocale, l'équivalent des Raccourcis iPhone est une application comme **HTTP Shortcuts** (gratuite) : on lui donne l'adresse `http://ADRESSE:7860/dire`, la méthode GET, et les paramètres `voix` et `texte`.


## Si ça ne marche pas

### L'iPhone ne se connecte pas (alors qu'un autre appareil y arrive)

Faites ces vérifications dans l'ordre.

1. **Tapez l'adresse complète, avec l'adresse IP** : `http://192.168.1.20:7860/ping`, avec `http://` au début. Sans `http://`, Safari essaie parfois une connexion sécurisée (`https`) que Voix ne connaît pas. Évitez aussi le nom en `.local` pour ce premier essai.
2. **Regardez la fenêtre du terminal sur l'ordinateur.** Chaque appareil qui arrive jusqu'à Voix s'y inscrit, par exemple « 📱 Connexion depuis 192.168.1.34 (iPhone, Safari) ».
   - **Si la ligne apparaît**, la connexion est bonne : la page `/ping` affiche « Voix répond ✓ », avec un lien qui parle. Si c'est l'interface complète qui ne s'ouvre pas, dites-le-moi.
   - **Si rien n'apparaît**, l'iPhone n'atteint pas l'ordinateur. Les causes les plus fréquentes sont ci-dessous.
3. **Le même Wi-Fi.** Dans Réglages, puis Wi-Fi, le nom doit être exactement celui de l'ordinateur. Un réseau « invité », ou un nom à part pour le 5 GHz, peut isoler les appareils les uns des autres.
4. **Un VPN actif.** Dans Réglages, regardez s'il y a une ligne « VPN » activée. Un VPN, un bloqueur de publicités ou une appli de sécurité qui passe par un VPN envoient tout vers Internet, y compris l'adresse de l'ordinateur. Désactivez-le le temps d'essayer.
5. **Les données mobiles.** Si le Wi-Fi est faible, l'iPhone peut passer sur la 4G/5G sans le dire (« Assistance Wi-Fi »). Désactivez les données mobiles le temps d'essayer.
6. **Pour les Raccourcis.** Dans Réglages, puis Confidentialité et sécurité, puis Réseau local, autorisez **Raccourcis**.

### Autres soucis

- **Ça marchait hier, plus aujourd'hui** : l'adresse IP a probablement changé. Prenez la nouvelle dans l'onglet À distance, ou utilisez le nom en `.local` une fois que l'adresse IP marche.
- **« Voix inconnue »** : le nom doit être exactement celui de la bibliothèque (les majuscules ne comptent pas). Vérifiez dans `/voix.json`.
- **Un réseau partagé** (atelier, lieu d'exposition…) : mettez un mot de passe.
  - Sur le Mac : dans `botqwenClaude_reseau.command`, enlevez le `#` devant `export BOTQWEN_MOT_DE_PASSE=…`.
  - Sur le PC : dans `LANCER_BOTQWENCLAUDE_RESEAU.bat`, enlevez le `rem ` devant `set BOTQWEN_MOT_DE_PASSE=…`.
  - Ajoutez ensuite `&mdp=votre-mot-de-passe` à vos liens. Dans l'interface, l'identifiant est `voix`.

Voix ne sort jamais du réseau local : rien ne passe par Internet.
