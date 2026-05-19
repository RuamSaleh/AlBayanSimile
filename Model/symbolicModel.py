from dataclasses import dataclass
from collections import defaultdict
import numpy as np

SIMILE_PARTICLES_STRONG = {
    "كأن",
    "كأنما",
    "كأنّ",
    "كأنّما",
    "كمثل"
}

SIMILE_PARTICLES_WEAK = {
    "كما",
    "كـ",
    "كال",
    "مثل"
}

SIMILE_VERBS = {
    "يشبه",
    "شبه",
    "يشابه",
    "شابه",
    "أشبه",
    "يماثل",
    "ماثل",
    "يضارع",
    "يضاهي",
    "يحاكي",
    "يقارب"
}

SIMILE_NOUNS = {
    "مثل",
    "شبيه",
    "شبه",
    "نظير",
    "مثيل",
    "مماثل",
    "قرين"
}

FALSE_CONTEXT = {
    "كما أن",
    "كما كان",
    "كما ينبغي",
    "كما هو",
    "كما هي",
    "كما تم",
    "كما ذكر",
    "كما ورد",
    "كما قال",
    "كما يبدو",
    "كما يمكن",
    "مثلًا",
    "مثلاً",
    "على سبيل المثال"
}
# DEFAULT FALLBACK CONFIDENCE
RULE_CONFIDENCE = {

    "explicit_particle": 0.90,
    "weak_particle": 0.60,
    "verb_simile": 0.75,
    "nominal_simile": 0.80,
    "prefix_simile": 0.55
}
def tokenize(text):
    return text.split()


def is_probable_noun(word):

    return (
        word.startswith("ال")
        or word.endswith("ة")
        or len(word) > 3
    )


def has_prefix(word):

    return (
        word.startswith("ك")
        and len(word) > 2
    )

# DATA STRUCTURES
@dataclass
class SimileStructure:

    subject: str
    particle: str
    object: str
    confidence: float
    rule: str


@dataclass
class Evidence:

    structures: list
    confidence: float

    def __init__(self):

        self.structures = []
        self.confidence = 0.0

# LEARN CONFIDENCE FROM DATA

def learn_rule_confidence(texts, labels):

# Store rule statistics:
    # TP = rule detected in a true simile sentence
    # FP = rule detected in a non-simile sentence
    stats = defaultdict(
        lambda: {
            "TP": 0,
            "FP": 0
        }
    )
    # Process each sentence and examine detected
    # symbolic simile structures
    for sentence, label in zip(texts, labels):
        # Run symbolic grammar detector
        result = symbolic_detector(sentence)
        # Update performance counts for each rule
        for s in result.structures:
            
            # Rule fired in a real simile → True Positive
            if label == 1:
                stats[s.rule]["TP"] += 1
            # Rule fired incorrectly → False Positive
            else:
                stats[s.rule]["FP"] += 1
# Convert raw counts into learned confidence scores
    learned_confidence = {}

    for rule, values in stats.items():

        tp = values["TP"]
        fp = values["FP"]
        # Precision estimates how trustworthy
        # the rule is:
        # precision = TP / (TP + FP)
        # Higher precision = stronger symbolic evidence
        precision = tp / (tp + fp + 1e-8)

        learned_confidence[rule] = round(
            float(precision),
            3
        )

    return learned_confidence

# MAIN SYMBOLIC DETECTOR
def symbolic_detector(sentence):

    words = tokenize(sentence)

    ev = Evidence()

    # FALSE CONTEXT FILTER
    if any(
        bad in sentence
        for bad in FALSE_CONTEXT
    ):
        return ev

    # MAIN PATTERN DETECTION
    for i in range(1, len(words)-1):

        w = words[i]

        left = words[i-1]
        right = words[i+1]

        # STRONG PARTICLES
        if w in SIMILE_PARTICLES_STRONG:

            ev.structures.append(

                SimileStructure(

                    subject=left,

                    particle=w,

                    object=right,

                    confidence=RULE_CONFIDENCE.get(
                        "explicit_particle",
                        0.90
                    ),

                    rule="explicit_particle"
                )
            )

        # WEAK PARTICLES
        if w in SIMILE_PARTICLES_WEAK:

            ev.structures.append(

                SimileStructure(

                    subject=left,

                    particle=w,

                    object=right,

                    confidence=RULE_CONFIDENCE.get(
                        "weak_particle",
                        0.60
                    ),

                    rule="weak_particle"
                )
            )

        # VERB SIMILES
        if w in SIMILE_VERBS:

            ev.structures.append(

                SimileStructure(

                    subject=left,

                    particle=w,

                    object=right,

                    confidence=RULE_CONFIDENCE.get(
                        "verb_simile",
                        0.75
                    ),

                    rule="verb_simile"
                )
            )
        # NOMINAL SIMILES
        if w in SIMILE_NOUNS:

            ev.structures.append(

                SimileStructure(

                    subject=left,

                    particle=w,

                    object=right,

                    confidence=RULE_CONFIDENCE.get(
                        "nominal_simile",
                        0.80
                    ),

                    rule="nominal_simile"
                )
            )

    # PREFIX PATTERNS
    for w in words:

        if has_prefix(w):

            ev.structures.append(

                SimileStructure(

                    subject=None,

                    particle="كـ",

                    object=w[1:],

                    confidence=RULE_CONFIDENCE.get(
                        "prefix_simile",
                        0.55
                    ),

                    rule="prefix_simile"
                )
            )

    # PROBABILISTIC EVIDENCE FUSION
    if ev.structures:

        probs = [
            s.confidence
            for s in ev.structures
        ]

        combined = 1.0

        for p in probs:

            combined *= (1 - p)

        ev.confidence = 1 - combined

    return ev

# SYMBOLIC FEATURE EXTRACTION
def extract_symbolic_features(sentence):

    ev = symbolic_detector(sentence)

    words = tokenize(sentence)

    # BASIC FEATURES
    has_structure = (
        1 if len(ev.structures) > 0
        else 0
    )

    num_structures = len(ev.structures)

    max_conf = ev.confidence

    avg_conf = 0.0

    if ev.structures:

        avg_conf = sum(
            s.confidence
            for s in ev.structures
        ) / len(ev.structures)

    # RULE FLAGS

    has_strong_particle = 0
    has_weak_particle = 0
    has_verb = 0
    has_noun = 0
    has_prefix = 0

    distances = []

    # ANALYZE DETECTED STRUCTURES

    for s in ev.structures:

        if s.rule == "explicit_particle":
            has_strong_particle = 1

        elif s.rule == "weak_particle":
            has_weak_particle = 1

        elif s.rule == "verb_simile":
            has_verb = 1

        elif s.rule == "nominal_simile":
            has_noun = 1

        elif s.rule == "prefix_simile":
            has_prefix = 1

        # DISTANCE FEATURE
        if s.subject and s.object:

            try:

                i = words.index(s.subject)

                j = words.index(s.object)

                distances.append(abs(i - j))

            except:
                continue

    avg_distance = (
        sum(distances) / len(distances)
        if distances
        else 0.0
    )

    # MULTI-SIMILE FEATURE
    multi_simile_flag = (
        1 if num_structures > 1
        else 0
    )

    # FINAL FEATURE VECTOR
    return [

        has_structure,
        num_structures,

        max_conf,
        avg_conf,

        has_strong_particle,
        has_weak_particle,

        has_verb,
        has_noun,
        has_prefix,

        avg_distance,

        multi_simile_flag
    ]

# BUILD FEATURE MATRIX
def build_symbolic_matrix(texts):

    features = [

        extract_symbolic_features(s)
        for s in texts
    ]

    return np.array(
        features,
        dtype=np.float32
    )

# EXPLAINABLE SYMBOLIC OUTPUT
def show_symbolic_result(sentence):

    result = symbolic_detector(sentence)

    print("=" * 50)

    print("Sentence:")
    print(sentence)

    print("\n[Symbolic Analysis]")

    if not result.structures:

        print("No simile detected")

    else:

        for s in result.structures:

            print("\nRule:", s.rule)

            print("المشبه:", s.subject)

            print("الأداة:", s.particle)

            print("المشبه به:", s.object)

            print(
                "Rule Confidence:",
                round(s.confidence, 3)
            )

            print("-" * 20)

        print(
            "\nFinal Symbolic Confidence:",
            round(result.confidence, 3)
        )

    print("=" * 50)