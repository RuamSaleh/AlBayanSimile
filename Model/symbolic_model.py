
from dataclasses import dataclass, field
import numpy as np
# import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer

SIMILE_PARTICLES_STRONG = {"كأن","كأنما"}
SIMILE_PARTICLES_WEAK = {"كما"}

SIMILE_VERBS = {"يشبه","شبه","يماثل","يضارع"}
SIMILE_NOUNS = {"مثل","شبيه","نظير"}

FALSE_CONTEXT = {"كما أن","كما كان","كما ينبغي"}

def tokenize(text):
    return text.split()

def is_probable_noun(word):
    return word.startswith("ال") or word.endswith("ة") or len(word) > 3

def has_prefix(word):
    return word.startswith("ك") and len(word) > 2

@dataclass
class SimileStructure:
    def __init__(self, subject, particle, obj, confidence, rule):
        self.subject = subject
        self.particle = particle
        self.object = obj
        self.confidence = confidence
        self.rule = rule
@dataclass
class Evidence:
    def __init__(self):
        self.structures = []
        self.confidence = 0.0


def detect_particle_patterns(words):
    structures = []

    for i in range(1, len(words)-1):
        w = words[i]

        # Strong particles: كأن
        if w in SIMILE_PARTICLES_STRONG:
            left, right = words[i-1], words[i+1]

            if is_probable_noun(left) and is_probable_noun(right):
                structures.append(
                    SimileStructure(
                        subject=left,
                        particle=w,
                        object=right,
                        confidence=0.9,
                        rule="explicit_particle"
                    )
                )

        # Weak particle: كما
        if w in SIMILE_PARTICLES_WEAK:
            left, right = words[i-1], words[i+1]

            if is_probable_noun(left) and is_probable_noun(right):
                structures.append(
                    SimileStructure(
                        subject=left,
                        particle=w,
                        object=right,
                        confidence=0.6,
                        rule="weak_particle"
                    )
                )

    return structures        

def detect_nominal_patterns(words):
    structures = []

    for i in range(1, len(words)-1):
        w = words[i]

        if w in SIMILE_NOUNS:
            left, right = words[i-1], words[i+1]

            if is_probable_noun(left) and is_probable_noun(right):
                structures.append(
                    SimileStructure(
                        subject=left,
                        particle=w,
                        object=right,
                        confidence=0.8,
                        rule="nominal_simile"
                    )
                )

    return structures


def detect_verb_patterns(words):
    structures = []

    for i in range(1, len(words)-1):
        w = words[i]

        if w in SIMILE_VERBS:
            left, right = words[i-1], words[i+1]

            if is_probable_noun(left) and is_probable_noun(right):
                structures.append(
                    SimileStructure(
                        subject=left,
                        particle=w,
                        object=right,
                        confidence=0.75,
                        rule="verb_simile"
                    )
                )

    return structures


def detect_prefix_patterns(words):
    structures = []

    for w in words:
        if has_prefix(w):
            candidate = w[1:]

            if is_probable_noun(candidate):
                structures.append(
                    SimileStructure(
                        subject=None,
                        particle="كـ",
                        object=candidate,
                        confidence=0.55,
                        rule="prefix_simile"
                    )
                )

    return structures

def has_false_context(sentence):
    for bad in FALSE_CONTEXT:
        if bad in sentence:
            return True
    return False

def symbolic_detector(sentence):
    words = tokenize(sentence)
    ev = Evidence()

    if any(bad in sentence for bad in FALSE_CONTEXT):
        return ev

    for i in range(1, len(words)-1):
        w = words[i]
        left, right = words[i-1], words[i+1]

        if w in SIMILE_PARTICLES_STRONG:
            ev.structures.append(SimileStructure(left, w, right, 0.9, "explicit_particle"))

        if w in SIMILE_PARTICLES_WEAK:
            ev.structures.append(SimileStructure(left, w, right, 0.6, "weak_particle"))

        if w in SIMILE_VERBS:
            ev.structures.append(SimileStructure(left, w, right, 0.75, "verb_simile"))

        if w in SIMILE_NOUNS:
            ev.structures.append(SimileStructure(left, w, right, 0.8, "nominal_simile"))

    for w in words:
        if has_prefix(w):
            ev.structures.append(SimileStructure(None, "كـ", w[1:], 0.55, "prefix_simile"))

    if ev.structures:
        ev.confidence = max(s.confidence for s in ev.structures)

    return ev
def extract_symbolic_features(sentence):
    ev = symbolic_detector(sentence)

    has_structure = 1 if len(ev.structures) > 0 else 0
    num_structures = len(ev.structures)
    max_conf = ev.confidence

    has_strong_particle = 0
    has_weak_particle = 0
    has_verb = 0
    has_noun = 0
    has_prefix = 0

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

    return [
        has_structure,
        num_structures,
        max_conf,
        has_strong_particle,
        has_weak_particle,
        has_verb,
        has_noun,
        has_prefix
    ]
def build_symbolic_matrix(sentences):
    features = [extract_symbolic_features(s) for s in sentences]
    return np.array(features, dtype=np.float32)