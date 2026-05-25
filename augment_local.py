"""
Local Torlak XPOS data augmentation using Gemini.
Runs without Colab — reads/writes to the local dataset/ folder.

Usage:
    python augment_local.py

API key is read from gemini_api_key.txt in the same directory.
Augmented data is appended to:
    tor_augmented_train.tsv
    tor_augmented_dev.tsv
"""

import os, re, time, random, math
from collections import defaultdict, Counter
import google.generativeai as genai

# ── CONFIG ────────────────────────────────────────────────────────────────────
HERE           = os.path.dirname(os.path.abspath(__file__))
TRAIN_TSV      = os.path.join(HERE, 'dataset', 'tor_train.tsv')
AUG_TRAIN_TSV  = os.path.join(HERE, 'tor_augmented_train.tsv')
AUG_DEV_TSV    = os.path.join(HERE, 'tor_augmented_dev.tsv')
API_KEY_FILE   = os.path.join(HERE, 'gemini_api_key.txt')

GEMINI_MODEL        = 'gemini-2.5-flash'
SENTENCES_PER_TAG   = 20    # increased — more attempts for hard tags
FEW_SHOT_SENTS      = 3
DEV_RATIO           = 0.15
SLEEP_BETWEEN       = 1.5
SEED                = 42
ADEQUATE_THRESHOLD  = 3     # all remaining tags have orig=0, need ≥ 3

# Tags still needing augmentation (orig=0, total < 3 in existing aug files)
TARGET_TAGS = [
    'Agcfsan', 'Agpfsayn', 'Agpmpay-t', 'Agsnsny', 'Appfsnn',
    'Mdc', 'Ncfpay', 'Ncfsant', 'Ncfsg-t', 'Ncfsgnt',
    'Ncfsnnt', 'Ncfsnt', 'Ncmpgn', 'Ncmpgy', 'Ncmpl',
    'Ncmsayt', 'Ncmsln', 'Ncmslnt', 'Ncmsnnt', 'Ncmsnyt',
    'Ncmsvn', 'Ncmsyv', 'Ncnpa-t', 'Ncnsant', 'Ncnsay',
    'Ncnsgyt', 'Ncnsny', 'Npmsnn-t', 'Pp1-say', 'Pp1mpa',
    'Pq-fsa', 'Pq-n-n', 'Ps1msg', 'Ps1nsn-t', 'Va',
    'Vae3p', 'Vmc-sn', 'Vmcp-sf', 'Vmf1p',
]

random.seed(SEED)

# ── MTE tag decoder — helps Gemini understand what to generate ────────────────
_N_TYPE   = {'c': 'common', 'p': 'proper'}
_V_TYPE   = {'m': 'main', 'a': 'auxiliary', 'c': 'copula', 'e': 'existential'}
_GENDER   = {'m': 'masculine', 'f': 'feminine', 'n': 'neuter'}
_NUMBER   = {'s': 'singular', 'p': 'plural', 'd': 'dual'}
_CASE     = {'n': 'nominative', 'g': 'genitive', 'd': 'dative',
             'a': 'accusative', 'l': 'locative', 'i': 'instrumental', 'v': 'vocative'}
_ANIMATE  = {'y': 'animate', 'n': 'inanimate'}
_A_DEGREE = {'p': 'positive', 'c': 'comparative', 's': 'superlative'}
_V_MOOD   = {'i': 'indicative', 'm': 'imperative', 'c': 'conditional'}
_V_TENSE  = {'p': 'present', 'r': 'aorist/past', 'f': 'future', 'a': 'aorist'}
_V_PERSON = {'1': '1st', '2': '2nd', '3': '3rd'}

def decode_mte(tag: str) -> str:
    """Return a human-readable English description of an MTE XPOS tag."""
    if not tag:
        return tag
    cat = tag[0].upper()
    rest = tag[1:]

    try:
        if cat == 'N':
            # Ncfsant → N c f s a n -t
            typ  = _N_TYPE.get(rest[0], rest[0]) if len(rest) > 0 else ''
            gen  = _GENDER.get(rest[1], rest[1]) if len(rest) > 1 else ''
            num  = _NUMBER.get(rest[2], rest[2]) if len(rest) > 2 else ''
            case = _CASE.get(rest[3], rest[3])   if len(rest) > 3 else ''
            anim = _ANIMATE.get(rest[4], '')     if len(rest) > 4 else ''
            suf  = ' toponym-suffix' if '-t' in tag else (
                   ' vocative-suffix' if '-v' in tag else (
                   ' non-standard'    if 'n' == rest[-1] and len(rest) > 5 else ''))
            parts = [p for p in [typ, gen, num, case, anim] if p]
            return f"Noun, {', '.join(parts)}{suf}"

        elif cat == 'V':
            typ   = _V_TYPE.get(rest[0], rest[0]) if len(rest) > 0 else ''
            mood  = _V_MOOD.get(rest[1], rest[1]) if len(rest) > 1 else ''
            tense = _V_TENSE.get(rest[2], rest[2]) if len(rest) > 2 else ''
            pers  = _V_PERSON.get(rest[3], rest[3]) if len(rest) > 3 else ''
            num   = _NUMBER.get(rest[4], rest[4]) if len(rest) > 4 else ''
            gen   = _GENDER.get(rest[5], '') if len(rest) > 5 else ''
            parts = [p for p in [typ, mood, tense, f'{pers} person' if pers else '', num, gen] if p]
            return f"Verb, {', '.join(parts)}"

        elif cat == 'A':
            # Agpfsn → A g(eneral) p(ositive) f s n
            typ   = {'g': 'general', 'o': 'ordinal', 's': 'superlative', 'p': 'possessive'}.get(rest[0], rest[0])
            deg   = _A_DEGREE.get(rest[1], rest[1]) if len(rest) > 1 else ''
            gen   = _GENDER.get(rest[2], rest[2])   if len(rest) > 2 else ''
            num   = _NUMBER.get(rest[3], rest[3])   if len(rest) > 3 else ''
            case  = _CASE.get(rest[4], rest[4])     if len(rest) > 4 else ''
            anim  = _ANIMATE.get(rest[5], '')       if len(rest) > 5 else ''
            yn    = ' (non-standard form)' if rest.endswith('n') and len(rest) > 6 else ''
            parts = [p for p in [typ, deg, gen, num, case, anim] if p]
            return f"Adjective, {', '.join(parts)}{yn}"

        elif cat == 'P':
            ptype = {'p': 'personal', 's': 'possessive', 'd': 'demonstrative',
                     'q': 'interrogative/relative', 'x': 'reflexive', 'i': 'indefinite'}.get(rest[0], rest[0])
            return f"Pronoun, {ptype}, {tag[2:]}"

        elif cat == 'M':
            mtype = {'l': 'cardinal', 'o': 'ordinal', 'd': 'adverbial',
                     'c': 'collective', 's': 'special'}.get(rest[0], rest[0])
            return f"Numeral, {mtype}"

    except IndexError:
        pass

    return tag   # fallback: return the tag itself

# ── Related-tag finder — for 0-example tags, borrow examples from cousins ─────
def find_related_tags(tag: str, tag_to_sents: dict, max_dist: int = 2) -> list:
    """Return sentences from tags that differ in ≤ max_dist characters."""
    candidates = []
    for other, sents in tag_to_sents.items():
        if other == tag or not sents:
            continue
        # Align by prefix — compare up to the shorter length
        n = min(len(tag), len(other))
        diffs = sum(1 for a, b in zip(tag[:n], other[:n]) if a != b)
        diffs += abs(len(tag) - len(other))
        if diffs <= max_dist:
            candidates.extend(sents[:2])
    return candidates[:FEW_SHOT_SENTS]

# ── TSV helpers ───────────────────────────────────────────────────────────────
def load_tsv(path):
    sentences, current = [], []
    if not os.path.exists(path):
        return sentences
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                if current:
                    sentences.append(current)
                    current = []
            else:
                parts = line.split('\t')
                if len(parts) == 3 and parts[0].strip():
                    current.append({'form': parts[0], 'lemma': parts[1], 'xpos': parts[2]})
    if current:
        sentences.append(current)
    return sentences

def sent_to_tsv(sent):
    return '\n'.join(f"{t['form']}\t{t['lemma']}\t{t['xpos']}" for t in sent)

def write_tsv(path, sentences, append=True):
    mode = 'a' if append and os.path.exists(path) else 'w'
    with open(path, mode, encoding='utf-8') as f:
        for sent in sentences:
            for tok in sent:
                f.write(f"{tok['form']}\t{tok['lemma']}\t{tok['xpos']}\n")
            f.write('\n')

# ── Gemini call ───────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert annotator of Torlak dialect Serbian (a South Slavic dialect spoken around Niš, Leskovac, Vranje) with the MTE (Multext-East) morphological tagset.

Rules:
- Write natural-sounding Torlak speech — use dialect forms, vowel reductions (e→i in unstressed syllables), loss of case endings, local lexicon. NOT standard Serbian.
- Output ONLY tab-separated TSV: form<TAB>lemma<TAB>xpos, one token per line.
- Separate sentences with a single blank line.
- Annotate EVERY token including punctuation (Z period = Z, Z comma = Z).
- Each sentence MUST contain at least one token tagged with the target XPOS.
- The lemma must be the dictionary form (nominative singular for nouns/adjectives).
- Do NOT add any commentary, headers, markdown, or code fences — raw TSV only."""

def build_prompt(tag: str, few_shots: list, n: int = SENTENCES_PER_TAG) -> str:
    description = decode_mte(tag)
    fs_block = '\n\n'.join(few_shots) if few_shots else '(no direct examples in corpus)'
    note = ''
    if not few_shots:
        note = (
            f"\nNote: '{tag}' does not appear in the training corpus. "
            f"Use the tag description and related examples above to construct correct sentences.\n"
        )
    return (
        f"Target XPOS tag: {tag}\n"
        f"Tag description: {description}\n"
        f"{note}"
        f"Generate {n} Torlak dialect sentences, each containing at least one token tagged '{tag}'.\n\n"
        f"Reference sentences (from corpus — same or related tags):\n"
        f"---\n{fs_block}\n---\n\n"
        f"Output {n} new annotated sentences in TSV format:"
    )

def call_gemini(client, prompt: str, retries: int = 4) -> str:
    for attempt in range(retries):
        try:
            resp = client.generate_content(
                [SYSTEM_PROMPT, prompt],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.85,
                    max_output_tokens=1500,
                )
            )
            return resp.text
        except Exception as e:
            wait = 6 * (attempt + 1)
            print(f'    Attempt {attempt+1} failed: {e}  (retry in {wait}s)')
            time.sleep(wait)
    return ''

def parse_response(text: str, target_tag: str) -> list:
    """Parse Gemini TSV output; keep only sentences containing target_tag."""
    sentences, current = [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if current:
                sentences.append(current)
                current = []
            continue
        if line.startswith('```'):
            continue
        parts = line.split('\t')
        if len(parts) != 3:
            continue
        form, lemma, xpos = [p.strip() for p in parts]
        if not form or not xpos:
            continue
        if ' ' in xpos or not xpos[0].isalpha():
            continue
        current.append({'form': form, 'lemma': lemma, 'xpos': xpos})
    if current:
        sentences.append(current)
    return [s for s in sentences if any(t['xpos'] == target_tag for t in s)]

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Load API key
    with open(API_KEY_FILE) as f:
        api_key = f.read().strip()
    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(GEMINI_MODEL)
    print(f'Gemini client ready ({GEMINI_MODEL})')

    # Load training data
    train_sents = load_tsv(TRAIN_TSV)
    tag_counts  = Counter(tok['xpos'] for s in train_sents for tok in s)
    tag_to_sents = defaultdict(list)
    for sent in train_sents:
        seen = set()
        for tok in sent:
            if tok['xpos'] not in seen:
                tag_to_sents[tok['xpos']].append(sent)
                seen.add(tok['xpos'])

    # Load existing augmented data
    existing = load_tsv(AUG_TRAIN_TSV) + load_tsv(AUG_DEV_TSV)
    aug_so_far = Counter(tok['xpos'] for s in existing for tok in s)

    # Filter to only tags still below threshold
    todo = []
    for tag in TARGET_TAGS:
        if aug_so_far.get(tag, 0) < ADEQUATE_THRESHOLD:
            todo.append(tag)

    print(f'\nTags to augment: {len(todo)}/{len(TARGET_TAGS)}\n')

    all_new = []
    results = {}

    for idx, tag in enumerate(todo):
        existing_count = aug_so_far.get(tag, 0)
        still_need     = ADEQUATE_THRESHOLD - existing_count
        # Generate extra to account for parsing failures
        n_gen = max(SENTENCES_PER_TAG, still_need * 5)

        print(f'[{idx+1}/{len(todo)}] {tag}  (aug_so_far={existing_count}, need={still_need}) … ', end='', flush=True)

        # Get few-shot examples: direct first, then related tags
        pool = tag_to_sents.get(tag, [])
        if pool:
            shots = random.sample(pool, min(FEW_SHOT_SENTS, len(pool)))
            shots.sort(key=len)
            few_shots = [sent_to_tsv(s) for s in shots]
        else:
            related = find_related_tags(tag, tag_to_sents, max_dist=2)
            few_shots = [sent_to_tsv(s) for s in related]

        prompt  = build_prompt(tag, few_shots, n=n_gen)
        raw     = call_gemini(client, prompt)
        parsed  = parse_response(raw, tag)
        print(f'{len(parsed)} valid sentences')

        # If still not enough, retry once with a simpler fallback prompt
        if len(parsed) < still_need:
            print(f'    Retrying (got {len(parsed)}, need {still_need}) …', end=' ', flush=True)
            fallback = build_prompt(tag, [], n=still_need * 4)
            raw2    = call_gemini(client, fallback)
            parsed2 = parse_response(raw2, tag)
            print(f'{len(parsed2)} additional valid sentences')
            parsed  = parsed + parsed2

        results[tag] = parsed
        all_new.extend(parsed)
        time.sleep(SLEEP_BETWEEN)

    # Shuffle and split
    random.shuffle(all_new)
    n_dev   = max(1, math.ceil(len(all_new) * DEV_RATIO))
    new_dev = all_new[:n_dev]
    new_tr  = all_new[n_dev:]

    write_tsv(AUG_TRAIN_TSV, new_tr, append=True)
    write_tsv(AUG_DEV_TSV,   new_dev, append=True)

    print(f'\nAppended {len(new_tr)} train / {len(new_dev)} dev sentences')

    # Final coverage report
    final_aug  = load_tsv(AUG_TRAIN_TSV) + load_tsv(AUG_DEV_TSV)
    final_cnt  = Counter(tok['xpos'] for s in final_aug for tok in s)

    print(f'\n{"Tag":<24} {"aug":>5}  {"need":>5}  status')
    print('-' * 45)
    ok = needs = 0
    for tag in sorted(TARGET_TAGS):
        added = final_cnt.get(tag, 0)
        status = 'OK' if added >= ADEQUATE_THRESHOLD else f'NEEDS MORE ({added}/{ADEQUATE_THRESHOLD})'
        print(f'  {tag:<22} {added:>5}  {ADEQUATE_THRESHOLD:>5}  {status}')
        if added >= ADEQUATE_THRESHOLD:
            ok += 1
        else:
            needs += 1
    print(f'\nAdequate: {ok}/{len(TARGET_TAGS)}   Still needs: {needs}')
    if needs == 0:
        print('All target tags covered — ready to upload to Drive and retrain!')
    else:
        print('Re-run this script to fill remaining gaps.')

if __name__ == '__main__':
    main()
