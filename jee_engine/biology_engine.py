"""
biology_engine.py
==================
A controlled, whitelisted layer of NEET biology calculations: molecular
biology (transcription/translation/GC content), Mendelian genetics
(monohybrid/dihybrid crosses, Hardy-Weinberg equilibrium) and basic
population ecology (exponential/logistic growth).

Same contract as physics_engine.py / chemistry_engine.py: every
function takes plain arguments, validates them, and returns either
    {"success": True, "result": ..., ...}
    {"success": False, "error": "..."}
never raises, and makes ZERO network calls.

Transcription/translation/reverse-complement are delegated to
Biopython's Bio.Seq (optional dependency, same guarded-import pattern
as pint/chempy elsewhere in this package) since the standard genetic
code table is exactly the kind of fixed-but-easy-to-mistype data that
is safer to reuse from a maintained library than to hand-transcribe.
Genetics (Punnett squares, Hardy-Weinberg) and ecology (growth models)
are plain combinatorics/math and are implemented directly with no
external dependency.
"""

from __future__ import annotations

import itertools
import math
import re
from typing import Dict, List, Optional

try:
    from Bio.Seq import Seq

    BIOPYTHON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when biopython is absent
    Seq = None
    BIOPYTHON_AVAILABLE = False


MAX_SEQUENCE_LENGTH = 5000   # bases, generous for a textbook-style question
MAX_LOCI = 4                 # independently-assorting genes in a cross (2^4 = 16 gametes/parent)


def _fail(msg: str) -> dict:
    return {"success": False, "error": msg}


def _ok(**kwargs) -> dict:
    return {"success": True, **kwargs}


def _require(**kwargs):
    missing = [k for k, v in kwargs.items() if v is None]
    if missing:
        raise ValueError(f"Missing required value(s): {', '.join(missing)}")


def _require_biopython() -> Optional[dict]:
    if not BIOPYTHON_AVAILABLE:
        return _fail(
            "biopython is not installed. Run 'pip install -r requirements.txt' "
            "to enable transcription/translation."
        )
    return None


_DNA_RE = re.compile(r"^[ACGTacgt]+$")
_RNA_RE = re.compile(r"^[ACGUacgu]+$")


def _validate_sequence(seq: str, allowed_re: "re.Pattern[str]", label: str) -> str:
    if not isinstance(seq, str) or not seq.strip():
        raise ValueError(f"{label} sequence is required.")
    seq = seq.strip().upper()
    if len(seq) > MAX_SEQUENCE_LENGTH:
        raise ValueError(f"Sequence too long (max {MAX_SEQUENCE_LENGTH} bases).")
    if not allowed_re.match(seq):
        raise ValueError(f"Not a valid {label} sequence (unexpected characters).")
    return seq


# --------------------------------------------------------------------------- #
# Molecular biology
# --------------------------------------------------------------------------- #

def transcribe_dna(dna_seq: Optional[str] = None) -> dict:
    """Coding-strand DNA -> mRNA (T -> U)."""
    err = _require_biopython()
    if err:
        return err
    try:
        _require(dna_seq=dna_seq)
        seq = _validate_sequence(dna_seq, _DNA_RE, "DNA")
        mrna = str(Seq(seq).transcribe())
        return _ok(result=mrna, operation="transcribe", input=seq)
    except ValueError as e:
        return _fail(str(e))


def translate_sequence(seq_str: Optional[str] = None, is_rna: bool = False,
                        to_stop: bool = True) -> dict:
    """DNA or mRNA -> amino acid sequence using the standard genetic code."""
    err = _require_biopython()
    if err:
        return err
    try:
        _require(seq_str=seq_str)
        pattern = _RNA_RE if is_rna else _DNA_RE
        seq = _validate_sequence(seq_str, pattern, "RNA" if is_rna else "DNA")
        protein = str(Seq(seq).translate(to_stop=to_stop))
        return _ok(result=protein, operation="translate", input=seq)
    except ValueError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001 - Bio.Seq raises on e.g. length not multiple of 3
        return _fail(f"Could not translate this sequence. ({e})")


def reverse_complement(dna_seq: Optional[str] = None) -> dict:
    err = _require_biopython()
    if err:
        return err
    try:
        _require(dna_seq=dna_seq)
        seq = _validate_sequence(dna_seq, _DNA_RE, "DNA")
        rc = str(Seq(seq).reverse_complement())
        return _ok(result=rc, operation="reverse_complement", input=seq)
    except ValueError as e:
        return _fail(str(e))


def gc_content(seq_str: Optional[str] = None) -> dict:
    """Percentage of G+C bases. Works on DNA or RNA (no external dependency)."""
    try:
        _require(seq_str=seq_str)
        seq = seq_str.strip().upper()
        if not seq:
            raise ValueError("Sequence is required.")
        if len(seq) > MAX_SEQUENCE_LENGTH:
            raise ValueError(f"Sequence too long (max {MAX_SEQUENCE_LENGTH} bases).")
        if not (_DNA_RE.match(seq) or _RNA_RE.match(seq)):
            raise ValueError("Not a valid DNA/RNA sequence (unexpected characters).")
        gc = seq.count("G") + seq.count("C")
        return _ok(result=round(gc / len(seq) * 100, 4), unit="%",
                    formula="GC% = (G + C) / total * 100")
    except ValueError as e:
        return _fail(str(e))


def base_composition(seq_str: Optional[str] = None) -> dict:
    """Counts of each base plus Chargaff's-rule ratios for double-stranded DNA."""
    try:
        _require(seq_str=seq_str)
        seq = seq_str.strip().upper()
        if not seq:
            raise ValueError("Sequence is required.")
        if len(seq) > MAX_SEQUENCE_LENGTH:
            raise ValueError(f"Sequence too long (max {MAX_SEQUENCE_LENGTH} bases).")
        is_rna = "U" in seq and "T" not in seq
        pattern = _RNA_RE if is_rna else _DNA_RE
        if not pattern.match(seq):
            raise ValueError("Not a valid DNA/RNA sequence (unexpected characters).")
        counts = {b: seq.count(b) for b in ("A", "U" if is_rna else "T", "G", "C")}
        total = len(seq)
        result = {
            "counts": counts,
            "percentages": {b: round(n / total * 100, 4) for b, n in counts.items()},
        }
        if not is_rna:
            t = counts["T"] or 1
            g = counts["G"] or 1
            result["at_gc_ratio"] = round((counts["A"] + counts["T"]) / (counts["G"] + counts["C"]), 4) \
                if (counts["G"] + counts["C"]) else None
        return _ok(result=result, molecule_type="RNA" if is_rna else "DNA")
    except ValueError as e:
        return _fail(str(e))


# --------------------------------------------------------------------------- #
# Mendelian genetics
# --------------------------------------------------------------------------- #

_GENOTYPE_RE = re.compile(r"^([A-Za-z]{2})+$")


def _parse_genotype(genotype: str) -> List[str]:
    """Parse a genotype string like 'AaBb' into ['Aa', 'Bb'] (one 2-letter
    pair per locus). Each locus must use one consistent letter (upper for
    dominant, lower for recessive), e.g. 'Aa', 'BB', 'cc'."""
    if not isinstance(genotype, str) or not genotype:
        raise ValueError("Genotype is required.")
    genotype = genotype.strip()
    if len(genotype) > MAX_LOCI * 2 or len(genotype) % 2 != 0:
        raise ValueError(f"Genotype must be pairs of letters, up to {MAX_LOCI} loci.")
    if not _GENOTYPE_RE.match(genotype):
        raise ValueError("Genotype must contain only letters, in pairs (e.g. 'AaBb').")
    loci = [genotype[i:i + 2] for i in range(0, len(genotype), 2)]
    for pair in loci:
        if pair[0].lower() != pair[1].lower():
            raise ValueError(f"Locus '{pair}' mixes two different genes; use one letter per gene.")
    return loci


def _gametes(loci: List[str]) -> List[str]:
    """All possible gametes for one parent's genotype loci, e.g.
    ['Aa','Bb'] -> ['AB','Ab','aB','ab'] (independent assortment)."""
    per_locus_alleles = [sorted(set(pair)) for pair in loci]
    combos = itertools.product(*per_locus_alleles)
    return ["".join(c) for c in combos]


def _offspring_genotype(g1: str, g2: str) -> str:
    """Combine one gamete from each parent into a sorted genotype string
    per locus, dominant allele first (e.g. 'A'+'a' -> 'Aa')."""
    pairs = []
    for a1, a2 in zip(g1, g2):
        pairs.append("".join(sorted((a1, a2), key=lambda c: (0 if c.isupper() else 1, c))))
    return "".join(pairs)


def _phenotype(genotype_pairs: str) -> str:
    """Dominant/recessive phenotype string: 'Aa' -> dominant trait letter,
    'aa' -> recessive trait letter, one letter per locus."""
    n_loci = len(genotype_pairs) // 2
    out = []
    for i in range(n_loci):
        pair = genotype_pairs[2 * i:2 * i + 2]
        has_dominant = any(c.isupper() for c in pair)
        out.append(pair[0].upper() if has_dominant else pair[0].lower())
    return "".join(out)


def cross(parent1: Optional[str] = None, parent2: Optional[str] = None) -> dict:
    """General Punnett-square cross supporting 1-4 independently assorting
    loci, e.g. cross('Aa','Aa') for a monohybrid cross or
    cross('AaBb','AaBb') for a dihybrid cross. Assumes simple complete
    dominance and independent assortment (no linkage)."""
    try:
        _require(parent1=parent1, parent2=parent2)
        loci1 = _parse_genotype(parent1)
        loci2 = _parse_genotype(parent2)
        if len(loci1) != len(loci2):
            raise ValueError("Both parents must have the same number of gene loci.")

        gametes1 = _gametes(loci1)
        gametes2 = _gametes(loci2)

        genotype_counts: Dict[str, int] = {}
        phenotype_counts: Dict[str, int] = {}
        for g1, g2 in itertools.product(gametes1, gametes2):
            offspring = _offspring_genotype(g1, g2)
            genotype_counts[offspring] = genotype_counts.get(offspring, 0) + 1
            pheno = _phenotype(offspring)
            phenotype_counts[pheno] = phenotype_counts.get(pheno, 0) + 1

        total = len(gametes1) * len(gametes2)

        def _ratio(counts: Dict[str, int]) -> Dict[str, str]:
            g = math.gcd(*counts.values()) if len(counts) > 1 else counts[next(iter(counts))]
            return {k: f"{v // g}" for k, v in counts.items()}

        return _ok(
            result={
                "genotype_counts": genotype_counts,
                "genotype_ratio": _ratio(genotype_counts),
                "phenotype_counts": phenotype_counts,
                "phenotype_ratio": _ratio(phenotype_counts),
                "total_combinations": total,
            },
            parent1=parent1, parent2=parent2,
        )
    except ValueError as e:
        return _fail(str(e))


def offspring_probability(single_trial_probability: Optional[float] = None,
                           n_offspring: Optional[int] = None,
                           exactly: Optional[int] = None) -> dict:
    """Binomial probability of exactly k offspring showing a given
    phenotype out of n, given the per-offspring probability p (e.g. 3/4
    for a dominant trait from an Aa x Aa cross)."""
    try:
        _require(single_trial_probability=single_trial_probability,
                 n_offspring=n_offspring, exactly=exactly)
        p = single_trial_probability
        n = int(n_offspring)
        k = int(exactly)
        if not (0 <= p <= 1):
            raise ValueError("single_trial_probability must be between 0 and 1.")
        if n < 1 or n > 200:
            raise ValueError("n_offspring must be between 1 and 200.")
        if k < 0 or k > n:
            raise ValueError("exactly must be between 0 and n_offspring.")
        prob = math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
        return _ok(result=round(prob, 10), unit="probability",
                    formula="P = C(n,k) * p^k * (1-p)^(n-k)")
    except ValueError as e:
        return _fail(str(e))


# --------------------------------------------------------------------------- #
# Population genetics - Hardy-Weinberg equilibrium
# --------------------------------------------------------------------------- #

def hardy_weinberg(p: Optional[float] = None, q: Optional[float] = None,
                    recessive_phenotype_freq: Optional[float] = None) -> dict:
    """p = dominant allele frequency, q = recessive allele frequency
    (p + q = 1). Provide any ONE of p, q, or recessive_phenotype_freq
    (= q^2) and the rest are derived."""
    try:
        provided = [x for x in (p, q, recessive_phenotype_freq) if x is not None]
        if len(provided) != 1:
            raise ValueError("Provide exactly one of: p, q, recessive_phenotype_freq.")
        if p is not None:
            if not (0 <= p <= 1):
                raise ValueError("p must be between 0 and 1.")
            q = 1 - p
        elif q is not None:
            if not (0 <= q <= 1):
                raise ValueError("q must be between 0 and 1.")
            p = 1 - q
        else:
            if not (0 <= recessive_phenotype_freq <= 1):
                raise ValueError("recessive_phenotype_freq must be between 0 and 1.")
            q = math.sqrt(recessive_phenotype_freq)
            p = 1 - q

        return _ok(
            result={
                "p": round(p, 6),
                "q": round(q, 6),
                "homozygous_dominant_freq_p2": round(p ** 2, 6),
                "heterozygous_freq_2pq": round(2 * p * q, 6),
                "homozygous_recessive_freq_q2": round(q ** 2, 6),
            },
            formula="p + q = 1; p^2 + 2pq + q^2 = 1",
        )
    except ValueError as e:
        return _fail(str(e))


# --------------------------------------------------------------------------- #
# Population ecology
# --------------------------------------------------------------------------- #

def exponential_growth(n0: Optional[float] = None, r: Optional[float] = None,
                        t: Optional[float] = None) -> dict:
    try:
        _require(n0=n0, r=r, t=t)
        if n0 < 0:
            raise ValueError("n0 cannot be negative.")
        result = n0 * math.exp(r * t)
        return _ok(result=round(result, 6), unit="population size",
                    formula="Nt = N0 * e^(r*t)")
    except ValueError as e:
        return _fail(str(e))


def logistic_growth(n0: Optional[float] = None, r: Optional[float] = None,
                     k: Optional[float] = None, t: Optional[float] = None) -> dict:
    try:
        _require(n0=n0, r=r, k=k, t=t)
        if n0 <= 0 or k <= 0:
            raise ValueError("n0 and k (carrying capacity) must be positive.")
        result = k / (1 + ((k - n0) / n0) * math.exp(-r * t))
        return _ok(result=round(result, 6), unit="population size",
                    formula="Nt = K / (1 + ((K-N0)/N0) * e^(-r*t))")
    except ValueError as e:
        return _fail(str(e))
