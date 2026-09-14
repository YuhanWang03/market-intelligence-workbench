"""Citation-integrity verification for Agent V2 answers.

The verifier is domain-neutral.  Beyond the universal checks (every citation
must exist, every figure must trace to evidence, a citation must support the
figures next to it) it enforces rules that adapters attach to their own
evidence and results:

``EvidenceItem.metadata``
    ``constraints``: list of ``{"require": regex, "warning": str}`` or
    ``{"forbid": regex, "unless": regex | None, "warning": str}``; each applies
    to the sentence that cites the item.
    ``citable``: ``False`` marks evidence the answer must not cite; the warning
    text comes from ``uncitable_warning``.

``ToolEnvelope.metadata``
    ``require_cited_numbers``: every sentence with a figure needs a citation.
    ``{"forbid_claim": sentence, "warning": str}`` (on evidence or on the
    answer): the text must not assert the sentence; judged by the model in
    one call when a ``judge`` is given, so paraphrases are caught too.
    ``answer_constraints``: list of ``{"forbid": regex, "warning": str}`` for
    the whole answer, ``{"max_cited": {"metadata": {...}, "max": n,
    "warning": str}}`` capping how many of *this result's* items with matching
    metadata may be cited, or ``{"require_cited": {"metadata": {...},
    "warning": str}}`` demanding that at least one of them is.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from v2.agent_v2.models import AnswerMode, EvidenceItem, ToolEnvelope, VerificationReport

#: A ``forbid_claim`` rule marked ``"soft": True`` produces a warning with this
#: prefix; it is reported but does not fail verification.
SOFT_PREFIX = "（提示）"

_CITATION = re.compile(r"\[([A-Za-z0-9_.:-]+)\]")
_SENTENCE = re.compile(r"[^。！？!?\n]+(?:[。！？!?]+|$)(?:\s*\[[A-Za-z0-9_.:-]+\])*")


#: ``judge(items) -> {id: quote}``: which of the ``{"id", "text", "claim"}`` items assert their claim.
Judge = Callable[[list[dict[str, str]]], dict[str, str]]


def verify_answer(
    answer: str,
    evidence: list[EvidenceItem],
    *,
    answer_mode: AnswerMode,
    results: list[ToolEnvelope] | None = None,
    judge: Judge | None = None,
) -> VerificationReport:
    known = {item.id for item in evidence}
    cited = set(_CITATION.findall(answer or ""))
    unknown = tuple(sorted(cited - known))
    warnings: list[str] = []
    ungrounded: tuple[str, ...] = ()
    traced: tuple[str, ...] = ()
    if answer_mode not in {AnswerMode.GENERAL_KNOWLEDGE, AnswerMode.INSUFFICIENT_EVIDENCE}:
        if evidence and not cited:
            warnings.append("answer contains evidence but cites none of it")
        if not evidence and answer:
            warnings.append("grounded answer has no structured evidence")
        if evidence:
            from v2.agent_common import grounding

            answer_without_citations = _CITATION.sub("", answer or "")
            observations = _observations(evidence) + "\n" + json.dumps(
                [
                    {
                        "summary": result.summary,
                        "metrics": result.metrics,
                        "findings": _without_identifiers(result.findings),
                        "limitations": result.limitations,
                    }
                    for result in results or []
                ],
                ensure_ascii=False,
                default=str,
            )
            report = grounding.check(answer_without_citations, observations)
            ungrounded = tuple(report.ungrounded)
            traced = tuple(dict.fromkeys(report.traced))
            claims: list[dict[str, str]] = []
            warnings.extend(_sentence_warnings(answer or "", evidence, results or [], grounding, claims))
            warnings.extend(_answer_warnings(answer or "", evidence, results or [], claims))
            warnings.extend(_judged_warnings(claims, judge))
    blocking = [warning for warning in warnings if not warning.startswith(SOFT_PREFIX)]
    return VerificationReport(
        ok=not unknown and not blocking and not ungrounded,
        unknown_citations=unknown,
        ungrounded_numbers=ungrounded,
        traced_numbers=traced,
        warnings=tuple(dict.fromkeys(warnings)),
    )


_DIGITS = re.compile(r"\d")


def _significant_digits(token: str) -> int:
    digits = _DIGITS.findall(str(token))
    return len("".join(digits).lstrip("0"))


def _agree(items: list[EvidenceItem]) -> bool:
    """Several carriers of one figure are as good as one when they describe the same thing on the same day."""

    entities = {(item.entity or "").upper() for item in items}
    days = {str(item.metadata.get("date") or item.as_of or "")[:10] for item in items}
    return len(entities) == 1 and "" not in entities and len(days) == 1 and "" not in days


def complete_citations(answer: str, evidence: list[EvidenceItem], results: list[ToolEnvelope] | None = None) -> tuple[str, list[str]]:
    """Add the one citation a sentence is missing when the evidence leaves no doubt.

    A model that rewrites a paragraph tends to move a citation one sentence
    over; the figures are right, the id next to them is not.  For each
    sentence whose cited items do not carry one of its figures, when that
    figure has at least three significant digits and exactly one citable
    item in this run's evidence carries it, that item's id is appended to
    the sentence.  Existing citations stay; vaguer figures (a year, "3") and
    figures several items carry are left for the model's repair round.

    Returns the completed text and one note per completion.
    """

    if not answer or not evidence:
        return answer or "", []
    from v2.agent_common import grounding

    known = {item.id: item for item in evidence}
    candidates = [item for item in evidence if item.metadata.get("citable", True)]
    notes: list[str] = []
    pieces: list[str] = []
    cursor = 0
    for match in _SENTENCE.finditer(answer):
        raw = match.group(0)
        cited = [known[value] for value in _CITATION.findall(raw) if value in known]
        plain = _CITATION.sub("", raw)
        # A sentence with no citation at all is completed only when every
        # one of its precise figures has a single carrier; a cited sentence
        # gets the figures that can be placed, the rest go to the repair.
        local = grounding.check(plain, _observations(cited) if cited else "")
        if not local.ungrounded:
            continue
        additions: list[str] = []
        placed: list[tuple[str, str]] = []  # (figure, carrier id)
        unplaced = False
        for token in dict.fromkeys(local.ungrounded):
            carriers = [] if _significant_digits(token) < 3 else [item for item in candidates if item not in cited and not grounding.check(token, _observations([item])).ungrounded]
            if not carriers or (len(carriers) > 1 and not _agree(carriers)):
                unplaced = True
                continue
            placed.append((token, carriers[0].id))
            if carriers[0].id not in additions:
                additions.append(carriers[0].id)
        if not additions or (not cited and unplaced):
            continue
        # The completed sentence must ground every figure that was placed.
        completed_items = cited + [known[value] for value in additions]
        remaining = set(grounding.check(plain, _observations(completed_items)).ungrounded)
        if any(token in remaining for token, _ in placed):
            continue
        notes.extend(f"{token} → [{carrier}]" for token, carrier in placed)
        # Splice the ids in before the sentence's closing punctuation, next
        # to the citation that was there.
        stripped = raw.rstrip()
        trailing = raw[len(stripped):]
        body = stripped.rstrip("。！？!?")
        closing = stripped[len(body):]
        pieces.append(answer[cursor : match.start()])
        pieces.append(body + "".join(f"[{value}]" for value in additions) + closing + trailing)
        cursor = match.end()
    if not pieces:
        return answer, []
    pieces.append(answer[cursor:])
    return "".join(pieces), notes


def _observations(items: list[EvidenceItem]) -> str:
    return "\n".join(
        " ".join(
            value
            for value in (
                item.claim,
                str(item.value) if item.value is not None else "",
                json.dumps(item.metadata, ensure_ascii=False, default=str),
            )
            if value
        )
        for item in items
    )


def locate_number(number: str, evidence: list[EvidenceItem], *, limit: int = 3) -> list[str]:
    """Ids of the evidence items whose observations contain ``number`` verbatim."""

    needle = str(number).strip()
    if not needle:
        return []
    found = [item.id for item in evidence if needle in _observations([item])]
    return found[:limit]


def _without_identifiers(value):
    """Remove opaque IDs before numeric grounding; digits inside IDs are not facts."""

    if isinstance(value, dict):
        return {
            key: _without_identifiers(item)
            for key, item in value.items()
            if not str(key).lower().endswith(("_id", "_ids")) and str(key).lower() != "id"
        }
    if isinstance(value, list):
        return [_without_identifiers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_without_identifiers(item) for item in value)
    return value


def _matches(pattern: Any, text: str) -> bool:
    return bool(pattern) and re.search(str(pattern), text) is not None


def _figure_key(value: str) -> str:
    """A figure as written, without sign, unit or separators: "-39.10%" and "39.10" are the same restatement."""

    return re.sub(r"[^\d.]", "", str(value)).rstrip(".")


def _excerpt(sentence: str, limit: int = 30) -> str:
    """The start of a sentence, enough to find it in the draft."""

    flat = " ".join(sentence.split())
    return flat if len(flat) <= limit else flat[:limit].rstrip() + "…"


def _sentence_warnings(answer: str, evidence: list[EvidenceItem], results: list[ToolEnvelope], grounding, claims: list[dict[str, str]] | None = None) -> list[str]:
    """Require nearby citations to support nearby figures and honour evidence-level rules.

    ``claims`` collects ``forbid_claim`` rules (a sentence in plain language the
    cited text must not assert) for the judge; they are decided in one call by
    the caller, not here.
    """

    known = {item.id: item for item in evidence}
    require_cited_numbers = any(result.metadata.get("require_cited_numbers") for result in results)
    warnings: list[str] = []
    seen_claims: set[tuple[str, str]] = set()
    # Figures an earlier sentence already tied to its evidence: a later sentence
    # may restate them ("所以 -39.10% 说的是那段回撤") without citing again.
    grounded_earlier: set[str] = set()
    for raw_sentence in _SENTENCE.findall(answer):
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        cited_items = [known[value] for value in _CITATION.findall(sentence) if value in known]
        plain = _CITATION.sub("", sentence)
        if not cited_items:
            bare = grounding.check(plain, "")
            if require_cited_numbers and bare.total and any(_figure_key(value) not in grounded_earlier for value in bare.ungrounded):
                warnings.append(f"行情事实缺少邻近引用：“{_excerpt(plain)}”")
            continue
        local = grounding.check(plain, _observations(cited_items))
        unsupported = [value for value in local.ungrounded if _figure_key(value) not in grounded_earlier]
        grounded_earlier.update(_figure_key(value) for value in local.traced)
        if unsupported:
            warnings.append("引用未支持邻近数字：" + ", ".join(unsupported[:4]) + f"（“{_excerpt(plain)}”）")
        for item in cited_items:
            if not item.metadata.get("citable", True):
                warnings.append(str(item.metadata.get("uncitable_warning") or f"引用了不可展示的证据：{item.id}"))
            for rule in item.metadata.get("constraints") or []:
                if not isinstance(rule, dict):
                    continue
                warning = str(rule.get("warning") or f"证据 {item.id} 的表述规则未满足")
                if rule.get("require") and not _matches(rule["require"], plain):
                    warnings.append(warning)
                if rule.get("forbid") and _matches(rule["forbid"], plain) and not _matches(rule.get("unless"), plain):
                    warnings.append(warning)
                claim = rule.get("forbid_claim")
                if claim and claims is not None and (item.id, str(claim)) not in seen_claims:
                    # The judge sees every sentence citing this item at once.
                    seen_claims.add((item.id, str(claim)))
                    citing = [_CITATION.sub("", s).strip() for s in _SENTENCE.findall(answer) if item.id in _CITATION.findall(s)]
                    # The judge sees the evidence the rule is about, so a sentence
                    # that also cites a confirmed driver is not read as calling
                    # this candidate confirmed.
                    claims.append({"id": f"evidence:{item.id}:{len(claims)}", "text": f"证据：{item.claim[:300]}\n回答：{' '.join(citing)}", "claim": str(claim), "warning": warning, "soft": bool(rule.get("soft"))})
    return warnings


def _judged_warnings(claims: list[dict[str, str]], judge: Judge | None) -> list[str]:
    """One judge call for every ``forbid_claim`` rule the answer triggered; without a judge the rules do not apply."""

    if not claims or judge is None:
        return []
    asserted = judge([{"id": row["id"], "text": row["text"], "claim": row["claim"]} for row in claims])
    warnings: list[str] = []
    for row in claims:
        quote = asserted.get(row["id"])
        if quote is not None:
            # A soft rule is advice for the next draft, not grounds to reject this one.
            warnings.append((SOFT_PREFIX if row.get("soft") else "") + row["warning"] + (f"（“{quote}”）" if quote else ""))
    return warnings


def _answer_warnings(answer: str, evidence: list[EvidenceItem], results: list[ToolEnvelope], claims: list[dict[str, str]] | None = None) -> list[str]:
    """Apply result-level rules that look at the answer as a whole (``forbid_claim`` rules go to ``claims`` for the judge)."""

    known = {item.id: item for item in evidence}
    cited = [known[value] for value in _CITATION.findall(answer) if value in known]
    warnings: list[str] = []
    plain_answer = _CITATION.sub("", answer)
    for result in results:
        own = {item.id for item in result.evidence}
        for rule in result.metadata.get("answer_constraints") or []:
            if not isinstance(rule, dict):
                continue
            if rule.get("forbid") and _matches(rule["forbid"], answer):
                warnings.append(str(rule.get("warning") or f"{result.capability} 的回答规则未满足"))
            if rule.get("forbid_claim") and claims is not None:
                claims.append({"id": f"answer:{result.capability}:{len(claims)}", "text": plain_answer, "claim": str(rule["forbid_claim"]), "warning": str(rule.get("warning") or f"{result.capability} 的回答规则未满足"), "soft": bool(rule.get("soft"))})
            cap = rule.get("max_cited")
            if isinstance(cap, dict):
                wanted = dict(cap.get("metadata") or {})
                # A cap is about this result's evidence: six tickers may each show one candidate.
                matching = list(dict.fromkeys(item.id for item in cited if item.id in own and all(item.metadata.get(key) == value for key, value in wanted.items())))
                limit = int(cap.get("max", 0))
                if len(matching) > limit:
                    # Name what to keep so one repair can comply: the first cited ones stay, the rest go.
                    keep, drop = matching[:limit], matching[limit:]
                    # Soft: a second candidate for a day is wording to trim, not a fabricated
                    # figure; the judged "candidate stated as confirmed" rule still blocks.
                    warnings.append(SOFT_PREFIX + str(cap.get("warning") or f"{result.capability} 引用了过多同类证据") + f"（{result.subject or result.capability}：只保留 " + "、".join(f"[{value}]" for value in keep) + "，去掉 " + "、".join(f"[{value}]" for value in drop) + "）")
            quoted = rule.get("quote_of")
            if quoted and rule.get("require") and any(item.id == quoted for item in cited) and not _matches(rule["require"], plain_answer):
                # A cited finding's quote must appear somewhere in the answer, not in
                # every sentence that cites it: the summary paragraph restates it bare.
                warnings.append(str(rule.get("warning") or f"引用了 [{quoted}] 却没有照抄它的引文"))
            need = rule.get("require_cited")
            if isinstance(need, dict):
                # A floor: at least one of this result's items with the given
                # metadata must be cited (the sector comparison of a drawdown).
                wanted = dict(need.get("metadata") or {})
                available = [item for item in result.evidence if all(item.metadata.get(key) == value for key, value in wanted.items())]
                if available and not any(item.id in {value.id for value in cited} for item in available):
                    warnings.append(str(need.get("warning") or f"{result.capability} 的关键证据未被引用：" + "、".join(f"[{item.id}]" for item in available[:3])))
    return warnings
