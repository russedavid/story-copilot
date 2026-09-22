# Source-grounded response review

The editing stage checks consequential claims across narration, direct answers, questions, requested checks, and private notes. A general “no issues found” verdict is insufficient when the draft contains an identified player or numerical-rule claim.

Each required claim gets one assessment. A supported claim needs an exact quote from a supplied source. Generated feedback and earlier drafts cannot provide that support. A numerical rule assertion needs a rule source; an unanswered question or resource record is insufficient. Explicit zero modifiers require explicit zero-valued rule evidence. An absent rule does not establish zero.

Declarative statements beginning with a player’s name must preserve the wording of one supporting source. This is deliberately conservative: a paraphrase must not add a gesture or feeling. The editor may remove an unnecessary player description while retaining creative NPC speech and world reactions. Exact quotation still does not prove all possible implications, actor identities, or temporal interpretations; semantic evaluation remains necessary.

The stage allows at most two model calls within one deadline. A repair that adds new consequential claims is checked again. If verification cannot finish, affected fields are withheld and the remaining response is marked partial; an empty or unhelpful answer is not an evaluation success. The trace retains the original, assessments, repairs, failures, and call accounting.

Replay saved response/context pairs against a configured model:

```sh
python -m story_copilot.review_regressions \
  --cases /path/to/private/cases.json --home /path/to/workspace \
  --output /path/to/new/private/review
```

The cases file is an array of objects with `id`, `context` (the exact original writer input), `answer` (the structured response), and `expect` (a source-based review criterion). A reference response is not the only valid wording. Include supported player restatements, useful NPC invention, legitimate questions, and explicit zero rules alongside negative cases to detect over-restriction. Keep new training data separate from these regression and evaluation cases.
