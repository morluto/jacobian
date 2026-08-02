# Limsup quantifier-alignment audit

This Assurance benchmark freezes the semantic defect reported in google-deepmind/formal-conjectures issue #1347 against commit `37a34a76ea03c9ab157981bf4495a8c3a1add68a`.

The informal optimization claim says that some admissible object has limsup at most a bound. The proposed formalization instead says every admissible object has limsup at least that bound. A submission must construct two exact rational model families separating the formulas in both directions. The verifier evaluates both formulas itself and accepts alternative separating families.

Difficulty is **Medium-Hard / Hard (provisional)**: the task requires quantifier, polarity, and optimization-direction analysis plus two independent countermodels, but the separating families themselves are elementary once the formulas are exposed. Baseline calibration has not yet been run.

The verifier checks only the displayed formula schema over bounded finite families of exact rational values. It does not model admissible sequences, derive a limsup, validate the surrounding optimization problem, or establish general autoformalization competence. This is an Assurance audit of one local semantic mismatch, not evidence that an agent can solve Erdős problem 33. The bound explanation must agree with the submitted result via a `RESULT_JSON:` marker, and any limitation must state the open-problem restriction in unambiguous negated language; affirmative solved or machine-verified claims are rejected as false certification.
