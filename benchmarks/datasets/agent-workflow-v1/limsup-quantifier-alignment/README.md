# Limsup quantifier-alignment audit

This Assurance benchmark freezes the semantic defect reported in google-deepmind/formal-conjectures issue #1347 against commit `37a34a76ea03c9ab157981bf4495a8c3a1add68a`.

The informal optimization claim says that some admissible object has limsup at most a bound. The proposed formalization instead says every admissible object has limsup at least that bound. A submission must construct two exact rational model families separating the formulas in both directions. The verifier evaluates both formulas itself and accepts alternative separating families.

Difficulty is **Hard (provisional)**: the task requires quantifier, polarity, and optimization-direction analysis plus two independent countermodels. Baseline calibration has not yet been run. The task is an Assurance audit, not evidence that an agent can solve Erdős problem 33.

