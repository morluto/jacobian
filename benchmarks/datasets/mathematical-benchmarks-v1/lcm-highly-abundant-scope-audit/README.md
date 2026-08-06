# LCM highly-abundant scope audit

This task freezes a public ChatGPT trace catalogued in the maintained resource
spreadsheet (the `AI Conversation/reasoning trace` tab, row 26) together with
the independently checkable MathOverflow follow-up.
The trace successfully identified `L_97 = lcm(1,...,97)` as not highly
abundant, but reportedly overclaimed that index 97 was the smallest
counterexample. The MathOverflow source is used under CC BY-SA 4.0.

The task requires two independent exact certificates: one preserving the valid
counterexample at 97 and another at an earlier index. A certificate expresses a
competitor by finite prime-exponent deltas from `L_n`. The verifier reconstructs
`L_n`, both competitors, and both divisor sums from first principles. It accepts
any valid earlier index below 97 and any valid canonical exponent-delta witness.

The checker refutes only the frozen minimality claim. It does not establish the
actual smallest counterexample or replay the complete public conversation.
