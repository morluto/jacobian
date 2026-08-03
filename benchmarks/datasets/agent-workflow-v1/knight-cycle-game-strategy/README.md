# Knight-cycle game strategy

This Hard (provisional) Regression benchmark asks for matching lower and upper
strategy certificates for the 20 by 20 distance-sqrt(5) stone game from IMO
2018 problem 4.

The verifier checks all 400 sites. It independently validates a 200-site
knight-independent parity class for Amy and a partition into 100 knight-move
4-cycles with a legal opposite-vertex response for Ben. Alternative parity
classes and alternative valid cycle partitions are accepted. The finite
strategy invariant is replayed exactly; no formal proof assistant is invoked,
so assurance is capped at `COMPUTED`.
