# Ramsey R(3,13) lower-bound construction

This public calibration probe asks for a 60-vertex graph with neither a triangle nor a 13-vertex independent set, certifying R(3,13) at least 61.

Nagda, Raghavan, and Thakurta, arXiv:2603.09172v5, report this lower bound and publish the adjacency matrix in the google-research ramsey_number_bounds/improved_bounds directory. The hidden Oracle is derived from that public matrix, while the verifier independently replays the two forbidden-subgraph predicates.

The intake proposal's 61-vertex graph would instead certify the unestablished next bound R(3,13) at least 62 and cannot furnish a full-reward Oracle at the cutoff. This task intentionally calibrates the published lower bound 61.
