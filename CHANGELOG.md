# Changelog

## [0.13.0](https://github.com/morluto/jacobian/compare/jacobian-v0.12.0...jacobian-v0.13.0) (2026-08-20)


### ⚠ Breaking Changes

* **runtime:** remove stateful commands and APIs, including `jacobian init`,
  `jacobian update`, `--state-dir`, workspaces, artifacts, value references,
  and checker/runtime interfaces. The stateless kernel now accepts one complete
  request per `math.run` call; callers own composition and durable state
  ([0ff90ee](https://github.com/morluto/jacobian/commit/0ff90ee22d2edd1b78a38041c5a92a0baf13dade),
  [62ce9d3](https://github.com/morluto/jacobian/commit/62ce9d371fc5501c0eb7a0843c4d757048a8e974)).
* **catalog:** curate the public operation basis and remove legacy operation
  IDs. Compatibility aliases are not provided; use `math.find` to locate a
  retained operation before replacing an old `math.run` call
  ([dd341af](https://github.com/morluto/jacobian/commit/dd341afbdba42186a4b511b830335ac62f21d05d)).


### Features

* **math:** add permanent, Kronecker product, partial trace, Walsh transform,
  and symbolic matrix operations ([#1651](https://github.com/morluto/jacobian/issues/1651))
  ([1ca7143](https://github.com/morluto/jacobian/commit/1ca7143be800e3973507b472e3dab0b15eeff44c)).
* **math:** add exact bounded operations across code theory, graph coloring,
  flows and spectra, groups, Markov chains, number fields, recurrence solving,
  and root isolation ([#1683](https://github.com/morluto/jacobian/issues/1683))
  ([a6adff8](https://github.com/morluto/jacobian/commit/a6adff87a2be8711fe0062b143bdc7af9add2af8)).
* **math:** add bounded composable operations across arithmetic, combinatorics,
  geometry, graphs, matrices, polynomials, and finite structures
  ([#1817](https://github.com/morluto/jacobian/issues/1817))
  ([68df662](https://github.com/morluto/jacobian/commit/68df66239fcce898a0087d72bae795904b126b00)).
* **graph:** add an exact maximal independent-set decision
  ([#1774](https://github.com/morluto/jacobian/issues/1774))
  ([7c84d1e](https://github.com/morluto/jacobian/commit/7c84d1e501e91e9438a418a96be47b52a6b3ee37)).
* **code-theory:** add a bounded exact code-covering radius operation
  ([#1721](https://github.com/morluto/jacobian/issues/1721))
  ([5d73f35](https://github.com/morluto/jacobian/commit/5d73f35c049d99fc2f58d19a65d488717b060c79)).
* **analysis:** add bounded Arb expression enclosures ([#2098](https://github.com/morluto/jacobian/issues/2098)) ([8a0c635](https://github.com/morluto/jacobian/commit/8a0c635b39af653f643ada288e20b860027d5037))
* **markov-chain:** add exact bounded mixing time ([#2100](https://github.com/morluto/jacobian/issues/2100)) ([f2fcd06](https://github.com/morluto/jacobian/commit/f2fcd0693c4c3ec61b31d9d66cd4d25cced83895))
* **math:** add algebraic topology operations ([#1878](https://github.com/morluto/jacobian/issues/1878)) ([#2061](https://github.com/morluto/jacobian/issues/2061)) ([213c37a](https://github.com/morluto/jacobian/commit/213c37a368e581379447202fd148681bf7ba75dc))
* **math:** add chip-firing domain with Laplacian and vertex firing ([#1741](https://github.com/morluto/jacobian/issues/1741)) ([#2017](https://github.com/morluto/jacobian/issues/2017)) ([52d9f33](https://github.com/morluto/jacobian/commit/52d9f33b8afa8f425a11bd1aa41097857b98ccc0))
* **math:** add combinatorial maps domain with faces and dual operations ([#1728](https://github.com/morluto/jacobian/issues/1728)) ([1232975](https://github.com/morluto/jacobian/commit/123297549123f91b16fd503b4a55f30a05578d12))
* **math:** add combinatorial matrices domain with Hadamard profiles, normalization, determinant, and Sylvester construction ([#1784](https://github.com/morluto/jacobian/issues/1784)) ([#2033](https://github.com/morluto/jacobian/issues/2033)) ([684b0ef](https://github.com/morluto/jacobian/commit/684b0eff475c11e7e0c4c969a98d5cb50c2aa24e))
* **math:** add commutative algebra operations ([#1869](https://github.com/morluto/jacobian/issues/1869)) ([#2055](https://github.com/morluto/jacobian/issues/2055)) ([6421fd8](https://github.com/morluto/jacobian/commit/6421fd84b9fc8ccbb98475401e73aa143cd08129))
* **math:** add context-free language operations ([#1894](https://github.com/morluto/jacobian/issues/1894)) ([#2051](https://github.com/morluto/jacobian/issues/2051)) ([55daad2](https://github.com/morluto/jacobian/commit/55daad2690a7eec4a85cdb593a5f60b944086bdb))
* **math:** add cubical complexes domain with f-vector and face closure ([#1888](https://github.com/morluto/jacobian/issues/1888)) ([#2020](https://github.com/morluto/jacobian/issues/2020)) ([772e960](https://github.com/morluto/jacobian/commit/772e960236db48c7babb4d2cba4349b71d51b961))
* **math:** add dual code and syndrome operations using SymPy ([#1851](https://github.com/morluto/jacobian/issues/1851)) ([#2026](https://github.com/morluto/jacobian/issues/2026)) ([e4b39e8](https://github.com/morluto/jacobian/commit/e4b39e88537eaf93dcfe0f53563dd5ea512db407))
* **math:** add finite categories domain with profile and opposite ([#1733](https://github.com/morluto/jacobian/issues/1733)) ([#2023](https://github.com/morluto/jacobian/issues/2023)) ([95b6eee](https://github.com/morluto/jacobian/commit/95b6eee85e345cf3458d35092b1e7812905a2784))
* **math:** add finite geometry operations ([#1853](https://github.com/morluto/jacobian/issues/1853)) ([#2041](https://github.com/morluto/jacobian/issues/2041)) ([7388708](https://github.com/morluto/jacobian/commit/7388708d154a4200229e40da24287c8156b34099))
* **math:** add finite semigroup domain with power profiles and generated subsemigroups ([#1858](https://github.com/morluto/jacobian/issues/1858)) ([#2011](https://github.com/morluto/jacobian/issues/2011)) ([da7b42f](https://github.com/morluto/jacobian/commit/da7b42f759667630d6a9221f0b20a5b881bc2cc5))
* **math:** add finite stochastic processes domain with sigma algebras, conditional expectation, filtrations, and Doob martingales ([#1806](https://github.com/morluto/jacobian/issues/1806)) ([#2036](https://github.com/morluto/jacobian/issues/2036)) ([739ff6a](https://github.com/morluto/jacobian/commit/739ff6a2dc694027bb3afae6d0a50883002b11f2))
* **math:** add finite topological spaces domain with interior, closure, boundary, Kolmogorov quotient, and continuity check ([#1920](https://github.com/morluto/jacobian/issues/1920)) ([#2037](https://github.com/morluto/jacobian/issues/2037)) ([1e74773](https://github.com/morluto/jacobian/commit/1e74773c5375e61bcbef9263a6b738ce9acea66f))
* **math:** add finite-dimensional algebra operations ([#1875](https://github.com/morluto/jacobian/issues/1875)) ([#2056](https://github.com/morluto/jacobian/issues/2056)) ([c51dcdf](https://github.com/morluto/jacobian/commit/c51dcdfe1c638dd116c2612a3de91b1e72dbd20c))
* **math:** add finitely generated abelian group operations ([#1857](https://github.com/morluto/jacobian/issues/1857)) ([#2044](https://github.com/morluto/jacobian/issues/2044)) ([6e67391](https://github.com/morluto/jacobian/commit/6e67391dfd69ba1d837f0ab6a9d50833803e626e))
* **math:** add formal concept analysis domain with derivation, closure, concepts, and lattice ([#1896](https://github.com/morluto/jacobian/issues/1896)) ([#2035](https://github.com/morluto/jacobian/issues/2035)) ([280df56](https://github.com/morluto/jacobian/commit/280df56cf5a407783165010d08d6512c8bd5b44f))
* **math:** add Galois theory operations ([#1862](https://github.com/morluto/jacobian/issues/1862)) ([#2047](https://github.com/morluto/jacobian/issues/2047)) ([dcd7286](https://github.com/morluto/jacobian/commit/dcd728625f27b784c8163e01eea9a5d78134a08e))
* **math:** add graph morphism operations ([#1856](https://github.com/morluto/jacobian/issues/1856)) ([#2043](https://github.com/morluto/jacobian/issues/2043)) ([3892666](https://github.com/morluto/jacobian/commit/38926667a7cf5499b0294737b346476a140d662f))
* **math:** add greedoids domain with recognize, rank, bases, basic words, convex geometry ([#1915](https://github.com/morluto/jacobian/issues/1915)) ([#2031](https://github.com/morluto/jacobian/issues/2031)) ([33650ef](https://github.com/morluto/jacobian/commit/33650ef0588c076f13608e6cdda4c4ad2682333e))
* **math:** add hyperplane arrangement operations ([#1885](https://github.com/morluto/jacobian/issues/1885)) ([#2052](https://github.com/morluto/jacobian/issues/2052)) ([0e6cb64](https://github.com/morluto/jacobian/commit/0e6cb64a0ce6e7e29a29629cc3697c0fd6bfbd27))
* **math:** add incidence structure domain with matrix and degree profiles ([#1732](https://github.com/morluto/jacobian/issues/1732)) ([#2018](https://github.com/morluto/jacobian/issues/2018)) ([947704d](https://github.com/morluto/jacobian/commit/947704d6edb725b283b6d119e02419799547ed56))
* **math:** add integer multiplicative normal-form operations ([#1893](https://github.com/morluto/jacobian/issues/1893)) ([#2009](https://github.com/morluto/jacobian/issues/2009)) ([1605b75](https://github.com/morluto/jacobian/commit/1605b751e16d43797dc210da311c154df97ae52d))
* **math:** add integer multiplicative normal-form operations ([#2038](https://github.com/morluto/jacobian/issues/2038)) ([5184b64](https://github.com/morluto/jacobian/commit/5184b645fec978e13a3f7f096dce0a3836c68144))
* **math:** add inverse multiplicative function operations ([#1867](https://github.com/morluto/jacobian/issues/1867)) ([#2058](https://github.com/morluto/jacobian/issues/2058)) ([a30efd7](https://github.com/morluto/jacobian/commit/a30efd7c7be072c1e8db11e7666325191309fd43))
* **math:** add Latin squares domain with check and transversal search ([#1887](https://github.com/morluto/jacobian/issues/1887)) ([#2021](https://github.com/morluto/jacobian/issues/2021)) ([ff00216](https://github.com/morluto/jacobian/commit/ff00216ef0ed95f4fa5308a0db16d5fc02e23004))
* **math:** add linear code structural operations ([#1851](https://github.com/morluto/jacobian/issues/1851)) ([#2040](https://github.com/morluto/jacobian/issues/2040)) ([d5b1eaa](https://github.com/morluto/jacobian/commit/d5b1eaaae96b6452306e9f1e755522a3529d2592))
* **math:** add network optimization operations ([#1852](https://github.com/morluto/jacobian/issues/1852)) ([#2042](https://github.com/morluto/jacobian/issues/2042)) ([0aedebc](https://github.com/morluto/jacobian/commit/0aedebc8a230822f18c3f31025c751f4b9b7999b))
* **math:** add nim sum and outcome profile operations ([#2010](https://github.com/morluto/jacobian/issues/2010)) ([4690ec7](https://github.com/morluto/jacobian/commit/4690ec71844e87fdfdd4ba594e88ce1f921df2df))
* **math:** add plane algebraic curve operations ([#1877](https://github.com/morluto/jacobian/issues/1877)) ([#2060](https://github.com/morluto/jacobian/issues/2060)) ([384b491](https://github.com/morluto/jacobian/commit/384b491cb733b25803f1fc6bd1486c0a7c2d1abd))
* **math:** add polynomial interpolation operations ([#1883](https://github.com/morluto/jacobian/issues/1883)) ([#2050](https://github.com/morluto/jacobian/issues/2050)) ([e0b54ba](https://github.com/morluto/jacobian/commit/e0b54ba8ac8d8e61f81286bce20fb621181c4350))
* **math:** add polynomial vector calculus operations ([#1865](https://github.com/morluto/jacobian/issues/1865)) ([#2045](https://github.com/morluto/jacobian/issues/2045)) ([fc79fd9](https://github.com/morluto/jacobian/commit/fc79fd9f734b1ded8663502fbcfeda95af63e5ac))
* **math:** add poset closure, dual, and induced subposet operations ([#2039](https://github.com/morluto/jacobian/issues/2039)) ([55e033d](https://github.com/morluto/jacobian/commit/55e033dcb329108b7ed792f9cde99590ed3e8bf2))
* **math:** add projective coordinate operations ([#1892](https://github.com/morluto/jacobian/issues/1892)) ([#2013](https://github.com/morluto/jacobian/issues/2013)) ([43d7133](https://github.com/morluto/jacobian/commit/43d7133da2fe90b3b1eee994acdc30c6fc1b689c))
* **math:** add quadratic forms domain with evaluate, discriminant, signature ([#1841](https://github.com/morluto/jacobian/issues/1841)) ([#2027](https://github.com/morluto/jacobian/issues/2027)) ([607fe27](https://github.com/morluto/jacobian/commit/607fe27923645b71398de73a9104338511a93eb8))
* **math:** add quiver and path algebra operations ([#1863](https://github.com/morluto/jacobian/issues/1863)) ([#2048](https://github.com/morluto/jacobian/issues/2048)) ([d1645f4](https://github.com/morluto/jacobian/commit/d1645f493efafe9c81226aa462ee2fd59e6216d8))
* **math:** add root system domain with Cartan matrix root computation ([#1810](https://github.com/morluto/jacobian/issues/1810)) ([#2015](https://github.com/morluto/jacobian/issues/2015)) ([3e08ee8](https://github.com/morluto/jacobian/commit/3e08ee8ca6d72e9c38917a735f240c05233789da))
* **math:** add RSK permutation correspondence operation ([#2014](https://github.com/morluto/jacobian/issues/2014)) ([c97dd35](https://github.com/morluto/jacobian/commit/c97dd351b632ecd6f524abfcd4c82d580814150f))
* **math:** add semigroup power, idempotents, and principal ideal operations ([#2081](https://github.com/morluto/jacobian/issues/2081)) ([5a62529](https://github.com/morluto/jacobian/commit/5a62529e2e3258784c75bbd36272dd7c85e69f7a))
* **math:** add simplicial complex f-vector operation ([#1798](https://github.com/morluto/jacobian/issues/1798)) ([#2022](https://github.com/morluto/jacobian/issues/2022)) ([be4fb5c](https://github.com/morluto/jacobian/commit/be4fb5c236540a9cc7ebefd0e709314f7c3e16a3))
* **math:** add simplicial complex link and f-vector operations ([#1850](https://github.com/morluto/jacobian/issues/1850)) ([#2025](https://github.com/morluto/jacobian/issues/2025)) ([3608695](https://github.com/morluto/jacobian/commit/3608695b9574558def8ce607d78152ca9d989e8e))
* **math:** add tree decompositions domain with width, occurrences, adhesions, reroot, restrict, bag intersection ([#1731](https://github.com/morluto/jacobian/issues/1731)) ([#2032](https://github.com/morluto/jacobian/issues/2032)) ([af37526](https://github.com/morluto/jacobian/commit/af3752692cfcd51afa1ed441a173fdf4a4478b1c))
* **math:** add universal algebra domain with term evaluation, equation profiles, subalgebras, congruences, and quotients ([#1912](https://github.com/morluto/jacobian/issues/1912)) ([#2034](https://github.com/morluto/jacobian/issues/2034)) ([c9872f0](https://github.com/morluto/jacobian/commit/c9872f0ab4775cb22d1a33d764dfdb9210b5cdc9))
* **math:** extract 8 closed math PRs with review-thread root-cause fixes ([#1959](https://github.com/morluto/jacobian/issues/1959)) ([7bbb302](https://github.com/morluto/jacobian/commit/7bbb3027d5783dc213fe8f32e6457a7a73435b47))
* **mcp:** add paginated operation browse ([0bb7bfc](https://github.com/morluto/jacobian/commit/0bb7bfc96a8d4fccb1e70701e1981d0b646943eb))
* **mcp:** expose canonical polynomial input guidance ([#1662](https://github.com/morluto/jacobian/issues/1662)) ([95b283c](https://github.com/morluto/jacobian/commit/95b283c54b7ec503e75bc339a241adb45f4329a2))
* **mcp:** publish valid examples for every operation ([#1690](https://github.com/morluto/jacobian/issues/1690)) ([ae9f4e1](https://github.com/morluto/jacobian/commit/ae9f4e15937a9015d688e20c739f3e2a019654e7))
* **npm:** add agent setup wizard ([850ca04](https://github.com/morluto/jacobian/commit/850ca040c1fc7550e546f36de0a84fdda378b402))
* **number-theory:** add budgeted certified factorization ([#2099](https://github.com/morluto/jacobian/issues/2099)) ([498c5b2](https://github.com/morluto/jacobian/commit/498c5b2bbae6de60f50f6933c45d6f423cbc97a6))


### Bug Fixes

* **math:** fix sequence signs, polygon areas, finite-game Nash supports, SMT
  behavior, discrete logarithms, and request-boundary validation
  ([#1948](https://github.com/morluto/jacobian/issues/1948))
  ([c02e53e](https://github.com/morluto/jacobian/commit/c02e53ebca172146a70f7d1a7d01f2f2772eb7e5)).
* **math:** accept canonical integers beyond Python's string-conversion limit
  and fix valuation primality and discrete-logarithm edge cases
  ([#1949](https://github.com/morluto/jacobian/issues/1949))
  ([89c0a1f](https://github.com/morluto/jacobian/commit/89c0a1fb4a993f6d0730fa703d7195848303a47b)).
* **math:** reject oversized algebraic comparisons and return typed outcomes for
  singular and non-unique linear solves
  ([#2097](https://github.com/morluto/jacobian/issues/2097))
  ([cf5d325](https://github.com/morluto/jacobian/commit/cf5d3258b8a9f31cd6499539d7eff08766f50f4b)).
* **boolean:** compute Walsh spectrum from the Boolean sign function ([#2083](https://github.com/morluto/jacobian/issues/2083)) ([2056dd0](https://github.com/morluto/jacobian/commit/2056dd045eee9a9534239b9b357e3891acfcabef))
* **ci:** route Harbor job tests through bounded runner ([9f2a0c0](https://github.com/morluto/jacobian/commit/9f2a0c0bc55d5c1092bda635ab3b47b2404dec25))
* **graph6:** validate graph6 format at the request boundary ([#2087](https://github.com/morluto/jacobian/issues/2087)) ([c0e2ef1](https://github.com/morluto/jacobian/commit/c0e2ef126554cd0920d84d1be826519d8a2d1d22))
* **math:** bind distance-matrix rows to labelled vertices ([#930](https://github.com/morluto/jacobian/issues/930)) ([#2008](https://github.com/morluto/jacobian/issues/2008)) ([c254f00](https://github.com/morluto/jacobian/commit/c254f00af0a093f212a6036ac5217a15bafa36fe))
* **math:** validate exact rational growth before execution ([#2101](https://github.com/morluto/jacobian/issues/2101)) ([11b556a](https://github.com/morluto/jacobian/commit/11b556aba454846fe178a03fc47f78da18e45332))
* **matrices:** reject rectangular matrices in square-only symbolic operations ([#2085](https://github.com/morluto/jacobian/issues/2085)) ([2eb7570](https://github.com/morluto/jacobian/commit/2eb757009042e37ffd113d9389713d5c1d04cb7b))
* **matrices:** return typed outcomes for singular/inconsistent/non-unique linear systems ([#2090](https://github.com/morluto/jacobian/issues/2090)) ([6ea7d6e](https://github.com/morluto/jacobian/commit/6ea7d6e2982706bdd8a97044ad1ffef62c3d5264))
* **mcp:** advertise math.run as read-only ([225ae41](https://github.com/morluto/jacobian/commit/225ae41062e1f0dcfabce2dcadf5ee413ae57f43))
* **polynomials:** reject gcd(0,0) and handle zero operands symmetrically ([#2084](https://github.com/morluto/jacobian/issues/2084)) ([b7ce9e8](https://github.com/morluto/jacobian/commit/b7ce9e86551125d5bbd4908aa06ec0efd9155a04))
* **primorial:** use primorial-specific result type to cover full n&lt;=1000 domain ([#2086](https://github.com/morluto/jacobian/issues/2086)) ([3ba9147](https://github.com/morluto/jacobian/commit/3ba914718f03a4f557f8c32b7673e04f8cd0d2cd))
* **probability:** bound graph reliability state mass ([#1508](https://github.com/morluto/jacobian/issues/1508)) ([f22a4f9](https://github.com/morluto/jacobian/commit/f22a4f927ac275cbcc66588965b7cc1c77122839))
* rank integer factorization for prime-power queries ([#2028](https://github.com/morluto/jacobian/issues/2028)) ([b8ba303](https://github.com/morluto/jacobian/commit/b8ba30393f5410e0eced679b5e73b0c542093ce2))
* **smt:** preserve fail-closed guards under optimization ([#1506](https://github.com/morluto/jacobian/issues/1506)) ([35a42ef](https://github.com/morluto/jacobian/commit/35a42ef11ea3de952fcaecd41226ea3af80423ab))
* **symbolic:** return exact polynomial branch for unrepresentable eigenvalues ([#2091](https://github.com/morluto/jacobian/issues/2091)) ([b58126f](https://github.com/morluto/jacobian/commit/b58126faaa8ed2041d722d468fef82e24af49a2f))


### Performance Improvements

* accelerate development validation ([#1509](https://github.com/morluto/jacobian/issues/1509)) ([af10bf5](https://github.com/morluto/jacobian/commit/af10bf596976b52472103aa41899f10afbc8b7d2))


### Dependencies

* **deps-dev:** bump hypothesis from 6.165.2 to 6.165.7 ([#1945](https://github.com/morluto/jacobian/issues/1945)) ([adfbb2c](https://github.com/morluto/jacobian/commit/adfbb2c003e2f18e85ca722eae160c570deca1a0))
* **deps-dev:** bump pre-commit from 4.6.1 to 4.6.2 ([#1943](https://github.com/morluto/jacobian/issues/1943)) ([7a7068d](https://github.com/morluto/jacobian/commit/7a7068de58c3c5bbfb3a57b9d340cdf293e3f528))
* **deps-dev:** bump ruff from 0.16.1 to 0.16.3 ([#1946](https://github.com/morluto/jacobian/issues/1946)) ([27fccd5](https://github.com/morluto/jacobian/commit/27fccd56f1ac15947fa60cb6ffd368003b5e84a6))
* **deps:** bump aws-actions/configure-aws-credentials ([#1944](https://github.com/morluto/jacobian/issues/1944)) ([7d38fc6](https://github.com/morluto/jacobian/commit/7d38fc676d7b8bba252e0614ca428f164c6f503e))
* **deps:** bump https://github.com/astral-sh/ruff-pre-commit ([#1942](https://github.com/morluto/jacobian/issues/1942)) ([775a038](https://github.com/morluto/jacobian/commit/775a0387178b1d441e6f2e6921f1a8b8eda1b72a))


### Documentation

* add public operation preflight ([#1694](https://github.com/morluto/jacobian/issues/1694)) ([e96fbf8](https://github.com/morluto/jacobian/commit/e96fbf808c14fad282f3ff36c47b2145d5429af3))
* align state migration references ([a06671f](https://github.com/morluto/jacobian/commit/a06671f3c703588f4246367352abec4ce179fe31))
* clarify executable vocabulary and admission ([#2072](https://github.com/morluto/jacobian/issues/2072)) ([e3ca01d](https://github.com/morluto/jacobian/commit/e3ca01d52937b0d4a0e347866e156575203babfd))
* clarify public operation adapter semantics ([2645e7e](https://github.com/morluto/jacobian/commit/2645e7eb7d55e44bb52e118299da7324fc4fbb5c))
* define the durable operation criterion ([1c9095a](https://github.com/morluto/jacobian/commit/1c9095a3a860812d9ddcb4d02ee89191ca9c28a8))
* describe current architecture instead of deleted systems ([#2016](https://github.com/morluto/jacobian/issues/2016)) ([140c607](https://github.com/morluto/jacobian/commit/140c607ffb0f0c91f2a3b858b44d0ae9e3a2488c))
* describe the stateless mathematical kernel ([#1511](https://github.com/morluto/jacobian/issues/1511)) ([ca9e499](https://github.com/morluto/jacobian/commit/ca9e499991d18fc99968c12626c90220fd5f129b))
* distinguish operations from backend routines ([#1702](https://github.com/morluto/jacobian/issues/1702)) ([f50dbf6](https://github.com/morluto/jacobian/commit/f50dbf6277341f8837ce006d8b10eda8edb6a4b6))
* document MCP argument validation boundary ([1a912de](https://github.com/morluto/jacobian/commit/1a912de28b252343ee7173047f1fc97e0898000c))
* document operation admission ledger ([7513f68](https://github.com/morluto/jacobian/commit/7513f689c47e4456ae3c8382025a73e004cb841e))
* explain Jacobian's executable math vocabulary hypothesis ([#2012](https://github.com/morluto/jacobian/issues/2012)) ([efce533](https://github.com/morluto/jacobian/commit/efce533714a7745c195e655fbb815359f9720304))
* explain semantic atomicity and vocabulary growth ([#2074](https://github.com/morluto/jacobian/issues/2074)) ([14f884d](https://github.com/morluto/jacobian/commit/14f884dca74fce927a01c6a71f6b059ae55c4b6f))
* **math:** require adapter boundary regressions ([f083434](https://github.com/morluto/jacobian/commit/f083434723358357319603158550cbd3becb24a7))
* **mcp:** clarify proactive math tool guidance ([71f17e6](https://github.com/morluto/jacobian/commit/71f17e63f84637ee9574d16dc99b45c1d78e9586))
* **mcp:** recommend local math operations ([6b6b234](https://github.com/morluto/jacobian/commit/6b6b234c02172dfbce9358d54e7a0d6bce1e1502))
* **readme:** add Chinese README ([0ed7a8c](https://github.com/morluto/jacobian/commit/0ed7a8c99bc299d7f05313a42d6f8854dcf41fa4))
* **readme:** add Chinese translation link ([80d1031](https://github.com/morluto/jacobian/commit/80d1031a75d78a4e56bfde7e3823a5d71c495eeb))
* **readme:** refresh product description and version ([3a5acc7](https://github.com/morluto/jacobian/commit/3a5acc75a62cc20209ed1bae667f852fab62ad19))
* require closure matrix when closing a broad operation parent ([9bda56a](https://github.com/morluto/jacobian/commit/9bda56a4e1ab6aa315c87a673dfa315d7c5e951e))
* separate benchmark material from product docs ([#2019](https://github.com/morluto/jacobian/issues/2019)) ([7ab1d47](https://github.com/morluto/jacobian/commit/7ab1d4746a94e233c97b1fcc175564c871dcc850))
* **skills:** add Jacobian math workflow ([134d98b](https://github.com/morluto/jacobian/commit/134d98b4fe5ac4cba5994816232f4713d2eac607))
* **skills:** clarify operation discovery modes ([cffda53](https://github.com/morluto/jacobian/commit/cffda53b6dfe03365451c055c56797859a468bab))
* split issue forms into gap diagnosis and admitted operation ([bd50e85](https://github.com/morluto/jacobian/commit/bd50e858319c9cc035719bc8f028e4bb818699d9))

## [0.12.0](https://github.com/morluto/jacobian/compare/jacobian-v0.11.0...jacobian-v0.12.0) (2026-08-13)


### Features

* add actionable checker diagnostics ([84d2f86](https://github.com/morluto/jacobian/commit/84d2f86fa18cf3d2e6aff626b85b636848669343))
* add exact finite polynomial map values ([f3d7a7f](https://github.com/morluto/jacobian/commit/f3d7a7f2c407fdba0addebfbdf3a9e4ed27eb872))
* add exact finite-field semantic values ([df8ad85](https://github.com/morluto/jacobian/commit/df8ad85910cdfc2573f4245e0709561e8fc22559))
* **arithmetic:** add verified real-quadratic order ([#1287](https://github.com/morluto/jacobian/issues/1287)) ([6ad9f41](https://github.com/morluto/jacobian/commit/6ad9f41dcdeec87868e65e576fdbb8b6d3a48dcd))
* bind complete projective lines as values ([b6e69d2](https://github.com/morluto/jacobian/commit/b6e69d2804487b7999306abc6376c5b994fa4119))
* bind finite-field operations with provisional ports ([a980d11](https://github.com/morluto/jacobian/commit/a980d11cd8d9817d808a965ccd25ddcddf2274b9))
* **checkers:** independently replay bounded LCM ([#1280](https://github.com/morluto/jacobian/issues/1280)) ([bbd8467](https://github.com/morluto/jacobian/commit/bbd8467640bc469af32f3e5e24d0040c0c2311d2))
* **checkers:** verify rational LP optimum certificates ([#1279](https://github.com/morluto/jacobian/issues/1279)) ([2aca3aa](https://github.com/morluto/jacobian/commit/2aca3aa8e1bb5f77cc31fc0bdb77894d7bbd3ffd))
* **combinatorics:** verify submitted P-recursive tables ([#1285](https://github.com/morluto/jacobian/issues/1285)) ([edd886e](https://github.com/morluto/jacobian/commit/edd886e58473a252727c0b70bc5b1868e5db9d91))
* compose finite polynomial map operations ([7b27725](https://github.com/morluto/jacobian/commit/7b27725c0e1a8b9f34028361087f661ab2dd5eef))
* compose the complete finite-field rank ledger ([403cf4a](https://github.com/morluto/jacobian/commit/403cf4a4f771ee055f610c4f103136d9c9f45b72))
* compose the finite-field direction-rank slice ([a573cfc](https://github.com/morluto/jacobian/commit/a573cfc1b741d6212e3e9b537e420495cdaca35e))
* compose typed values by opaque reference ([f1fb03a](https://github.com/morluto/jacobian/commit/f1fb03ae86f85c818a5e88c859852ad522276d8d))
* **composition:** pass typed results to exact checkers ([#1298](https://github.com/morluto/jacobian/issues/1298)) ([abcd9a8](https://github.com/morluto/jacobian/commit/abcd9a8449beb91b7fe46d43045e54ec7800743c))
* **geometry:** expose bounded projective flat previews ([#1294](https://github.com/morluto/jacobian/issues/1294)) ([7182818](https://github.com/morluto/jacobian/commit/7182818787b6432e43f47c1b1a4209120d6f1c04))
* **graphs:** add canonical graph6 decoding and verification ([#1284](https://github.com/morluto/jacobian/issues/1284)) ([d178bd4](https://github.com/morluto/jacobian/commit/d178bd4518975c88d9c462f3de5e083a776ece3a))
* **graphs:** extend bounded independence number to order 128 ([#1297](https://github.com/morluto/jacobian/issues/1297)) ([cf146a0](https://github.com/morluto/jacobian/commit/cf146a0514b63100e01d8665a677d88261146a61))
* independently replay finite polynomial maps ([2afc7d7](https://github.com/morluto/jacobian/commit/2afc7d7cf51c2a24e3ec3307e9c26a431d03cb23))
* independently verify finite-field map rank ([8197270](https://github.com/morluto/jacobian/commit/81972706c104e7ad21f5bc09407d3e93097525d7))
* independently verify finite-field restriction ([4aac03a](https://github.com/morluto/jacobian/commit/4aac03af387f859382002549cad958ed43b431c3))
* **matrix:** extend exact determinant to order 64 ([#1302](https://github.com/morluto/jacobian/issues/1302)) ([83ef960](https://github.com/morluto/jacobian/commit/83ef960577b29287d94f92c31c57a2d7fd3e3f2c))
* **number-theory:** add modular polynomial identity verification ([#1286](https://github.com/morluto/jacobian/issues/1286)) ([ac740ca](https://github.com/morluto/jacobian/commit/ac740ca07418cc3ec9651144e6cb0498f2fda2bc))
* **number-theory:** add verified finite abelian factorizations ([#1301](https://github.com/morluto/jacobian/issues/1301)) ([7a2074e](https://github.com/morluto/jacobian/commit/7a2074ed919d1c8e847f8f30ec0be1ddf39f9642))
* **probability:** add verified finite-table mutual information ([#1300](https://github.com/morluto/jacobian/issues/1300)) ([8809168](https://github.com/morluto/jacobian/commit/8809168b1e52083a6bd292c53eea306c32cd227f))
* tighten domain diagnostics and Gaussian inputs ([6c269fe](https://github.com/morluto/jacobian/commit/6c269fee7c2550d07ac5ae7bab5122dac55e58d9))


### Bug Fixes

* align operation effects and bounded timeout results ([3982826](https://github.com/morluto/jacobian/commit/3982826ce901a36ffb347c759c4170011eeb8117))
* authorize immutable Mathlib package checkouts ([85352b3](https://github.com/morluto/jacobian/commit/85352b3db6d13db452afcc3eff3d37d8f867ffdc))
* **benchmarks:** accept reordered cubic residue coverage ([#1422](https://github.com/morluto/jacobian/issues/1422)) ([4bf70a3](https://github.com/morluto/jacobian/commit/4bf70a3121e5ce27f3fef929dba408833639cc27))
* **benchmarks:** accept reordered zero residual indices ([#1425](https://github.com/morluto/jacobian/issues/1425)) ([4872ed7](https://github.com/morluto/jacobian/commit/4872ed79c932a2ef3af3b2187a91e140a0bee0c9))
* **benchmarks:** publish Farkas scalar replay contract ([#1309](https://github.com/morluto/jacobian/issues/1309)) ([bfacfb5](https://github.com/morluto/jacobian/commit/bfacfb5c5d7afb0ef0839381f12c4ec24eee2871))
* bind diagnostic evidence conditions ([fd337e2](https://github.com/morluto/jacobian/commit/fd337e210bba125599ee1c6027e9fae1dbb42785))
* bind Lean recovery claims ([309885d](https://github.com/morluto/jacobian/commit/309885df5edd90114a92663267f55235f0a2b49e))
* bind rank results to the direction field ([2b62f9d](https://github.com/morluto/jacobian/commit/2b62f9dfa3caa04e2de0335a9c7c8817a83e3c63))
* bind verification records to projected identities ([ea054ed](https://github.com/morluto/jacobian/commit/ea054ed5cebb554fecef98e363b454971b001ba3))
* bound finite-field validation before computation ([3eb1164](https://github.com/morluto/jacobian/commit/3eb11643870788a31d7382e9ab1e5c54205495a8))
* build the complete Lean release runtime ([909d341](https://github.com/morluto/jacobian/commit/909d34112b3903957c794f93f5bbbe155097b4f2))
* **ci:** preserve required PR evidence ([#1426](https://github.com/morluto/jacobian/issues/1426)) ([364b075](https://github.com/morluto/jacobian/commit/364b0752f4422791cd35408e96dff0e6ebdbd130))
* **ci:** sync packaged backends in container ([5d73334](https://github.com/morluto/jacobian/commit/5d73334bb10c80ff681fc8472b096dad5a86a7b8))
* classify verification from record identity ([07164ef](https://github.com/morluto/jacobian/commit/07164ef6b85c847dbde742b0b3f330690f7d46a3))
* confine benchmark process metadata ([b68dbd0](https://github.com/morluto/jacobian/commit/b68dbd0471c849c9546f8bd81cbc37677b2599e0))
* correct inline verifier lineage guidance ([#1234](https://github.com/morluto/jacobian/issues/1234)) ([16a2da6](https://github.com/morluto/jacobian/commit/16a2da6b64d73c544b2ee60e8cec60ca8d6d24c8))
* **discovery:** expose P-recursive sequence aliases ([#934](https://github.com/morluto/jacobian/issues/934)) ([3ff985b](https://github.com/morluto/jacobian/commit/3ff985bc1a2d93e76e9e0f9d488841d0b98c9ce7))
* **discovery:** expose polynomial expansion normalization ([#1273](https://github.com/morluto/jacobian/issues/1273)) ([8e585f0](https://github.com/morluto/jacobian/commit/8e585f09a814a3533d5f1154a452e8907a863849))
* **discovery:** surface finite expectations ([#1247](https://github.com/morluto/jacobian/issues/1247)) ([2b13f16](https://github.com/morluto/jacobian/commit/2b13f16dd16fa9f561d26faf7906f1e5a2a8f6ca))
* **discovery:** surface polynomial bound checks ([#1250](https://github.com/morluto/jacobian/issues/1250)) ([2e579f0](https://github.com/morluto/jacobian/commit/2e579f0c6a142feefa2398dfcb3144bfaa019688))
* **discovery:** surface rational polynomial resultants ([#1244](https://github.com/morluto/jacobian/issues/1244)) ([6e87091](https://github.com/morluto/jacobian/commit/6e87091eea58a5f26d4931f23868a0a384d412cf))
* distinguish invalid operation results ([d8237b7](https://github.com/morluto/jacobian/commit/d8237b7ccfb1ea6e7d2a03dd2222b292c6a1a398))
* **eval:** preserve incomplete observation evidence ([#1312](https://github.com/morluto/jacobian/issues/1312)) ([84f95c2](https://github.com/morluto/jacobian/commit/84f95c22f5446ef91f693737fe072a238d1d9715))
* **evals:** read top-level verification records ([a7a272e](https://github.com/morluto/jacobian/commit/a7a272e0fa8171648045cae383de46e320b7e119))
* **examples:** align cold-worker budgets and Arb discovery ([#1283](https://github.com/morluto/jacobian/issues/1283)) ([50c3fce](https://github.com/morluto/jacobian/commit/50c3fceb81c1696d2b4f50eb6efa4816d0ae0e50))
* **finite-fields:** bound producer work before execution ([8f9132f](https://github.com/morluto/jacobian/commit/8f9132f23f7680a5a638a7be65f25baf3adf929c))
* **finite-fields:** reject cross-field rank requests ([a6f1630](https://github.com/morluto/jacobian/commit/a6f1630b899cf72e27808d8650505463d9225d40))
* **finite-fields:** replay ledgers before orbit aggregation ([5950271](https://github.com/morluto/jacobian/commit/5950271c41aa5eb8c849dd87be9d4da35e916ce0))
* **finite-fields:** validate polynomial map tables ([99c2921](https://github.com/morluto/jacobian/commit/99c29212c82cf6ce114596a8cf6d5952e9bbe553))
* **graph:** raise distance matrix order bound ([#1369](https://github.com/morluto/jacobian/issues/1369)) ([bf8b380](https://github.com/morluto/jacobian/commit/bf8b3805c304c550b58ae0eb3d4b5654c54aea51))
* handle one-row lattice reduction ([#1203](https://github.com/morluto/jacobian/issues/1203)) ([b5826b6](https://github.com/morluto/jacobian/commit/b5826b689343857143d50e223f6bf43c6a2f2859))
* **harbor:** launch the remote MCP observation host ([#1296](https://github.com/morluto/jacobian/issues/1296)) ([5ac9a8c](https://github.com/morluto/jacobian/commit/5ac9a8ce56ebf14d09ee79bcef2473b25b09cf3c))
* hide internal Lean scaffold warnings ([d9c43bf](https://github.com/morluto/jacobian/commit/d9c43bf35bc34480be737608204d0f80850ecd82))
* keep Lean recovery evidence retryable ([a89c296](https://github.com/morluto/jacobian/commit/a89c2968dec8d8a166be20ab76134aad85d21316))
* **lean:** remove unchanged corrupt declaration caches ([#1295](https://github.com/morluto/jacobian/issues/1295)) ([da49edc](https://github.com/morluto/jacobian/commit/da49edc2ba6cbd7b8fb1c153b7e3affaf68289cc))
* make Lean deployments portable and service-readable ([a1f7373](https://github.com/morluto/jacobian/commit/a1f737384e1fc05d00001caa88487b9638bd186a))
* match the pinned REPL pickle response ([58c5deb](https://github.com/morluto/jacobian/commit/58c5debf65a91176d008e5f875a1e1470282a939))
* **matrix:** publish the certified Smith input bound ([#1271](https://github.com/morluto/jacobian/issues/1271)) ([236df59](https://github.com/morluto/jacobian/commit/236df591495ad522a00e9243da4f36b78aadcfda))
* **mcp:** consolidate local routing and status-first guidance ([#1269](https://github.com/morluto/jacobian/issues/1269)) ([9fb1412](https://github.com/morluto/jacobian/commit/9fb1412d4a36f1bac8423710d01cbefb34615e7c))
* **polynomials:** preserve discriminant domains ([ea5d8f7](https://github.com/morluto/jacobian/commit/ea5d8f75fed0da362cac68f0edcf3931aef21eaa))
* **polynomials:** preserve monic factor content ([9618959](https://github.com/morluto/jacobian/commit/9618959833a19c73ad5ff2498fafdb28778a3185))
* **polynomials:** reject non-rational Groebner inputs ([82ac501](https://github.com/morluto/jacobian/commit/82ac501355901c1c62a402494c2de8c96b8eeea6))
* preserve named Lean checker diagnostics ([28bccf0](https://github.com/morluto/jacobian/commit/28bccf0bcaccc92529b252c82dd1891ebec5d38e))
* preserve portable Lean runtime paths ([2f39844](https://github.com/morluto/jacobian/commit/2f39844b215af4057e7ce2241c146062b5c8e8c7))
* probe Lean before Mathlib compilation ([ae27ef1](https://github.com/morluto/jacobian/commit/ae27ef1ee3a1b0433cb663ae7dc747210bfc7f7d))
* **providers:** validate the complete backend stack ([ece7469](https://github.com/morluto/jacobian/commit/ece746944c7184140ed3f9da9bf82b88a4a96030))
* reclaim bounded value references by recency ([551899c](https://github.com/morluto/jacobian/commit/551899c69f3f0012a9bff775d3573ff1e5ce2592))
* **release:** synchronize npm lockfile to 0.11.0 ([bd272f9](https://github.com/morluto/jacobian/commit/bd272f9e12f7e7cfc6c34e48945dee8e3b307c71))
* reuse the pinned Lean toolchain on redeploy ([d67597c](https://github.com/morluto/jacobian/commit/d67597cd824447cb09092d943db13928f5c6e321))
* run Lean frontends from immutable releases ([d5a0a26](https://github.com/morluto/jacobian/commit/d5a0a26dd8c150480983bd823e4930277e6d22d0))
* separate Lean setup diagnostics ([f4866a9](https://github.com/morluto/jacobian/commit/f4866a9672241158686ccd6ff70bb3150f523185))
* **skills:** align math.run payload envelope ([#1206](https://github.com/morluto/jacobian/issues/1206)) ([5265758](https://github.com/morluto/jacobian/commit/526575833ef9e849aa5366a55ac5d375d65d36fb))
* **tests:** add __init__.py to directories with duplicate test basenames ([0ecd977](https://github.com/morluto/jacobian/commit/0ecd977b2f78b0e570a86b16351e9916d877591a))
* update operational clients for simplified discovery ([84f51a6](https://github.com/morluto/jacobian/commit/84f51a627e4f9f57189a9f32010729d30b2da110))
* validate root-owned Lean package checkouts ([7c02433](https://github.com/morluto/jacobian/commit/7c024339d60bb644c6c7f69b3712175597dc4645))
* **validation:** bound public rejected-input diagnostics ([#1242](https://github.com/morluto/jacobian/issues/1242)) ([9482a9d](https://github.com/morluto/jacobian/commit/9482a9dd3fc7cc1077b051520d180d1d0627684b))
* **verification:** bind inline records to exact values ([3f282e9](https://github.com/morluto/jacobian/commit/3f282e9927299aff1cde3c05464d85e04a14405a))


### Performance Improvements

* **lean:** cache pinned declaration metadata ([#1272](https://github.com/morluto/jacobian/issues/1272)) ([1360962](https://github.com/morluto/jacobian/commit/1360962d8a35a6d504a1d34b7ccac90477501cc3))


### Dependencies

* bump release-please action to 5.0.0 ([#1002](https://github.com/morluto/jacobian/issues/1002)) ([fd089d7](https://github.com/morluto/jacobian/commit/fd089d7962d6881f9c3860b7b6db5e02cb89c5e7))
* **deps:** bump typer from 0.27.0 to 0.27.1 ([#1003](https://github.com/morluto/jacobian/issues/1003)) ([f55eddd](https://github.com/morluto/jacobian/commit/f55eddd91069e5dcb0abb861e74b086a087b479d))


### Documentation

* add architecture budgets to pull requests ([5a62920](https://github.com/morluto/jacobian/commit/5a62920b8adbe8763bfc075fa225a64953e63ee1))
* clarify mathematical validation diagnostics ([5e45a93](https://github.com/morluto/jacobian/commit/5e45a933b24d128a9d7b4ba70f16141cd5239d17))
* clarify mathematical value layering ([f491a3f](https://github.com/morluto/jacobian/commit/f491a3f5a1be58c9ebbc523688e7328369d06a0b))
* define request-local reference lifetime ([14f49da](https://github.com/morluto/jacobian/commit/14f49da5d2291aa1b19cf66f8042ec126af022da))
* define the atomic mathematics product ([272744d](https://github.com/morluto/jacobian/commit/272744d02600fac94b73b6863d74bfd3a691590a))
* describe verification by record identity ([c114896](https://github.com/morluto/jacobian/commit/c114896dc75b08773c8500269b50f89e3f33785c))
* document the finite-field native API ([5f340b8](https://github.com/morluto/jacobian/commit/5f340b861afe526e10c7b43fd1a5b94623c99082))
* freeze the minimal composition port contract ([9a7bd57](https://github.com/morluto/jacobian/commit/9a7bd570f5c990c12b8a9d9d76f47903bd8b606b))
* **math:** clarify semantic backend boundaries ([9e93c8a](https://github.com/morluto/jacobian/commit/9e93c8ae096f3518a97735eb9e2f13b8fcc8b546))
* narrow discovery to implemented applicability ([53d5007](https://github.com/morluto/jacobian/commit/53d5007eda940a12b89f0c830d70278c835507bd))
* **providers:** define mandatory but lazy backends ([e571737](https://github.com/morluto/jacobian/commit/e571737601e86df7c92b29aba8c10f40f317cfd0))
* **topology:** add runnable invocation examples ([#1251](https://github.com/morluto/jacobian/issues/1251)) ([2b1c961](https://github.com/morluto/jacobian/commit/2b1c961ff4bbc27cd28401e19fcd7b8fbb8173d6))

## [0.11.0](https://github.com/morluto/jacobian/compare/jacobian-v0.10.0...jacobian-v0.11.0) (2026-08-10)


### Features

* verify polynomial-coefficient recurrences ([#902](https://github.com/morluto/jacobian/issues/902)) ([b9b01b3](https://github.com/morluto/jacobian/commit/b9b01b358dfa4743b673ded9d9694f069e9f8058))


### Bug Fixes

* **benchmarks:** harden Jacobian evaluation evidence ([#946](https://github.com/morluto/jacobian/issues/946)) ([0052a5b](https://github.com/morluto/jacobian/commit/0052a5bf78f63f5539be13da6493abb395c5026d))
* **ci:** validate release branch updates ([1a5048c](https://github.com/morluto/jacobian/commit/1a5048c3d13cffbd8a9debe947807aa1c1815b08))
* **cli:** repair onboarding setup flow ([11b4b28](https://github.com/morluto/jacobian/commit/11b4b28365f47afe19f38f43f30721a6497fc246))
* honor selected npm gates and core setup profile ([ee8ef12](https://github.com/morluto/jacobian/commit/ee8ef12103bb811a18ee6af20d3289e9b13aaa3b))
* **mcp:** publish object-rooted math.find schema ([#898](https://github.com/morluto/jacobian/issues/898)) ([332a5e4](https://github.com/morluto/jacobian/commit/332a5e4af884fb9df953d538c90890ee17e6834b))


### Dependencies

* **deps-dev:** bump hypothesis from 6.164.0 to 6.165.2 ([#1005](https://github.com/morluto/jacobian/issues/1005)) ([8818f9e](https://github.com/morluto/jacobian/commit/8818f9e98bb94fe6714f884d47e1420bf72de7e7))
* **deps:** bump actions/cache/restore from 4.3.0 to 6.1.0 ([#1004](https://github.com/morluto/jacobian/issues/1004)) ([50d663f](https://github.com/morluto/jacobian/commit/50d663f65b45266fecadd2fe5421b662874fd25b))
* **deps:** bump pypa/gh-action-pypi-publish from 1.14.1 to 1.14.2 ([#1006](https://github.com/morluto/jacobian/issues/1006)) ([d3ed473](https://github.com/morluto/jacobian/commit/d3ed47327153a1b1bad069c8d1c301ae0d11ac89))


### Documentation

* lead quickstart with one-time npx setup ([344b7b6](https://github.com/morluto/jacobian/commit/344b7b62e08a13510fdb4b13f88a789521146802))
* reframe product as search/execute math toolbox ([#1147](https://github.com/morluto/jacobian/issues/1147)) ([e443138](https://github.com/morluto/jacobian/commit/e443138da8ba56910719b7ea94414d32e4dfd733))
* reserve VERIFIED assurance in control evaluations ([#906](https://github.com/morluto/jacobian/issues/906)) ([d600db0](https://github.com/morluto/jacobian/commit/d600db076d2d87ad2697b81d9d6b2363cd1db82f))
* **skill:** tighten Harbor evaluation boundaries ([3b37422](https://github.com/morluto/jacobian/commit/3b374229ed27ace3f0e857bec5cf540b936a041e))

## [0.10.0](https://github.com/morluto/jacobian/compare/jacobian-v0.9.0...jacobian-v0.10.0) (2026-08-08)


### Features

* **benchmarks:** add fail-closed aggregate reward policy and ratchets ([70963ed](https://github.com/morluto/jacobian/commit/70963ed2ceb104475a39208abad6da0b3bdec2b5))
* **benchmarks:** add selected-task workflow ([#572](https://github.com/morluto/jacobian/issues/572)) ([9d94310](https://github.com/morluto/jacobian/commit/9d943108c6650200c67eb5aa70fb27b78b048d52))
* **benchmarks:** move verifier exceptions into task-local contracts ([ccf4b7c](https://github.com/morluto/jacobian/commit/ccf4b7c898387583ebbbf1e26d86f6b0ce862a0c))


### Bug Fixes

* address post-merge MCP and observation review findings ([7f5a035](https://github.com/morluto/jacobian/commit/7f5a035c369f5b6445c5236a384b3adaa8852206))
* address post-merge review findings ([9913af4](https://github.com/morluto/jacobian/commit/9913af4ea0fec4c686edd940bb673f3a4688f89a))
* **benchmarks:** accept compact network_mode TOML assignments ([#668](https://github.com/morluto/jacobian/issues/668)) ([5957cd2](https://github.com/morluto/jacobian/commit/5957cd24b5acb207d99e0ee0cd98c5714fd6a468))
* **benchmarks:** address reconstruction deck certificate review comments ([#657](https://github.com/morluto/jacobian/issues/657)) ([7154612](https://github.com/morluto/jacobian/commit/71546120bc94035a20822a5ef26a3a9d1b6a12d5))
* **benchmarks:** bind cddlib/regina and tighten provider Oracle evidence ([#709](https://github.com/morluto/jacobian/issues/709)) ([bf44e7a](https://github.com/morluto/jacobian/commit/bf44e7ab470d739274f57bcb78c8017d03288247))
* **benchmarks:** discard consumed streaming whitespace prefix in reconstruction deck verifier ([#682](https://github.com/morluto/jacobian/issues/682)) ([256ff13](https://github.com/morluto/jacobian/commit/256ff1318e024e9bcf28f0544ed4b77b0b35b263))
* **benchmarks:** discover conjecture host validations ([#829](https://github.com/morluto/jacobian/issues/829)) ([08f7d95](https://github.com/morluto/jacobian/commit/08f7d951cbc2c2ee63824bf9d029cbbd33be3616))
* **benchmarks:** fail-closed reward, support, and provider evidence ([f03baaf](https://github.com/morluto/jacobian/commit/f03baafb40e22cb9a37392053e4a5d52a1d97b69))
* **benchmarks:** identify provider feasibility tasks accurately ([#771](https://github.com/morluto/jacobian/issues/771)) ([4de8452](https://github.com/morluto/jacobian/commit/4de8452081397cb056be0a98bf320ca032204d6a))
* **benchmarks:** install jsonschema in upgraded verifier images ([64132a9](https://github.com/morluto/jacobian/commit/64132a94d689ade8e9fbd447101df3cafe5d35cf))
* **benchmarks:** land Hadwiger verifier review follow-ups (evidence schema, JSON type fidelity, assurance guard, non-finite JSON) ([#675](https://github.com/morluto/jacobian/issues/675)) ([c174872](https://github.com/morluto/jacobian/commit/c1748722017a66295a317188885acace6d753faa))
* **benchmarks:** make mathematical rewards fail closed ([b3dbc1c](https://github.com/morluto/jacobian/commit/b3dbc1c6d00338a5af980253c04abc0880829b7e))
* **benchmarks:** migrate legacy verifier diagnostics ([ee14bf7](https://github.com/morluto/jacobian/commit/ee14bf7470d28f2c59e319e8159d3c75c3c942a2))
* **benchmarks:** pin deterministic pytest-randomly seed for host shards ([#625](https://github.com/morluto/jacobian/issues/625)) ([6c893d3](https://github.com/morluto/jacobian/commit/6c893d3d9e003585c1eeac1eb394566d35924dfb))
* **benchmarks:** preserve decoupled input diagnostics ([9b05caf](https://github.com/morluto/jacobian/commit/9b05caf5e94cbacb8f9100c3859af783d8284138))
* **benchmarks:** preserve fail-closed diagnostics and validation gates ([e2c264b](https://github.com/morluto/jacobian/commit/e2c264bd0145ab3314a69fce9df961bdda946b89))
* **benchmarks:** publish bound input for provider Oracle verifiers ([#683](https://github.com/morluto/jacobian/issues/683)) ([41504be](https://github.com/morluto/jacobian/commit/41504be7ac07cb6d9983a5ba81750c1c851b316c))
* **benchmarks:** repair shared CI root causes for oracle and contracts ([#640](https://github.com/morluto/jacobian/issues/640)) ([95b2024](https://github.com/morluto/jacobian/commit/95b202450872a59f7aa944dc17f03e6e699bf0b5))
* **benchmarks:** require authoritative completion counts ([556dfae](https://github.com/morluto/jacobian/commit/556dfae0e38c9e7b6d1b07c4c7cd5aea7fe628ca))
* **benchmarks:** skip vanished Oracle evidence candidates ([#743](https://github.com/morluto/jacobian/issues/743)) ([a3b35ac](https://github.com/morluto/jacobian/commit/a3b35ac6f1b230e3a20da1044086353e44f860b0))
* **benchmarks:** sort functools import for schema validator cache CI ([#621](https://github.com/morluto/jacobian/issues/621)) ([db4cc38](https://github.com/morluto/jacobian/commit/db4cc3834dadb6b956048c6edfb8e9daddfda418))
* **capabilities:** return typed failures for invalid adapter output ([#739](https://github.com/morluto/jacobian/issues/739)) ([166a380](https://github.com/morluto/jacobian/commit/166a380f956c56842ade0a78a988a4030bfe073c))
* **checkers:** bind the Lake launcher digest to the authorized Lean runtime ([b84a9ca](https://github.com/morluto/jacobian/commit/b84a9ca819ff298eabaaa3ee482a3e2a66a5adb9))
* **ci:** account for shared Lean process memory ([6114242](https://github.com/morluto/jacobian/commit/6114242ad9f36c9561b9bcbbfcc95be93b55e1ff))
* **ci:** avoid repeated trajectory cluster validation ([a7c312b](https://github.com/morluto/jacobian/commit/a7c312bc252ca822d0e17cdcf5a884b058910150))
* **ci:** format eval telemetry test signature ([#777](https://github.com/morluto/jacobian/issues/777)) ([bad5a78](https://github.com/morluto/jacobian/commit/bad5a78d7aac6cdec801f57537898bdc4c0e5626))
* **ci:** restore lint-full and test-architecture on main ([0891065](https://github.com/morluto/jacobian/commit/089106563c8947ec8794da2065f385f64b24e425))
* **ci:** restore lint-full and test-architecture on main ([f39f3eb](https://github.com/morluto/jacobian/commit/f39f3eb1ecec7cf525a1f0b7068e46784c748472))
* **ci:** restore static gates after benchmark additions ([#723](https://github.com/morluto/jacobian/issues/723)) ([b5fc67a](https://github.com/morluto/jacobian/commit/b5fc67a7c934c6f9eb286d31656ab879bc70aedd))
* **ci:** tolerate duplicate timing entries across benchmark shards ([#624](https://github.com/morluto/jacobian/issues/624)) ([6b3b58a](https://github.com/morluto/jacobian/commit/6b3b58afd2c2140dd036b63e768dd1e15afa232f))
* **cli:** preserve command failures during cleanup ([#713](https://github.com/morluto/jacobian/issues/713)) ([fb118bb](https://github.com/morluto/jacobian/commit/fb118bb118ff2370430845cd6ac250d979174093))
* **eval:** guard resource-read tool field against non-hashable values ([2e08f67](https://github.com/morluto/jacobian/commit/2e08f67de52cb80089f005becd5c634dbee87bd2))
* **eval:** ignore malformed telemetry status and tool fields ([#708](https://github.com/morluto/jacobian/issues/708)) ([f64d891](https://github.com/morluto/jacobian/commit/f64d891b0a855c4af77c91376e5925889ada9a60))
* **experiments:** use first-page parent capacity ([#716](https://github.com/morluto/jacobian/issues/716)) ([10421cb](https://github.com/morluto/jacobian/commit/10421cb6839e24867f025d88d9ab799b850121d2))
* fail closed on malformed JSON and purge stale plugin imports ([4cdb2b4](https://github.com/morluto/jacobian/commit/4cdb2b436c6314c35bd6e90d2ef5766981e5b2e7))
* fail closed on malformed observation/control JSON and purge stale plugin imports ([3bfaf5f](https://github.com/morluto/jacobian/commit/3bfaf5f364b67907a415c5346da87a327b28279b))
* fail-closed evaluation integrity across verifiers and kernel ([#585](https://github.com/morluto/jacobian/issues/585)) ([5aec651](https://github.com/morluto/jacobian/commit/5aec6514f06f47a0745c2c3ff4db0deb7a11c4e0))
* **lean:** resolve toolchain lean instead of elan proxy ([#627](https://github.com/morluto/jacobian/issues/627)) ([c91140e](https://github.com/morluto/jacobian/commit/c91140ee365252e7b86844c71001a3e0023cdb02))
* **matrix:** reject invalid flint worker protocols ([#725](https://github.com/morluto/jacobian/issues/725)) ([f2d3525](https://github.com/morluto/jacobian/commit/f2d3525ca7452506511ce66066015475ebf7b654))
* MCP concurrency, fail-closed benchmark, and resource safety fixes ([ed07ff7](https://github.com/morluto/jacobian/commit/ed07ff780e453c953e35ed5c6d3b5c991be55200))
* MCP concurrency, fail-closed benchmark, and resource safety fixes ([6dab752](https://github.com/morluto/jacobian/commit/6dab75299303ad374826192cc3a860049082901b))
* **mcp:** bound cancellation drain with monotonic deadline ([fec2f27](https://github.com/morluto/jacobian/commit/fec2f27e93f925d7d9302401cbb90d5b9a7dbd74))
* **mcp:** clarify validation and assurance diagnostics ([#602](https://github.com/morluto/jacobian/issues/602)) ([0733334](https://github.com/morluto/jacobian/commit/07333343e70d6e09534db92c4dfeff1c96b6def3))
* **mcp:** distinguish unknown discovery domains ([#604](https://github.com/morluto/jacobian/issues/604)) ([fcfe72f](https://github.com/morluto/jacobian/commit/fcfe72f0169697ddd2e37d3af96358df21536246))
* **mcp:** quarantine failed tenant runtimes and serialize shutdown ([#745](https://github.com/morluto/jacobian/issues/745)) ([ce452d9](https://github.com/morluto/jacobian/commit/ce452d96d1aae6c918ea62fe32243cd2b51a1ec0))
* **mcp:** report nested required constraints, keep operation selection agent-owned ([#678](https://github.com/morluto/jacobian/issues/678)) ([25141d0](https://github.com/morluto/jacobian/commit/25141d0e293ea0d8023273cea3ad20049ef47340))
* **npm:** resolve latest bootstrap for upgrades ([fb884b3](https://github.com/morluto/jacobian/commit/fb884b3a97a1f317e60b09034af9ce598ca94e37))
* **polynomials:** bound inverse solver termination ([#744](https://github.com/morluto/jacobian/issues/744)) ([024d9c7](https://github.com/morluto/jacobian/commit/024d9c715b328680b184de973e958807cd3212c6))
* **pre-commit:** remove stale verifier-support hook ([#730](https://github.com/morluto/jacobian/issues/730)) ([383d17e](https://github.com/morluto/jacobian/commit/383d17ee08cfc8cc5ab3734eb5296ff830f22987))
* **providers:** correct Z3 mismatch diagnostic ([#748](https://github.com/morluto/jacobian/issues/748)) ([0f7a445](https://github.com/morluto/jacobian/commit/0f7a445910852accd0f92cc9f4368bc9dd6c92d8))
* **runtime:** attempt all owners during shutdown ([#714](https://github.com/morluto/jacobian/issues/714)) ([b423a41](https://github.com/morluto/jacobian/commit/b423a41d29669ecc0adc0308359d47ff1e67a04b))
* **runtime:** preserve bootstrap failures during cleanup ([#724](https://github.com/morluto/jacobian/issues/724)) ([42fc483](https://github.com/morluto/jacobian/commit/42fc483b82d045864f605693ce0a4a74e10aa08d))
* **search:** bound nomination lineage before persistence ([#750](https://github.com/morluto/jacobian/issues/750)) ([f6043fe](https://github.com/morluto/jacobian/commit/f6043fe3fff61d8a0c4a3d068caefbbb0ca4610e))
* **smt:** allow declare-const in cvc5 worker ([#715](https://github.com/morluto/jacobian/issues/715)) ([92ca13f](https://github.com/morluto/jacobian/commit/92ca13f40bbe303e71503c0eacb21d38cceca10e))
* **smt:** reject non-string cvc5 worker statuses ([#726](https://github.com/morluto/jacobian/issues/726)) ([43ceea8](https://github.com/morluto/jacobian/commit/43ceea8587f46983944b90695ff53cb47005803b))
* **tests:** audit and clean up false greens, assertion debt, and antipatterns ([#628](https://github.com/morluto/jacobian/issues/628)) ([8f79a00](https://github.com/morluto/jacobian/commit/8f79a007287922cc17a7bdc69d12c4062be32c65))
* **tests:** audit and clean up test anti-patterns ([#641](https://github.com/morluto/jacobian/issues/641)) ([7eafbea](https://github.com/morluto/jacobian/commit/7eafbea0c7234b57b4acc0ff6a2300919a1a7dc0))
* **tests:** avoid sharing template blob inodes ([2d3d260](https://github.com/morluto/jacobian/commit/2d3d2601da23ed79452f17dadb2c72483b3afe91))
* **tests:** avoid sharing template blob inodes ([a729e16](https://github.com/morluto/jacobian/commit/a729e16e109719e96b43995d40879ae7ff392ec2))
* **verification:** gate VERIFIED on TRUE and fail closed on unknown ops ([721635e](https://github.com/morluto/jacobian/commit/721635e0118fdb556854e41185fd4aef5efe3428))
* **verification:** include supporting artifact metadata ([#711](https://github.com/morluto/jacobian/issues/711)) ([6714453](https://github.com/morluto/jacobian/commit/67144534397c4876f79ba7a1a3290c98bb18445f))
* **verification:** verify accepted false conclusions ([d478e2e](https://github.com/morluto/jacobian/commit/d478e2e4bf14844022975309c782ad465f3a1577))
* **verifier:** retain false-certification diagnostics ([#773](https://github.com/morluto/jacobian/issues/773)) ([a3cbb91](https://github.com/morluto/jacobian/commit/a3cbb912c0c3ae782564243da512b0b1d0a87cb8))
* verify accepted false conclusions ([dfcc29c](https://github.com/morluto/jacobian/commit/dfcc29c010fb004bb3fd5bf7faaeb6a090b7698a))


### Performance Improvements

* **benchmarks:** cache schema validators with lru_cache ([#626](https://github.com/morluto/jacobian/issues/626)) ([f55147f](https://github.com/morluto/jacobian/commit/f55147f9f1735a556085f059690c8e442c446bec))
* **schema:** skip meta-schema validation for Pydantic-generated schemas ([c928039](https://github.com/morluto/jacobian/commit/c9280392842f78eeb6aa81abd8e2e838cdbe59ee))
* **schema:** skip meta-schema validation for Pydantic-generated schemas ([18d6042](https://github.com/morluto/jacobian/commit/18d60427806e889edff61b917abd7fc369e29c35))
* **storage:** reuse prepared artifact identities ([#747](https://github.com/morluto/jacobian/issues/747)) ([d43167a](https://github.com/morluto/jacobian/commit/d43167ab27d3f917b41961d0b3ac844a163eb970))
* **tests:** hardlink immutable blobs in template copy ([65b0092](https://github.com/morluto/jacobian/commit/65b0092d5cc72edebf0e23b11c6cf5b60a377932))
* **tests:** hardlink immutable blobs in template copy instead of copying ([26fc5af](https://github.com/morluto/jacobian/commit/26fc5afd50c9674e5ea85b828e9dc2d023e42473))


### Documentation

* add comprehensive audit report ([e42392b](https://github.com/morluto/jacobian/commit/e42392b57e4ed61f131213f1c2ff3b841c03cfdf))
* **agents:** define mathematical interoperability rules ([#769](https://github.com/morluto/jacobian/issues/769)) ([3097458](https://github.com/morluto/jacobian/commit/309745890cd29da85d1abc41f2522755ce926af4))
* **benchmarks:** document prebuilt agent environment direction ([e24dbf6](https://github.com/morluto/jacobian/commit/e24dbf6b8aa5c94531958e5b3efcef17b2b51dcc))
* clarify inline verification evidence ([53bd6aa](https://github.com/morluto/jacobian/commit/53bd6aafe24b6f927ad01726baa6bd6e0f310020))
* clarify mathematical agent workflow ([033bb2b](https://github.com/morluto/jacobian/commit/033bb2bee006092d9b8be472e646be0fca830c4d))
* explain why atomic capabilities scale ([a9df893](https://github.com/morluto/jacobian/commit/a9df893083fd7b16bafd70cc287f521e15c76234))
* **skill:** capture benchmark validation lessons ([294e38f](https://github.com/morluto/jacobian/commit/294e38f4490a0db842f53eb0d56be79d252bddc2))
* **skills:** capture verifier validation lessons ([ec56113](https://github.com/morluto/jacobian/commit/ec561139e0a11c26c4df65dd8b206757cd660880))

## [0.9.0](https://github.com/morluto/jacobian/compare/jacobian-v0.8.0...jacobian-v0.9.0) (2026-08-06)


### Features

* **benchmarks:** normalize reasoning protocol evidence ([4e0abb2](https://github.com/morluto/jacobian/commit/4e0abb28f20f34900a61b456a94bbfa2a17af186))
* **codex:** make Jacobian math affordances visible and reduce tool context cost ([#563](https://github.com/morluto/jacobian/issues/563)) ([#564](https://github.com/morluto/jacobian/issues/564))
* **codex:** improve direct math invocation efficiency ([#567](https://github.com/morluto/jacobian/issues/567))
* **combinatorics:** add verified difference-set decisions ([#436](https://github.com/morluto/jacobian/issues/436)) ([757667d](https://github.com/morluto/jacobian/commit/757667d209b9be7417f2d89a0653ca1898df2b89))
* **evaluations:** publish digest-bound Jacobian images ([b8424f8](https://github.com/morluto/jacobian/commit/b8424f8ac0ed554c07f4c4c767d699c139ca2d0a))
* **mcp:** add experimental math tool surface ([1e2a82f](https://github.com/morluto/jacobian/commit/1e2a82f1968e13a9b20efc0e0ace249e6c4c2fd9))
* **mcp:** make math tool names canonical ([98a3213](https://github.com/morluto/jacobian/commit/98a32134e26fcb2ac2166e10ab7c02f55b85a1e2))
* **mcp:** require bounded reasoning logs ([8f04639](https://github.com/morluto/jacobian/commit/8f04639a2fa0aac79e96f9480cd30e0ef238ffa9))
* **npm:** add a guided one-line installer ([#568](https://github.com/morluto/jacobian/issues/568))


### Bug Fixes

* align generated contracts and evaluation evidence ([15b4746](https://github.com/morluto/jacobian/commit/15b474620b220a369d62fb29b96c90b1acee8da5))
* align symbolic coordination evidence diagnostics ([baba112](https://github.com/morluto/jacobian/commit/baba112883caf30c77be6fab9820e312795356df))
* **npm:** consume streamed installers before early exits ([#569](https://github.com/morluto/jacobian/issues/569))
* **benchmark:** close bounded-variation verifier review gaps ([1508b06](https://github.com/morluto/jacobian/commit/1508b06b933202ac8049cd6ef2eddabc811574da))
* **benchmarks:** accept EOF-terminated Steiner certificates ([d8cf7cf](https://github.com/morluto/jacobian/commit/d8cf7cf685cf79e58f3b14ff65402ef744b453e0))
* **benchmarks:** accept rational finite-support evidence ([1d19b05](https://github.com/morluto/jacobian/commit/1d19b0521e126c2362c52801bf8750c8dc2d7816))
* **benchmarks:** accept unordered finite-support witnesses ([a37010e](https://github.com/morluto/jacobian/commit/a37010e7487b110ed62ddeff3b05dded3239bb18))
* **benchmarks:** align sine malformed diagnostics ([922b524](https://github.com/morluto/jacobian/commit/922b52434f18ab11596689f45eb224a8f19624b1))
* **benchmarks:** align Steiner malformed diagnostics ([fa15d53](https://github.com/morluto/jacobian/commit/fa15d538e134f3b3f06cd18182732175bef667d9))
* **benchmarks:** audit vocabulary, probe minimum, evidence descriptor protocol ([5f4b9d0](https://github.com/morluto/jacobian/commit/5f4b9d004abc3f53214d0cf193379958475be51a))
* **benchmarks:** auto-sync verifier checksums and formatting in harbor-check-task ([54bfc6a](https://github.com/morluto/jacobian/commit/54bfc6a5898f9ce074c51df78466bd83d0516975))
* **benchmarks:** bind evidence limitations to submitted limitations ([901475a](https://github.com/morluto/jacobian/commit/901475a63fc08f9dccf8b5b47e4cff785cf503a1))
* **benchmarks:** bind Lean traces to spike contract ([a5739f2](https://github.com/morluto/jacobian/commit/a5739f242313933a4cc9e9cf934ae0b1b7563422))
* **benchmarks:** bound exponential integer parsing ([#501](https://github.com/morluto/jacobian/issues/501)) ([94df513](https://github.com/morluto/jacobian/commit/94df5137fd1e605a5e35bae2c2a81e9457bec191))
* **benchmarks:** bound Steiner evidence parsing ([da5243e](https://github.com/morluto/jacobian/commit/da5243ea315626b41ab5dd8da02131c8134b4df5))
* **benchmarks:** close integer audit contract gaps ([cc89e33](https://github.com/morluto/jacobian/commit/cc89e3338656cdbb7ce9de5488f9882b01197cea))
* **benchmarks:** close integer audit contract gaps ([6972b3d](https://github.com/morluto/jacobian/commit/6972b3dbc93f51afb83e26044a30faead0a7f967))
* **benchmarks:** close remaining verifier coercion gaps ([3ddae6a](https://github.com/morluto/jacobian/commit/3ddae6a5a6486b978d121ee43be2bc83856da8af))
* **benchmarks:** close review contract gaps ([9707cc5](https://github.com/morluto/jacobian/commit/9707cc52fecf61ee62e0de505445057b89c278a9))
* **benchmarks:** decouple scope from contract, report limitation failures ([444cffe](https://github.com/morluto/jacobian/commit/444cffebb13b352bfa73e67e3314f3fce6cd75f0))
* **benchmarks:** format before syncing verifier checksums in harbor-sync ([a7d43ed](https://github.com/morluto/jacobian/commit/a7d43ed4feb7443865a302b4bb235722b40ca91f))
* **benchmarks:** gate scope accuracy on submission contract ([7d1f761](https://github.com/morluto/jacobian/commit/7d1f761d5771e67d15cda4477ba3d582c9076aba))
* **benchmarks:** gate scope accuracy on submission contract ([253fe9f](https://github.com/morluto/jacobian/commit/253fe9fe42f6f18fdf3864363501dfebe397bf90))
* **benchmarks:** guard None submission and validate evidence types ([0b5bf52](https://github.com/morluto/jacobian/commit/0b5bf5276a0827fadd21a9b81f4cb56ce571b4d1))
* **benchmarks:** harden edge-pair ordering verifier ([af2238f](https://github.com/morluto/jacobian/commit/af2238fd124dccb6fb54d2f2fa02213cec252b09))
* **benchmarks:** harden extremal-subset-sum verifier ([b1e9730](https://github.com/morluto/jacobian/commit/b1e97303f73b812070eb82180d50a9bcd684a319))
* **benchmarks:** harden grid transfer verifier against OOM and assurance coupling ([b43fe8e](https://github.com/morluto/jacobian/commit/b43fe8e376df3f0e404fb78f4b170608fb15a0df))
* **benchmarks:** harden Harbor verifiers ([f5ad747](https://github.com/morluto/jacobian/commit/f5ad7474ac79ebf7a387c3aa9e5ef544bf6143d6))
* **benchmarks:** harden Harbor verifiers ([6fb7bf2](https://github.com/morluto/jacobian/commit/6fb7bf215e8eaff8b3cee940ac9518da0f80c8c9))
* **benchmarks:** harden image complement verifier against OOM and assurance coupling ([06f5590](https://github.com/morluto/jacobian/commit/06f5590d25ee9df945d49d2041a9b89be201f146))
* **benchmarks:** harden image complement verifier against OOM and assurance coupling ([4d718a7](https://github.com/morluto/jacobian/commit/4d718a7eedd67cda1399d9c1887f8fa6d9f99230))
* **benchmarks:** harden integer perturbation domain verifier ([c73b5f5](https://github.com/morluto/jacobian/commit/c73b5f58ec7ca932542f077b6088842ddf219387))
* **benchmarks:** harden inversion aggregate verifier ([d904e14](https://github.com/morluto/jacobian/commit/d904e143bc95b8410582c5127ca78eb74bd70b45))
* **benchmarks:** harden lp-integrability verifier diagnostics and evidence bound ([dd5a82f](https://github.com/morluto/jacobian/commit/dd5a82f26d2886b3e4a168e222e5601ecdda5ace))
* **benchmarks:** harden Metamath verifier contract ([5f1e55a](https://github.com/morluto/jacobian/commit/5f1e55a09918e5f61006521b64d046daba98ba1b))
* **benchmarks:** harden necklace-burnside verifier diagnostics ([bc10026](https://github.com/morluto/jacobian/commit/bc10026ff9e29826bdb419148fa4902ef9f8dc2a))
* **benchmarks:** harden permutation-inversion-involution verifier ([591ecff](https://github.com/morluto/jacobian/commit/591ecff71a7115e854c6c25e58199d8685e45e10))
* **benchmarks:** harden pythagorean verifier diagnostics and evidence bound ([08ea8bc](https://github.com/morluto/jacobian/commit/08ea8bcd5b5c520dd22f323120f908dd66e3bd0e))
* **benchmarks:** harden radical-distance-triangle-certificate verifier ([da99b8b](https://github.com/morluto/jacobian/commit/da99b8bd1321b8a6b18856fbffbc54bbc63c1484))
* **benchmarks:** harden sine integral audit ([47f279c](https://github.com/morluto/jacobian/commit/47f279c8f10ef673f383ea7e521a492a63e06a35))
* **benchmarks:** harden sine protocol diagnostics ([c6839df](https://github.com/morluto/jacobian/commit/c6839df591e3d7a80215fd222e746be7bdb0874d))
* **benchmarks:** harden Steiner design verification ([6031296](https://github.com/morluto/jacobian/commit/60312967a831f59a261b41045370a587e6750185))
* **benchmarks:** make Metamath oracle self-contained ([5dda293](https://github.com/morluto/jacobian/commit/5dda293626d5dca8c5e02c05ba38cc345dc59705))
* **benchmarks:** make verifier support task-owned ([5af350e](https://github.com/morluto/jacobian/commit/5af350e071234e49f01ca5818125b55fc7c5795a))
* **benchmarks:** normalize nested collections before comparison ([43060a8](https://github.com/morluto/jacobian/commit/43060a8c7df41348e7f8bff801073cc1f1905d0b))
* **benchmarks:** preserve observation trial statuses ([eed6a40](https://github.com/morluto/jacobian/commit/eed6a4065d0eeb69955531bd7cf2775574fa1020))
* **benchmarks:** preserve observation trial statuses ([978602f](https://github.com/morluto/jacobian/commit/978602f810a8ca1d555721e861318baa25838ad7))
* **benchmarks:** preserve scope assurance exceptions ([11d1c1d](https://github.com/morluto/jacobian/commit/11d1c1d660d2f11e1b9e244923b922e1a848f0bd))
* **benchmarks:** publish auxiliary congruence identities ([cfa1950](https://github.com/morluto/jacobian/commit/cfa1950f4d36c60bdac953f4f57114acf3cfe7b5))
* **benchmarks:** reject non-integer trace values, decouple scope from assurance ([e2ddbf3](https://github.com/morluto/jacobian/commit/e2ddbf391296893d17b93cddeee25a1601e3fad1))
* **benchmarks:** remove forbidden 'jacobian' word from instruction ([5d1b6c6](https://github.com/morluto/jacobian/commit/5d1b6c62dd77db5d50c9587578a29736494f829a))
* **benchmarks:** remove nonexistent evidence_schema.json from environment Dockerfile ([f0f1fa0](https://github.com/morluto/jacobian/commit/f0f1fa0c79c6aedebf6068b6b79af7e1b8f3bde7))
* **benchmarks:** remove Steiner evidence byte cap ([8b908c0](https://github.com/morluto/jacobian/commit/8b908c09d5dfef23d2f671e27380e0fdd9ecd0b8))
* **benchmarks:** restore integer audit Harbor contract ([397c757](https://github.com/morluto/jacobian/commit/397c7576dc74ad5783597377c5c40d4e1a8a188f))
* **benchmarks:** semantic validation, exact evidence types, COMPLETE-only ([41bf5b9](https://github.com/morluto/jacobian/commit/41bf5b9736ff2add87f06c9999ea5cd0e2e17ef1))
* **benchmarks:** separate sine audit diagnostics ([22fe785](https://github.com/morluto/jacobian/commit/22fe785e32ab8b44338a005d632756393243e71f))
* **benchmarks:** separate Steiner assurance diagnostics ([1328468](https://github.com/morluto/jacobian/commit/132846875c2ffe4d81bfa89f5081052a31f08ab3))
* **benchmarks:** separate Steiner diagnostics ([d8b29d4](https://github.com/morluto/jacobian/commit/d8b29d40dc8a0feb1352b96699548458b3322a72))
* **benchmarks:** stream monotone audit evidence safely ([#502](https://github.com/morluto/jacobian/issues/502)) ([9aa7822](https://github.com/morluto/jacobian/commit/9aa7822c8623759d129f0de955fba89f50937331))
* **benchmarks:** validate construction formulas and bound recursion ([68c03b3](https://github.com/morluto/jacobian/commit/68c03b3fbec22aa728495c7dec65174758d07a45))
* **benchmarks:** validate held-out evidence run entries ([d428f26](https://github.com/morluto/jacobian/commit/d428f2616f02da21efa5e92e8299931593c0b272))
* **benchmarks:** validate held-out evidence runs ([823e13c](https://github.com/morluto/jacobian/commit/823e13cb18d05719096dc879ab9ffebff7e0a603))
* **benchmarks:** validate Lean trace completion evidence ([9a4b79f](https://github.com/morluto/jacobian/commit/9a4b79fdcac43b0040bd84a46452463c044fd104))
* **benchmarks:** validate result shape and preserve diagnostics on input tamper ([3b1fffe](https://github.com/morluto/jacobian/commit/3b1fffe7d4c3aa060c23fa45f76a9fc9b5fb8258))
* **benchmarks:** validate result shape and preserve diagnostics on input tamper ([068a563](https://github.com/morluto/jacobian/commit/068a563f0c48b591a60e7e6e4a021264887330b7))
* **ci:** focus benchmark host validation ([a9a0562](https://github.com/morluto/jacobian/commit/a9a056227757735827dc8b6fdc712416db0ee198))
* **ci:** forward pytest arguments through topology runner ([e286366](https://github.com/morluto/jacobian/commit/e286366dcd5906f73e39ce7c95ff9c3c58ad3ef1))
* **ci:** let explicit xdist args suppress lane worker pool ([70ce3e4](https://github.com/morluto/jacobian/commit/70ce3e46ae1a310cf7a73d970151bc013e39c646))
* **ci:** remove redundant local validation work ([#512](https://github.com/morluto/jacobian/issues/512)) ([3939c0f](https://github.com/morluto/jacobian/commit/3939c0fc6998edc96570608286eab970317f1eff))
* close execution and benchmark review gaps ([318d302](https://github.com/morluto/jacobian/commit/318d3022cfda39be8729b68839f546aa761e4181))
* close review and CI boundary gaps ([8e3bf70](https://github.com/morluto/jacobian/commit/8e3bf704f68a5bf875250bc581ae0bd515f19054))
* **devex:** diagnose incompatible Jacobian state in doctor ([937df38](https://github.com/morluto/jacobian/commit/937df3801d23a904d28a2e19a483d154b2dbf3fc))
* **harbor:** bound sine evidence digest ([7a2c744](https://github.com/morluto/jacobian/commit/7a2c744e28ab666418b134abbf1f06806bedcdfb))
* **harbor:** bound Steiner evidence digest ([a0dd2bb](https://github.com/morluto/jacobian/commit/a0dd2bbb230d49e4454d6cced867ce6e72dfc304))
* **harbor:** harden sine audit protocol validation ([93def71](https://github.com/morluto/jacobian/commit/93def711d2b666b0f1cfd66e61f3bc09c94bd4b9))
* **harbor:** separate Steiner protocol diagnostics ([2a0b7ee](https://github.com/morluto/jacobian/commit/2a0b7ee0ec71907f169fb8848fcc8edda41acef0))
* **harbor:** stream Steiner evidence without hidden cap ([13ec89c](https://github.com/morluto/jacobian/commit/13ec89c3112d6d80a8f6192b7178ed2386c3c77d))
* keep bounded-variation diagnostics independent ([43f859a](https://github.com/morluto/jacobian/commit/43f859a2ac36a9b4a2097b9a010af2aa8cbdb2f5))
* keep README-only task edits out of Oracle selection ([d4f653f](https://github.com/morluto/jacobian/commit/d4f653f9dfc8227105237093512de1bfa6aa8d04))
* **mcp:** address post-merge reasoning-log review threads ([fb4f9a6](https://github.com/morluto/jacobian/commit/fb4f9a60ecb8f9b6ad54f1d947002b51e79783b0))
* **mcp:** address post-merge reasoning-log review threads ([acf0bd1](https://github.com/morluto/jacobian/commit/acf0bd1063e494fad8616533f96a2ce656c3f984))
* **mcp:** address reasoning-log review threads ([5a91310](https://github.com/morluto/jacobian/commit/5a9131036adb5996db46dae01e78d5ff0460e8f3))
* **mcp:** harden required reasoning logs ([888f455](https://github.com/morluto/jacobian/commit/888f4554a2d8db112ded53603dff34a449f0acdd))
* **oracle:** align evidence explanation with verifier keyword check ([4fc0be3](https://github.com/morluto/jacobian/commit/4fc0be30c9e6166d100244bc9b6a54f83e85993d))
* **process:** avoid checkpoint lock join deadlock ([3f54f09](https://github.com/morluto/jacobian/commit/3f54f09989dc09f36ed6b596a9f0dafd11eec519))
* **process:** close pipes on post-start failure, fix stderr overflow detection ([#562](https://github.com/morluto/jacobian/issues/562)) ([9cc7e9a](https://github.com/morluto/jacobian/commit/9cc7e9a3a429bf11beb644f5b70f472fd9b68d3d))
* **process:** detect prlimit portably ([bd17aff](https://github.com/morluto/jacobian/commit/bd17aff150345ee461b72ad9355a386f1b069a9b))
* **process:** preserve stderr overflow detection ([7fd976e](https://github.com/morluto/jacobian/commit/7fd976ecae398927fa04e1224feb95f34b6376a5))
* **verifier:** accept schema-allowed completeness in structural diagnostics ([8be1b05](https://github.com/morluto/jacobian/commit/8be1b0527ffdfbb6ff1207c65e433d189b6e32f1))
* **verifier:** address image complement review threads ([14ba1bf](https://github.com/morluto/jacobian/commit/14ba1bf9a1c93a39d73e6c3a6cde31c2de96d27c))
* **verifier:** address necklace Burnside certificate review threads ([302587c](https://github.com/morluto/jacobian/commit/302587cab122b0c6d31e470f2ee9f0c7d9bb7e65))
* **verifier:** address pythagorean generator recurrence review threads ([aac064c](https://github.com/morluto/jacobian/commit/aac064c86c38fd1a007d60d53bddd7ae07d5e0c1))
* **verifier:** bind image complement evidence ([4db87e7](https://github.com/morluto/jacobian/commit/4db87e75796277be7e4b5dd4dcfacd7d3b6c46f1))
* **verifier:** bound workspace input, enforce 16 MiB evidence, decouple scope ([f49a926](https://github.com/morluto/jacobian/commit/f49a92637b6921487b389f3ef76cd307b1cb37c5))
* **verifier:** classify nested schema violations as protocol, emit input_binding ([4711b9d](https://github.com/morluto/jacobian/commit/4711b9ddb842146b4e8f9e6d13f6c5f4553b85b6))
* **verifier:** decouple diagnostics from assurance and reject typed evidence ([ca54359](https://github.com/morluto/jacobian/commit/ca543594173b879060c8069261183fc7c8f6292f))
* **verifier:** decouple evidence and scope diagnostics from assurance ([31ba11c](https://github.com/morluto/jacobian/commit/31ba11c2fabd48542f7ea055d8fb0b2753ebf0af))
* **verifier:** document digest rule, separate abstentions, remove task_id from instruction ([07a7ff7](https://github.com/morluto/jacobian/commit/07a7ff72467692d809fc57792f24e79280ce96fd))
* **verifier:** emit protocol_compliance, decouple evidence from correctness ([5c8469a](https://github.com/morluto/jacobian/commit/5c8469a7ee7e457188cebc8c59b0fa60770f5fb6))
* **verifier:** gate fallback reward on protocol, bound evidence files ([3d3d417](https://github.com/morluto/jacobian/commit/3d3d417ff878f16eec94285f8d5f8e6ce28d0c6e))
* **verifier:** preserve bounded variation syntax ([e7b60ce](https://github.com/morluto/jacobian/commit/e7b60cee33cd330db916505794542e2f46041052))
* **verifiers:** reject malformed evidence and claims ([2148c7b](https://github.com/morluto/jacobian/commit/2148c7b4ffdb0324ad366f75ef16c97a48bde999))
* **verifiers:** reject malformed evidence and claims ([bae0cc1](https://github.com/morluto/jacobian/commit/bae0cc109f33c35064de57b64803f8481f85da03))
* **verifiers:** reject malformed evidence and claims ([8d56124](https://github.com/morluto/jacobian/commit/8d561240a55348fccfc6830146a3b6c1896bc710))
* **verifiers:** reject malformed evidence and claims ([56c5683](https://github.com/morluto/jacobian/commit/56c5683816989046f3ea6bf4ece331eda9a05476))
* **verifiers:** reject malformed evidence and claims ([9592626](https://github.com/morluto/jacobian/commit/9592626f7a7973b4256d0ba30c5d510c2307089c))
* **verifiers:** reject malformed evidence and claims ([85ff351](https://github.com/morluto/jacobian/commit/85ff351c16b3ddb50b56fede983ff99326d060c6))
* **verifiers:** reject malformed evidence and claims ([51df9fd](https://github.com/morluto/jacobian/commit/51df9fd2443596313be1e966ea2569e3120b8c22))
* **verifier:** type-sensitive evidence comparison and independent scope diagnostic ([8cb9bd2](https://github.com/morluto/jacobian/commit/8cb9bd223d155ae0991cf9c6de5a903b027ce5ad))
* **verifier:** validate replay evidence semantics ([12be8ac](https://github.com/morluto/jacobian/commit/12be8ac9ea4612ed501c784160de7ef86015ad41))
* **verifier:** validate representative types, enforce result constraints, refresh checksum ([f0f7ddf](https://github.com/morluto/jacobian/commit/f0f7ddfa968f098c574c7bd782992972db4d8cab))
* **verifier:** validate row types and refresh checksum ([aa72e04](https://github.com/morluto/jacobian/commit/aa72e04b4c9579adf1a642536890868741e03946))


### Dependencies

* **deps-dev:** bump hypothesis from 6.161.6 to 6.164.0 ([#382](https://github.com/morluto/jacobian/issues/382)) ([b84a135](https://github.com/morluto/jacobian/commit/b84a135147c79e14ab43ddca6c0d9427fc79a7e1))


### Documentation

* **benchmarks:** correct migration references ([e3a7e9c](https://github.com/morluto/jacobian/commit/e3a7e9c12b89e4dc210800a7597b078944eb65b3))
* clarify agent-owned mathematical research ([ef00a30](https://github.com/morluto/jacobian/commit/ef00a309bfa13a67ec5021f8b5d8f55e1e33a943))
* clarify execution and evaluation boundaries ([450e313](https://github.com/morluto/jacobian/commit/450e313b12f17477eee23c2a2ee34524f5893657))
* **harbor:** document task-local verifier support ([c3e5d2c](https://github.com/morluto/jacobian/commit/c3e5d2cc3bcb7fcac77f46f2e2e25fe0636f63b1))
* restructure documentation by Diátaxis with domain-owned subdirs ([3ae8651](https://github.com/morluto/jacobian/commit/3ae8651f16a96a35ed124ec10fbddced44af7daa))
* **skill:** name input_binding as a diagnostic dimension, make checksum refresh mandatory ([ae64aae](https://github.com/morluto/jacobian/commit/ae64aae480023ef22560636ed9e62bba4fbffffb))
* **skill:** refresh benchmark dataset guidance ([455a51b](https://github.com/morluto/jacobian/commit/455a51bc76244d2f363fa8dc57fb00ffb2580669))
* **skills:** capture verifier evaluation lessons ([245920d](https://github.com/morluto/jacobian/commit/245920d2df0379fe9e15cdc5f6327fef08f8a49d))
* **verifiers:** codify adversarial contract lessons ([a65bb4a](https://github.com/morluto/jacobian/commit/a65bb4af84e68505bd49e25ad1aea309d604605d))

## [0.8.0](https://github.com/morluto/jacobian/compare/jacobian-v0.7.0...jacobian-v0.8.0) (2026-08-03)


### Features

* **benchmarks:** add task-scoped Harbor gates ([a68b43e](https://github.com/morluto/jacobian/commit/a68b43e2aaf08252120ffe051ee41f3235bcdd09))
* **storage:** remove retired workspace schema ([8f6dc25](https://github.com/morluto/jacobian/commit/8f6dc257a75085aa7e4295fc8bfb1adcd5fc846e))


### Bug Fixes

* address lifecycle, Lean, and Harbor review findings ([e254903](https://github.com/morluto/jacobian/commit/e254903000da70ae1125fcc4e422fb60e1cb0b15))
* **benchmark:** accept positive denominator factors and bound evidence ([11c7ac0](https://github.com/morluto/jacobian/commit/11c7ac0c236093d83fbd1aa0ff48234ebad977c3))
* **benchmark:** accept semantic frozen API scope wording ([981b4a4](https://github.com/morluto/jacobian/commit/981b4a47cf655f57a03e99eea9c9dc8f04af1d7f))
* **benchmark:** align continuant schema and refresh verifier checksum ([ae58f2e](https://github.com/morluto/jacobian/commit/ae58f2e9783c0a5f3f868746c8e9fe51040531a0))
* **benchmark:** allow independently checkable irreducibility certificates ([c550df1](https://github.com/morluto/jacobian/commit/c550df1014ed1fab89622bd92c53036ab59519d8))
* **benchmark:** bind convergence scope to Lebesgue semantics ([84aab8a](https://github.com/morluto/jacobian/commit/84aab8a5b7f1213008a8014bc5ec7c005a9925bd))
* **benchmark:** bind indexed evidence and bound duplicate inputs ([3c5ad74](https://github.com/morluto/jacobian/commit/3c5ad748a3a7a13387fa15484af2e52eb4b4b55a))
* **benchmark:** bind infinite-spectrum evidence to all basis actions ([70fe213](https://github.com/morluto/jacobian/commit/70fe21336da5619cdb354fc89af9f40cbee99585))
* **benchmark:** build research-status evidence semantics from prose only ([949bee5](https://github.com/morluto/jacobian/commit/949bee59c4ea65ab6e3aad57dacfef4d8a317e35))
* **benchmark:** disclose result marker and catch nested JSON ([13c1201](https://github.com/morluto/jacobian/commit/13c12014a88a7af304f40664f81d84e3feb778b4))
* **benchmark:** fail closed on malformed symmetric submissions ([cb637e2](https://github.com/morluto/jacobian/commit/cb637e23b59fb691a6712f463fa5d2491c93f4ec))
* **benchmark:** gate research-status-evidence-audit base reward on evidence, scope, and assurance ([ed44019](https://github.com/morluto/jacobian/commit/ed440193c55d1d397598ac90cf393b11249805e4))
* **benchmark:** harden convergence scope and parsing ([b5bf476](https://github.com/morluto/jacobian/commit/b5bf476a0d9253f2bcdeaada78c954c24e7570cd))
* **benchmark:** harden projection scope and tail parsing ([008a356](https://github.com/morluto/jacobian/commit/008a3561dc5c2e79b3226106c7e91fb75d060d60))
* **benchmark:** harden research-status-evidence-audit verifier ([b9937df](https://github.com/morluto/jacobian/commit/b9937df148c6cc5e99ea59f2919bbf5a892be2d6))
* **benchmark:** make trigonometric derivation strategy agent-owned ([6c6b87c](https://github.com/morluto/jacobian/commit/6c6b87cdf1a638ee07c00dfd10fe664f92aa7a87))
* **benchmark:** refresh research-status verifier checksum ([8127e61](https://github.com/morluto/jacobian/commit/8127e61249e54fd621199fda9a5ddb435590216c))
* **benchmark:** regenerate Harbor verifier checksum and dataset manifest for research-status-evidence-audit ([71072a5](https://github.com/morluto/jacobian/commit/71072a502868b24e91c514f1580e4f0cee173fc4))
* **benchmark:** reject boolean continuant proof lengths ([7a6e21b](https://github.com/morluto/jacobian/commit/7a6e21b5c416b8a0b06b10e04959d1359ec38696))
* **benchmark:** reject negated symmetric scope claims ([dc06808](https://github.com/morluto/jacobian/commit/dc06808243f684e23a3638216b4f52d7047ed826))
* **benchmarks:** align Harbor job schema and digest ([104d94b](https://github.com/morluto/jacobian/commit/104d94b08825c4c8d973efaa9df94fb99fcd8b63))
* **benchmarks:** allow bounded agent networking without web search ([b7108ef](https://github.com/morluto/jacobian/commit/b7108efda5581ab9f9b2b7ddee5a20ee564ca459))
* **benchmarks:** bind adapter evidence to checked results ([5afdb15](https://github.com/morluto/jacobian/commit/5afdb154ab2a2b720871251662a64da27a5a5992))
* **benchmarks:** bind and bound research evidence ([79d0e95](https://github.com/morluto/jacobian/commit/79d0e9585ac5db0b7c5a69fcddf164ec018c382e))
* **benchmarks:** bind snapshot provenance and retire stale research names ([e3ef4e9](https://github.com/morluto/jacobian/commit/e3ef4e9e265a57c0548a3106f407e2e35f5fea40))
* **benchmarks:** complete research-status verifier review fixes ([035cb21](https://github.com/morluto/jacobian/commit/035cb218e917edd0c90530a459bc97094a577812))
* **benchmarks:** fail closed on malformed assurance claims ([faeacb3](https://github.com/morluto/jacobian/commit/faeacb3fd3e9ac78b4eb7acf32935d5571bb45df))
* **benchmarks:** fix import sorting in merged leaf tests and update complexity baseline ([b62d286](https://github.com/morluto/jacobian/commit/b62d2860554d370675f917dcc222ea168b1fea68))
* **benchmarks:** fix import sorting in test_exact_farkas_ldl_slice leaf file ([13a1c38](https://github.com/morluto/jacobian/commit/13a1c38368531da2f4e1d2a7d8d8c0449b147a65))
* **benchmarks:** harden convergence evidence claims ([b0c9c27](https://github.com/morluto/jacobian/commit/b0c9c27cdd081cb3c6985e159832304ecceaa5cd))
* **benchmarks:** keep Jacobian observations offline ([45c4ff9](https://github.com/morluto/jacobian/commit/45c4ff9ccccb5090b97895d1b515a0ad09611d85))
* **benchmarks:** match lowercased evidence term in inseparable-polynomial verifier ([9c0e83c](https://github.com/morluto/jacobian/commit/9c0e83c646bc82036a25697aed9fcb89bf962f16))
* **benchmarks:** refresh Harbor adapter digest ([fe22e54](https://github.com/morluto/jacobian/commit/fe22e544522466d9ca0961a041990507c6203e32))
* **benchmarks:** refresh stale symmetric divisibility verifier checksum ([84710ff](https://github.com/morluto/jacobian/commit/84710ff86913989016c4e2a5a4ff98ddef40983d))
* **benchmarks:** require integral divisibility multipliers ([fca3890](https://github.com/morluto/jacobian/commit/fca3890e3c48dc272bf52f03f118fe7a0700a9df))
* **benchmarks:** ruff format fixup ([5b9fec2](https://github.com/morluto/jacobian/commit/5b9fec2f2508f438282f928f67dc252b93d65d6a))
* **benchmarks:** sync vendored verifier support to canonical source ([00044b2](https://github.com/morluto/jacobian/commit/00044b2ccc8cac5c697c1bc1340a2f275e3d8d21))
* **benchmarks:** update Farkas verifier checksum after ruff format ([9151005](https://github.com/morluto/jacobian/commit/915100504ef1078c76600ffac97585a950e73242))
* **benchmarks:** update gap count, fix lint, update complexity baseline ([fd0dd0e](https://github.com/morluto/jacobian/commit/fd0dd0e54c1c942552dea495ab196a7d6a9543cf))
* **ci:** accept valid irreducibility strategy ([091631d](https://github.com/morluto/jacobian/commit/091631d5505890317182f852d3130b80355d756f))
* **ci:** close planner review gaps ([ea187ab](https://github.com/morluto/jacobian/commit/ea187ab1fc626761ad8101f4320026e38fa3496f))
* **ci:** keep timing artifact failures out of test gates ([5205fd9](https://github.com/morluto/jacobian/commit/5205fd9a37cb5da981bddd3dae7a0d67924882b5))
* **ci:** pass PATHS through temporary files ([35dee17](https://github.com/morluto/jacobian/commit/35dee17ff84babade21002f7545571ef55149b5e))
* **ci:** route test-changed through PATHS file ([c935497](https://github.com/morluto/jacobian/commit/c935497562ab19cd82af24139d91d89f637b74ef))
* **ci:** sync symmetric verifier support ([ac4d66d](https://github.com/morluto/jacobian/commit/ac4d66d8772071fc8a2776ae3508e7aea1dd338a))
* **ci:** sync vendored Harbor verifier support ([058ff8b](https://github.com/morluto/jacobian/commit/058ff8b9aaf78ced46ffb7d60221fa7f05816af6))
* **ci:** sync vendored Harbor verifier support ([357b90c](https://github.com/morluto/jacobian/commit/357b90cd600010221e13bff7ec09b08ee8316b64))
* fail-closed JSON type checks for indexed pairwise vacuity verifier ([29d96c4](https://github.com/morluto/jacobian/commit/29d96c49054e7ddf56d4de47b23e69d325511985))
* **harbor:** make Codex evaluations proxy-aware ([f8bd184](https://github.com/morluto/jacobian/commit/f8bd1847ad9c28483f2a54eeffc7263bb52c6eaa))
* **release:** sync MCP package metadata versions ([9b9d9d5](https://github.com/morluto/jacobian/commit/9b9d9d5e9f95e96e75545c38a4e94e8dd7de890f))
* **security:** update cryptography lock entry ([e0b9b75](https://github.com/morluto/jacobian/commit/e0b9b7509bd601a783ca7dc1e4436018284abbce))
* **storage:** handle foreign keys during workspace removal ([6d0cd7a](https://github.com/morluto/jacobian/commit/6d0cd7ad95a35de1ca1b80cdce3ae9cfe645ea5d))


### Dependencies

* **deps-dev:** bump ruff from 0.16.0 to 0.16.1 ([#379](https://github.com/morluto/jacobian/issues/379)) ([30e59f9](https://github.com/morluto/jacobian/commit/30e59f9e6262ea85ad7ad24180be9ab1ff1578a1))
* **deps-dev:** bump types-networkx ([#381](https://github.com/morluto/jacobian/issues/381)) ([e5e31c5](https://github.com/morluto/jacobian/commit/e5e31c5189e0f7d306bba2e9fbe101e56f33a793))
* **deps:** bump astral-sh/setup-uv from 8.3.2 to 9.0.0 ([#383](https://github.com/morluto/jacobian/issues/383)) ([c61c7f2](https://github.com/morluto/jacobian/commit/c61c7f23211985e6789b1f4588cc76f8c8def272))
* **deps:** bump https://github.com/astral-sh/ruff-pre-commit ([#378](https://github.com/morluto/jacobian/issues/378)) ([213b00b](https://github.com/morluto/jacobian/commit/213b00b3c1bd463c156f44e03ea4673901c8edf0))


### Documentation

* add Simplified Chinese README ([#400](https://github.com/morluto/jacobian/issues/400)) ([94d02bc](https://github.com/morluto/jacobian/commit/94d02bc116f25e63ee5e2c5fbb56b9218fc44d13))
* **benchmarks:** publish research-status scope ([203302e](https://github.com/morluto/jacobian/commit/203302ec6d1f59217d1acf32893ef6445e24199f))
* simplify operation result guidance ([7b6e7cb](https://github.com/morluto/jacobian/commit/7b6e7cbdf50daddcb76dbbb9fb4d6c5d5b819dc7))
* **skills:** add verifier evaluation guidance ([7508f5a](https://github.com/morluto/jacobian/commit/7508f5ae6be5657e9243eec1460f49b81e134a9a))
* **skills:** clarify agent-facing eval contracts ([be9c9cc](https://github.com/morluto/jacobian/commit/be9c9cc50adb68b461ca305a762dcb344b9bb4ae))
* streamline focused validation guidance ([afff87b](https://github.com/morluto/jacobian/commit/afff87b52b5bbd180a801ce6d3e99124021e19d5))
* **workflow:** remove mandatory operation development path ([3aea2dc](https://github.com/morluto/jacobian/commit/3aea2dc8cbba5e940d8b78ba768077d8ff76b185))

## [0.7.0](https://github.com/morluto/jacobian/compare/jacobian-v0.6.0...jacobian-v0.7.0) (2026-08-01)


### Features

* **benchmarks:** expand agent workflow suite to 26 tasks ([#298](https://github.com/morluto/jacobian/issues/298)) ([3dc38cf](https://github.com/morluto/jacobian/commit/3dc38cffcbffc9335a05f126fae7b0d12444e993))
* bootstrap agents from source checkouts ([62e0837](https://github.com/morluto/jacobian/commit/62e08371f769203cf249a91e9c0ec3431264a219))
* bootstrap agents from source checkouts ([4b4331d](https://github.com/morluto/jacobian/commit/4b4331df16cd78e7c136b6f75626d0f3afbfdc89))
* **evals:** add Jacobian control treatment toggle ([f0dd82b](https://github.com/morluto/jacobian/commit/f0dd82be33da0de88a8ec553ceaa4447e641682b))
* **lean:** add axiom closure inspection ([446a5d2](https://github.com/morluto/jacobian/commit/446a5d2ee1644decad1e7e8f2c4a6b6080479a0a))
* **mcp:** evaluate ResourceLink handoffs ([2a3f6f7](https://github.com/morluto/jacobian/commit/2a3f6f7becfc045980653eef1661085fd295714a))
* **npm:** add explicit Python package upgrade command ([2dabcc4](https://github.com/morluto/jacobian/commit/2dabcc4e970355165f6378bd1f329cac2af2f985))
* **storage:** add explicit state revision upgrade ([f851035](https://github.com/morluto/jacobian/commit/f8510358906ef559fd0bd51f0e8aee4cec542500))


### Bug Fixes

* **benchmarks:** accept equivalent Lefschetz witnesses ([11aba34](https://github.com/morluto/jacobian/commit/11aba349483054e92851294cc8285eaee5534d76))
* **benchmarks:** align degree-sequence audit contracts ([78a10cc](https://github.com/morluto/jacobian/commit/78a10cc84d00be88225d298e95d7af28a16c2e89))
* **benchmarks:** bind sharp-bound evidence to result ([faca89e](https://github.com/morluto/jacobian/commit/faca89e3f6c58e593f114c532b4dc88f457debcd))
* **benchmarks:** clarify LCM evidence scope ([f590f70](https://github.com/morluto/jacobian/commit/f590f705ba9677090f00eb22a96d4a3ddfbf7f4a))
* **benchmarks:** distinguish valuation lower bounds ([ff44346](https://github.com/morluto/jacobian/commit/ff44346ce3309830ca55bf20e99d100c6b6f3c18))
* **benchmarks:** enforce generated-lemma witness bounds ([2472849](https://github.com/morluto/jacobian/commit/24728496bf243403feca0610d4319cb83ceee9dc))
* **benchmarks:** enforce symbolic basis ordering ([da60dd7](https://github.com/morluto/jacobian/commit/da60dd79017f4103add693b99200888b0a5f1315))
* **benchmarks:** exercise transitive Lean closure ([268bcd4](https://github.com/morluto/jacobian/commit/268bcd44f9f10c6333da05179d51d267129c7b33))
* **benchmarks:** harden fourth-power scope audit ([eb19ba9](https://github.com/morluto/jacobian/commit/eb19ba9a0b71d29a9b0f6b69b418cbcc72c798cf))
* **benchmarks:** harden Harbor planning and verification ([949540d](https://github.com/morluto/jacobian/commit/949540dc1a9c981752354b2e8a9ad0be5c0d0459))
* **benchmarks:** harden local-density evidence contract ([893aa87](https://github.com/morluto/jacobian/commit/893aa875cb32df607b94838b5790c6a693ad78fe))
* **benchmarks:** publish asymptotic frame normalization ([b48a649](https://github.com/morluto/jacobian/commit/b48a6495cea3a75ca8766b17b9793d6c0c997907))
* **benchmarks:** reduce sharp-bound verifier complexity ([98a5aab](https://github.com/morluto/jacobian/commit/98a5aabca186739c97004e3dac123e1311e33fef))
* **benchmarks:** reduce symbolic verifier complexity ([bb407f4](https://github.com/morluto/jacobian/commit/bb407f45926a8e1f2500da9b4961402aab32c174))
* **benchmarks:** remove prescribed elimination strategy ([196316c](https://github.com/morluto/jacobian/commit/196316c4edc81339206c014b79ea503a9c227081))
* bind bootstrap environment identity ([9af999b](https://github.com/morluto/jacobian/commit/9af999bb41dc3a4bb968324d54cb7e06f846983d))
* bind bootstrap state and image identity ([20330df](https://github.com/morluto/jacobian/commit/20330dfd3551984ac4fb63bacef2cf6717e1c16b))
* **build:** use available uv trixie image ([3057357](https://github.com/morluto/jacobian/commit/30573574efac455a55e78873b8af306b132d4768))
* **ci:** close benchmark validation gaps ([6e4f31f](https://github.com/morluto/jacobian/commit/6e4f31fc8beca5bc0df720068fde280cbd9a3ab9))
* **ci:** handle Oracle and npm runtime environments ([92a93c5](https://github.com/morluto/jacobian/commit/92a93c53fedb7944dbd411780544e73c97d5609e))
* **ci:** run checks after pull request base edits ([b1b501a](https://github.com/morluto/jacobian/commit/b1b501af7ff8f16662e99c36813a4be2bfa99dcd))
* **ci:** run checks after pull request base edits ([2e9577f](https://github.com/morluto/jacobian/commit/2e9577f00edcf0004a9969926725de76b7f4fbe5))
* **ci:** separate Lean ownership from PR scheduling ([b686c25](https://github.com/morluto/jacobian/commit/b686c256f423b3247693419b085507c5efc37d31))
* close bootstrap transaction edge cases ([dd34daa](https://github.com/morluto/jacobian/commit/dd34daa075fdcff8a0bcaa435f122fe26357e5f5))
* close source launcher identity gaps ([b661065](https://github.com/morluto/jacobian/commit/b661065735b07133287c82c31c99787ed663128e))
* **evals:** make Docker proxy configuration optional ([afe691d](https://github.com/morluto/jacobian/commit/afe691dfa4be52f04c7963d781bd067367ff54a8))
* **evals:** make Docker proxy configuration optional ([b65e1cb](https://github.com/morluto/jacobian/commit/b65e1cbd8100f40db827e0804093a5fc14cc5da6))
* **harbor:** allow model phase network dependencies ([21fdade](https://github.com/morluto/jacobian/commit/21fdadeba593b97c2d46899e997d7e85ea0a57c8))
* **harbor:** allow single-task observation runs ([fc66115](https://github.com/morluto/jacobian/commit/fc661153a476d3e436e1f4bf3d2319ec8deea0d8))
* **harbor:** constrain observation to prepared tasks ([4420d97](https://github.com/morluto/jacobian/commit/4420d9750a93fe863026a6263337670178b77777))
* **harbor:** normalize result paths before evidence capture ([81f0a94](https://github.com/morluto/jacobian/commit/81f0a948b5a06220476f329c45b2d73956b13faa))
* **harbor:** preinstall codex in graph task image ([1faa22a](https://github.com/morluto/jacobian/commit/1faa22a3baf6101eae12eb692b82d5322bbf670b))
* harden source bootstrap preflight ([2821e71](https://github.com/morluto/jacobian/commit/2821e7107e7b896d7320630db05f91f0c8caa958))
* honor directory-only state ignores ([21dce13](https://github.com/morluto/jacobian/commit/21dce13763434a3e497b05d3a70dda977517612f))
* **npm:** launch npx wrapper without exec ([0751650](https://github.com/morluto/jacobian/commit/0751650b4cb8f8c8d7079211cbe97d1a9a2d5b61))
* **observation:** allow local auth proxy image override ([dc9dd0b](https://github.com/morluto/jacobian/commit/dc9dd0b718843e7e756cbe976426148067f69371))
* **observation:** use localhost for egress sidecars ([cae82bf](https://github.com/morluto/jacobian/commit/cae82bfb18d2e19d45f3f0162e6ed56f2bfa56df))
* reject dirty source before Python discovery ([1b73493](https://github.com/morluto/jacobian/commit/1b73493b7578bf7f4f84bbe4e5bceff8e5b1d68c))
* **review:** close trust-boundary findings ([d1fbde8](https://github.com/morluto/jacobian/commit/d1fbde83a43217f485b46fa4b24e7f3a213cf7ae))
* **runtime:** preserve trust and persistence invariants ([42cb53e](https://github.com/morluto/jacobian/commit/42cb53e0a88fafb8f15f8f05df1062023ce54e12))
* **tests:** follow canonical Harbor task paths ([0b75057](https://github.com/morluto/jacobian/commit/0b750577b6cee9d2e1be674dadae916b8a06024b))
* version MCP registry metadata with releases ([11e3c4c](https://github.com/morluto/jacobian/commit/11e3c4cf2e3bf22b03a501e021fd37a03b938113))


### Documentation

* align 0.6 support status ([7564200](https://github.com/morluto/jacobian/commit/7564200e6a5d95f211016db3591ae900f692c045))
* align benchmark skills with canonical layout ([5138006](https://github.com/morluto/jacobian/commit/5138006602be4a5d8146078629824a48358b4456))
* **evals:** split reference and run guide ([38c5db5](https://github.com/morluto/jacobian/commit/38c5db514bcd6ce0ccb114b2a93257fdf5cdb4fb))
* **lrat:** record Lean-only authority gate ([b1f5bc3](https://github.com/morluto/jacobian/commit/b1f5bc36411a30853f2f1e5345a2eb0090135cb2))
* **skills:** tighten benchmark evaluation gates ([ba3dc7f](https://github.com/morluto/jacobian/commit/ba3dc7fc4f5edcd7b8dbad3e68734ba3dcd6d056))
* **skills:** update Harbor dataset workflow ([cfb15f0](https://github.com/morluto/jacobian/commit/cfb15f08428dcbf6fc1ceafad463824ae4db95ad))

## [0.6.0](https://github.com/morluto/jacobian/compare/jacobian-v0.6.0-alpha.0...jacobian-v0.6.0) (2026-07-31)


### Features

* **benchmarks:** add autoformalization semantic audit ([6666594](https://github.com/morluto/jacobian/commit/6666594e7bc4435a5037f3e941f2951d9d6c66cb))
* **benchmarks:** add curated resource-derived Harbor tasks ([4dee6b6](https://github.com/morluto/jacobian/commit/4dee6b63c2c243200f60462d55107f07f595a092))
* **benchmarks:** add divisibility construction witness ([be22f2c](https://github.com/morluto/jacobian/commit/be22f2c9f53c707b9a0b016f518068cafa182b56))
* **benchmarks:** add Euler-line symbolic certificate task ([fa035ab](https://github.com/morluto/jacobian/commit/fa035abb1a5b3ead36f236f28fc17002805103f7))
* **benchmarks:** add grounded premise proof ([748699e](https://github.com/morluto/jacobian/commit/748699e1775ea8ee4b1ec26ea2a50207e84ec3e9))
* **benchmarks:** add layered meta-verification audit ([5234cb1](https://github.com/morluto/jacobian/commit/5234cb1938dc3f5bf20ce37902f2e9a7989ddf8b))
* **benchmarks:** add modular obstruction certificate ([f3494fb](https://github.com/morluto/jacobian/commit/f3494fb8657a8c8cca198453b2618daaa923e2a7))
* **benchmarks:** add proof-audit Harbor tasks ([88e4c91](https://github.com/morluto/jacobian/commit/88e4c911ef0cc0b71935c287dad2f23876260ea5))
* **benchmarks:** convert cases to Harbor-native datasets ([c6a7c01](https://github.com/morluto/jacobian/commit/c6a7c017e71acc115e1fe25e0b3fe7d817b1db4d))
* **benchmarks:** convert cases to Harbor-native datasets ([8ddabcc](https://github.com/morluto/jacobian/commit/8ddabccdee1bee24f4b5f4314d67af2b36efb3c5))
* **graph:** add exact weighted MST ([#284](https://github.com/morluto/jacobian/issues/284)) ([5d4c596](https://github.com/morluto/jacobian/commit/5d4c596fe38a46c05fabd0b4e99699e9deb7cfe9))
* **graph:** record complete local minimality evidence ([71f14f5](https://github.com/morluto/jacobian/commit/71f14f520d5a11b00b5a7a09bf05a64b54605361))
* **math:** add typed native Python API ([552b909](https://github.com/morluto/jacobian/commit/552b9099212f714ea6501625f3e212ee4e831aed))
* **math:** add typed native Python API ([745afa1](https://github.com/morluto/jacobian/commit/745afa1d85be5045639552a2ec84ead12c788670))
* **matrix:** add exact multiplication ([#283](https://github.com/morluto/jacobian/issues/283)) ([ac4bd66](https://github.com/morluto/jacobian/commit/ac4bd6677482eaa55e6af2c5c0b2e6652075c7ea))
* **matrix:** expose exact rational relations ([#255](https://github.com/morluto/jacobian/issues/255)) ([b7ace44](https://github.com/morluto/jacobian/commit/b7ace445069ef7804132825f2d02367c91650297))
* **mcp:** align adapter with released SDK ([66f83bc](https://github.com/morluto/jacobian/commit/66f83bc961da7bc3e7518b6d27c16806dc39c969))
* **modular:** compute bounded polynomial residue images ([#249](https://github.com/morluto/jacobian/issues/249)) ([378c882](https://github.com/morluto/jacobian/commit/378c882be32cfcf63441b0f7189d837e55477f45))
* **modular:** independently verify residue images ([#251](https://github.com/morluto/jacobian/issues/251)) ([62bdb55](https://github.com/morluto/jacobian/commit/62bdb559b6195270e168e426ccb6a9539709e341))
* **plugins:** enforce typed request contracts ([eeae9ac](https://github.com/morluto/jacobian/commit/eeae9ac6c9f736f454a6fa8ba1592fc0ca3d6144))
* **process:** enforce canonical worker protocols ([df7d8e6](https://github.com/morluto/jacobian/commit/df7d8e6f578d5044bf2bc5e6634f013e50f00432))
* **provider:** separate identity and readiness checks ([fda833d](https://github.com/morluto/jacobian/commit/fda833d7aa25cc6bd2e572f6c026ce3132397e10))
* **storage:** centralize persisted model decoding ([1bf1c3d](https://github.com/morluto/jacobian/commit/1bf1c3dbcc63dd2bac2a812a63037a4953fc4aaf))
* **verification:** bound proof artifacts and LRAT authority ([cfd564f](https://github.com/morluto/jacobian/commit/cfd564fef764064b03e430d7e8620ae451d62894))


### Bug Fixes

* **benchmarks:** accept bounded semantic witnesses ([fec7d89](https://github.com/morluto/jacobian/commit/fec7d8999bbb4ab7dcb21bb227fb6ce23c9547e1))
* **benchmarks:** bind Euler certificate evidence ([c006ed7](https://github.com/morluto/jacobian/commit/c006ed7ae84465b83c773e4fab227be196016b86))
* **benchmarks:** bind Euler certificate evidence ([36ea800](https://github.com/morluto/jacobian/commit/36ea80045aaf8e66170faffc36c96f5aea152d12))
* **benchmarks:** bind modular obstruction evidence ([c9f7c3b](https://github.com/morluto/jacobian/commit/c9f7c3b02e0563d9104e725869dbd655e7e020fe))
* **benchmarks:** bind modular obstruction evidence ([bdd8070](https://github.com/morluto/jacobian/commit/bdd8070aa833244c66ef3dc249c362bbbedcb6e7))
* **benchmarks:** bind pairing evidence ([f26ad0b](https://github.com/morluto/jacobian/commit/f26ad0b534f81002b731203009a2b0c7469a6df3))
* **benchmarks:** bind semantic audit evidence ([422eeac](https://github.com/morluto/jacobian/commit/422eeac31959b4d6d393365df442af40d425a4cd))
* **benchmarks:** bind semantic audit evidence ([fcf9904](https://github.com/morluto/jacobian/commit/fcf990486d89cd206d13f0b505e941509638bb7e))
* **benchmarks:** enforce two-dimensional audit witnesses ([e322d12](https://github.com/morluto/jacobian/commit/e322d127ab43a5a9d7c645cf460b1f9b0f91e625))
* **benchmarks:** harden resource-derived verifiers ([7129419](https://github.com/morluto/jacobian/commit/7129419a5160cb377970b92b5d3af482d9a1eb7c))
* **benchmarks:** harden resource-derived verifiers ([02e7a71](https://github.com/morluto/jacobian/commit/02e7a71878dc74095b2382529420ab71a2fb86ce))
* **benchmarks:** harden resource-derived verifiers ([919362d](https://github.com/morluto/jacobian/commit/919362d3906efb6b1192ce47dbdbc0eb10ebfc72))
* **benchmarks:** harden verifier evidence and fixture binding ([2a8be13](https://github.com/morluto/jacobian/commit/2a8be13dd9d7a8679a8f6535bb8f26a8ba297472))
* **benchmarks:** harden verifier evidence and fixtures ([1c4da95](https://github.com/morluto/jacobian/commit/1c4da95a6647756fc18cbc4d3634d59748435685))
* **benchmarks:** honor rendered provider and fixture contracts ([df0d124](https://github.com/morluto/jacobian/commit/df0d124e6988820409c4671921e6ede1478b6188))
* **benchmarks:** keep Harbor validation offline ([06904ef](https://github.com/morluto/jacobian/commit/06904efaa5e8dfb554e1e23040bb8602f5c0d082))
* **benchmarks:** keep Harbor validation offline ([afb9e18](https://github.com/morluto/jacobian/commit/afb9e18f19178209d327ac269a9e242b19037a28))
* **benchmarks:** keep scoring tests unique ([771b473](https://github.com/morluto/jacobian/commit/771b473229d07227fd32dfbeb2ba9a5c048b9569))
* **benchmarks:** refresh TSP evidence digest ([1c2dace](https://github.com/morluto/jacobian/commit/1c2dacefdc524aba955b987e19a26a568db0a628))
* **benchmarks:** refresh TSP evidence digest ([08507a8](https://github.com/morluto/jacobian/commit/08507a80a3f36fd22ce3c106942f3145e610f215))
* **benchmarks:** validate divisibility evidence ([c0ae540](https://github.com/morluto/jacobian/commit/c0ae5400bc6ba50a9779600875667cb878b24231))
* **benchmarks:** validate divisibility evidence ([b5ceedf](https://github.com/morluto/jacobian/commit/b5ceedfb8d0523e4ba591d2ca9d95d02d50bde66))
* **benchmarks:** validate TSP repair evidence ([da26375](https://github.com/morluto/jacobian/commit/da2637582af61e7959c914660e039c29f66531da))
* **benchmarks:** validate TSP repair evidence ([d1dbce1](https://github.com/morluto/jacobian/commit/d1dbce133d1ac62ce3a3b2a6f45d0226fc928515))
* **checkers:** omit exact replay when its provider is unavailable ([1002833](https://github.com/morluto/jacobian/commit/1002833110bb9591434fe01f988a59af8e06ab8f))
* **checkers:** omit exact replay when its provider is unavailable ([955b6d8](https://github.com/morluto/jacobian/commit/955b6d8a78c6e61b657faca79f02db6edb5c26f6))
* **checkers:** preserve required replay failures ([e4da91e](https://github.com/morluto/jacobian/commit/e4da91e5178b4dcd6162dca82729a940388025c5))
* **math:** reject nested SymPy floats ([fc92842](https://github.com/morluto/jacobian/commit/fc9284203704ba81832e8436dd6b9e8810e49897))
* **release:** publish stable npm channel ([fa39e3f](https://github.com/morluto/jacobian/commit/fa39e3f965ca0f71b75bdad37c0ef8b8039f5bc6))


### Documentation

* **benchmarks:** state natural proof scope ([e80a1c8](https://github.com/morluto/jacobian/commit/e80a1c83a0df300e59d04bab68dfc19778df53cf))
* **math:** define native API boundaries ([34bab03](https://github.com/morluto/jacobian/commit/34bab035a20e09c3ecd807564dbfee26060c9328))

## [0.6.0-alpha.0](https://github.com/morluto/jacobian/compare/jacobian-v0.5.0-alpha.0...jacobian-v0.6.0-alpha.0) (2026-07-30)


### Features

* add license-aware conjecture ingestion ([1858b49](https://github.com/morluto/jacobian/commit/1858b490f9dc4acac0021c68ec8c0dfc2eacb31d))
* **benchmarks:** add versioned math source catalog ([5f80489](https://github.com/morluto/jacobian/commit/5f80489eef182c57518aba5d370d0a4b7fd225dd))
* **benchmarks:** compile deterministic Harbor math tasks ([732c926](https://github.com/morluto/jacobian/commit/732c9268e9673e7eab669c608339824f3ecf04a5))
* **benchmarks:** replace legacy harness with Harbor regression dataset ([e4e9b67](https://github.com/morluto/jacobian/commit/e4e9b67a893576f0b3cab6cb474eb8aaf0820107))
* **harbor:** centralize verifier protocol support ([57dedf2](https://github.com/morluto/jacobian/commit/57dedf2217baedc0f229d5d5377beb980e9c4c58))
* materialize formal dataset rows ([cd7f7e8](https://github.com/morluto/jacobian/commit/cd7f7e8eb05124a2fe7ff0913fbf30ad2b351fbe))
* **polynomial:** add Keller and inverse obstruction verification ([fdc331c](https://github.com/morluto/jacobian/commit/fdc331c58040ef5aaba809564c604afde0bf802f))


### Bug Fixes

* address Lean frontend review findings ([19d5b95](https://github.com/morluto/jacobian/commit/19d5b95252b3af983cae8a09c40871e93f81aacb))
* **benchmarks:** accept bound verification records ([5913175](https://github.com/morluto/jacobian/commit/5913175cae7fa703843b49390d17ed9bd8311906))
* **benchmarks:** accept verification metadata in math_contract ([1176c4d](https://github.com/morluto/jacobian/commit/1176c4d9c1ec707756b18640880136f5e6ffdf92))
* **benchmarks:** add unique LABELs to prevent BuildKit cross-task cache contamination ([ca55f94](https://github.com/morluto/jacobian/commit/ca55f9497fd73538030f648fa43c6f238ebe8229))
* **benchmarks:** address review comments on assurance, scoring, and guidance ([6cf26b6](https://github.com/morluto/jacobian/commit/6cf26b6a0528b308c694df7499625500ec2fcdff))
* **benchmarks:** bind generation to source provenance ([dd5a183](https://github.com/morluto/jacobian/commit/dd5a1838ec49bd2f0cc65d25214b2f220be8c176))
* **benchmarks:** bind remaining evaluation contracts ([a33e353](https://github.com/morluto/jacobian/commit/a33e3534e590926b23dbdc61f1f7e40ab58db450))
* **benchmarks:** close Harbor observation integrity gaps ([9e419f5](https://github.com/morluto/jacobian/commit/9e419f557aebb8dd091aa276a8127d8d80b6c5dc))
* **benchmarks:** close provenance and verifier gaps ([361305a](https://github.com/morluto/jacobian/commit/361305ad318d0cb2464cba8ce7af89b27e6006d3))
* **benchmarks:** close regression verifier gaps ([437bac6](https://github.com/morluto/jacobian/commit/437bac6af00bfb12c334b3f87920b09a13d3940e))
* **benchmarks:** close remaining Harbor review gaps ([9c84306](https://github.com/morluto/jacobian/commit/9c8430681b2362c07d0fcc90b9fd7c455805b849))
* **benchmarks:** complete Harbor verification contracts ([#226](https://github.com/morluto/jacobian/issues/226)) ([74173f9](https://github.com/morluto/jacobian/commit/74173f9cb5e406fc64e6d74060d88fe80dbc6bb3))
* **benchmarks:** fix Docker ENTRYPOINT conflict and job-level env resolution ([3650f23](https://github.com/morluto/jacobian/commit/3650f23fc73ec1abaf0ff428c35c9ccc80aeb44d))
* **benchmarks:** harden Harbor verifier records ([9b1bff4](https://github.com/morluto/jacobian/commit/9b1bff435bed868bd0cd1345686dfce83f80d99b))
* **benchmarks:** harden regression-v1 verification ([5dd350a](https://github.com/morluto/jacobian/commit/5dd350ae0b0f244a693d0b17338dc708b1832dc0))
* **benchmarks:** harden review edge cases ([4b12ea8](https://github.com/morluto/jacobian/commit/4b12ea8ee40121f2cca779c69b052f127cb0a426))
* **benchmarks:** harden verifier and snapshot provenance ([f7b8c2d](https://github.com/morluto/jacobian/commit/f7b8c2dc341bf9d95048587619e6829f111d5e48))
* **benchmarks:** keep regression tasks tool-agnostic ([b9ad03d](https://github.com/morluto/jacobian/commit/b9ad03d3e73dfea727f557f75cb165ec5464688d))
* **benchmarks:** preserve correctness under false certification ([3514b9c](https://github.com/morluto/jacobian/commit/3514b9c073844b1ed46164e93f55861825c987b8))
* **benchmarks:** preserve provider spike evidence ([a1e7236](https://github.com/morluto/jacobian/commit/a1e7236d5a7413610fe718d197643e9300e1d517))
* **benchmarks:** separate mathematical correctness from verification binding ([7aa29c5](https://github.com/morluto/jacobian/commit/7aa29c5d05deaceb4f5c778cd81d8d8098524c41))
* **benchmarks:** set maximum_assurance to VERIFIED where authorized checkers exist ([56b023e](https://github.com/morluto/jacobian/commit/56b023eacafe377bf180425f06bfd3929f92ec4a))
* bind formal dataset derived provenance ([c89ade3](https://github.com/morluto/jacobian/commit/c89ade3017fd685b1506f323547196c6d3ef9f9a))
* **capabilities:** bound exact result materialization ([30438fb](https://github.com/morluto/jacobian/commit/30438fb960e4543f8fe2f8a61591e887d0fae349))
* enforce conjecture ingestion policy invariants ([bdba1a6](https://github.com/morluto/jacobian/commit/bdba1a6fe441f47ef438ef4df34310f26db62548))
* harden formal dataset materialization ([2354077](https://github.com/morluto/jacobian/commit/2354077def4b20e088084f08a233ddfcd6a190f0))
* harden Lean executable replay ([b6da5fa](https://github.com/morluto/jacobian/commit/b6da5fa82ec383c672a9ea2034ed0f813d0cfa2b))
* **ingestion:** canonicalize source provenance and text ([33feba7](https://github.com/morluto/jacobian/commit/33feba7444aeb7813ebd482bdc3e9cb8511b94e6))
* **lean:** align moved frontend assets and test seams ([b02fdbb](https://github.com/morluto/jacobian/commit/b02fdbbcceca73b86dc48fdbcaffb9ebc7862605))
* **mcp:** add type: ignore for untyped MCP decorators ([386ddf3](https://github.com/morluto/jacobian/commit/386ddf3f0bbfa8ad8cd039f2480c55ca6f8f1027))
* **mcp:** add type: ignore for untyped MCP decorators ([ef1727d](https://github.com/morluto/jacobian/commit/ef1727d8780b5827be0e80ec15e1c1196081e3d4))
* **mcp:** measure discovery responses as rendered ([2dd2f6c](https://github.com/morluto/jacobian/commit/2dd2f6c6cd033af3b9f1f319698c6dcb6b88e551))
* **mcp:** validate raw workspace writes ([3980f8d](https://github.com/morluto/jacobian/commit/3980f8de01d60e5860b0fb39dd4ecd0f764d7a49))
* **mcp:** validate the public invocation boundary ([4c11df8](https://github.com/morluto/jacobian/commit/4c11df881b4ae46d7c706b984b8b4d81dd6dad7c))
* **npm:** refresh stale Python package before MCP startup ([86dd077](https://github.com/morluto/jacobian/commit/86dd07748bfcb641bca0abef5be6693152420fed))
* **polynomial:** align inverse witness with negative claim ([9b4289a](https://github.com/morluto/jacobian/commit/9b4289ab21c2c676ec799c1e9d1519bd0225efe1))
* **portfolio:** skip bundles with unavailable dependencies ([5629b10](https://github.com/morluto/jacobian/commit/5629b100c568ca485a73584babcb6ee8896a44b1))
* preserve fail-closed runtime boundaries ([68a3d20](https://github.com/morluto/jacobian/commit/68a3d20b865c3fabb9792d10f85373357c604a26))
* preserve formal source semantics ([3351dfb](https://github.com/morluto/jacobian/commit/3351dfbb63b290aa29a27e316984018d277d2d31))
* protect policy-owned conjecture artifacts ([d505c4f](https://github.com/morluto/jacobian/commit/d505c4f75776ca475a429404927c7c71635a56f5))
* record redacted conjecture episodes ([ca2751f](https://github.com/morluto/jacobian/commit/ca2751fa98be6b350b596cca4d32c4e582d31dce))
* revalidate Lean frontend identity ([8d3e123](https://github.com/morluto/jacobian/commit/8d3e123798dab6493adad270d2097480848682e2))
* ruff fixes and type annotations for recovery helpers ([8cfce62](https://github.com/morluto/jacobian/commit/8cfce62c0711d4ef5ff3b6ed273a8c7514036885))
* ruff format and update C901 baseline after refactor ([d3c6a5c](https://github.com/morluto/jacobian/commit/d3c6a5c506aa5169ac70372a4c79d883c4da2c54))
* **sat:** advertise assignment artifact verification ([805bb39](https://github.com/morluto/jacobian/commit/805bb39891096324ae693b283753001e18b63ab3))
* **search:** serialize starts against shutdown ([1af5565](https://github.com/morluto/jacobian/commit/1af5565ff6a765c589cb40354c0ab747aa374c27))
* **tests:** repair stale _launch monkeypatch and sorted-parents assertions ([1c19f1c](https://github.com/morluto/jacobian/commit/1c19f1c383f0ac40135f4aa243c4bd4f65b7cd0a))
* **tooling:** allow pinned Harbor CLI import ([4044a30](https://github.com/morluto/jacobian/commit/4044a306148b9b0be40d0c3ca393eb9513a0b220))
* **tooling:** pack npm package from its directory ([a507871](https://github.com/morluto/jacobian/commit/a507871668a79a44969886246df43394cd240ea0))
* update C901 complexity baseline after server.py refactor ([2247aaf](https://github.com/morluto/jacobian/commit/2247aaf097c2bb881b89607e5bffc126bb0cd770))


### Documentation

* **benchmarks:** clarify math eval provenance and status ([9aaac8d](https://github.com/morluto/jacobian/commit/9aaac8daca8ebc64ba5e9ae76bd5ed9d191a7e48))
* **benchmarks:** explain math evaluation workflow ([879101f](https://github.com/morluto/jacobian/commit/879101f34463f6b6a3c10628483e587506733062))
* **benchmarks:** use guarded Jacobian runner ([a65879b](https://github.com/morluto/jacobian/commit/a65879b2399079540f0d81c7b04eee4be70d5d88))
* define Harbor workflow observation boundary ([6897390](https://github.com/morluto/jacobian/commit/6897390317b35976b3191be18766e6782c11a97a))
* explain domain-owned installation boundaries ([da218fb](https://github.com/morluto/jacobian/commit/da218fbed19521b395b0f529873ae0a2cce5f193))
* **mcp:** document artifact:// resource envelope format in operating guide ([508aa0f](https://github.com/morluto/jacobian/commit/508aa0f27d00434577ab535b135c95b7866ed787))
* **polynomial:** document why spawn is required for inverse solver ([2ebc5e1](https://github.com/morluto/jacobian/commit/2ebc5e1499a6007a4eb805a88c046ef28bd4c6c4))
* remove shared operation registries ([8a383e4](https://github.com/morluto/jacobian/commit/8a383e48ae20d19b7379fb26544651ed2bbff647))
* **skills:** add Jacobian Harbor benchmark workflow ([d0c9b93](https://github.com/morluto/jacobian/commit/d0c9b9322e40051e0c5abd8209bd9abf59d5787f))

## [0.5.0-alpha.0](https://github.com/morluto/jacobian/compare/jacobian-v0.4.1-alpha.0...jacobian-v0.5.0-alpha.0) (2026-07-29)


### Features

* add one-command MCP deployment installer ([c6121c7](https://github.com/morluto/jacobian/commit/c6121c76275594ee36797dc16828d14f83be41e4))
* add one-command MCP deployment installer ([f960bcd](https://github.com/morluto/jacobian/commit/f960bcd82d8d6e739ff291667d058cc55020aa13))
* **eval:** harden Frontier evaluation transport ([83006d0](https://github.com/morluto/jacobian/commit/83006d0268a0a4cd35fa06756f9279ebc63be532))
* **eval:** harden Frontier evaluation transport ([c7b9efe](https://github.com/morluto/jacobian/commit/c7b9efe701726db314e10fda6d51f9871bc44dd6))
* **graph:** independently verify diameter and radius ([2826ea2](https://github.com/morluto/jacobian/commit/2826ea2e3ec4665eb93eb3b96e060015013860f8))
* **graph:** verify diameter and radius independently ([46a4bd0](https://github.com/morluto/jacobian/commit/46a4bd02da61da18c9d6a65c398e43c14bc8d71f))
* **graph:** verify exact distance matrices ([cbdee9b](https://github.com/morluto/jacobian/commit/cbdee9be478ffac27ebe7144e9c2e1f49c76519e))
* **integer:** independently verify prime factorizations ([4b02dc2](https://github.com/morluto/jacobian/commit/4b02dc2aca5eacd481661bc02ddf9b6b46e2332e))
* **integer:** verify prime factorizations independently ([4d90c25](https://github.com/morluto/jacobian/commit/4d90c257bbc22638c97e6044446ea6780131ccdf))
* **number-theory:** decide powerful numbers ([2fb1c0f](https://github.com/morluto/jacobian/commit/2fb1c0ff65223edf9d2e50d6add621cab9c9d6ea))
* **number-theory:** decide powerful numbers ([9881133](https://github.com/morluto/jacobian/commit/988113312fd2a61933bbf3b8d0cd9036527e4ffd))
* **number-theory:** verify powerful decisions ([508dbd3](https://github.com/morluto/jacobian/commit/508dbd32c703198aae4665b45e396cb3ffe289ed))
* **number-theory:** verify powerful decisions ([ee946ad](https://github.com/morluto/jacobian/commit/ee946ad733277a66cda5d21c63290f84995c9fbd))
* **skills:** orchestrate operation development ([adb3a1b](https://github.com/morluto/jacobian/commit/adb3a1bcd60a728ac178e54d7d26e579b765600b))
* **skills:** orchestrate operation development ([7fb67c9](https://github.com/morluto/jacobian/commit/7fb67c97086a9d807b11ac1448b111f0d13bd895))


### Bug Fixes

* align CI callers with runtime ownership ([d143d69](https://github.com/morluto/jacobian/commit/d143d696e00bff3b5ae95d90ac1c1351deca30da))
* align MCP deployment and agent handoff ([2213172](https://github.com/morluto/jacobian/commit/221317299b2786a0d8d0a80022aa66de46434347))
* align MCP deployment and agent handoff ([0fa8798](https://github.com/morluto/jacobian/commit/0fa879829ab40d2d5f4fb0bec7e3f87473867e27))
* **ci:** close semantic lane review gaps ([7007ece](https://github.com/morluto/jacobian/commit/7007ece42bb4bb0d0cd6475a523906d04d7fa95f))
* **ci:** enforce semantic validation boundaries ([585fc6e](https://github.com/morluto/jacobian/commit/585fc6e0dba61e224244f62028ce3f7aed550932))
* **ci:** restore historical integration timing downloads ([fc6f7f5](https://github.com/morluto/jacobian/commit/fc6f7f5df90a717431cf41fcacf2a2ccaddc9215))
* **cli:** make init output readable by default ([3c3f788](https://github.com/morluto/jacobian/commit/3c3f7886811abab0e2eaf30168e9f38537d62217))
* **combinatorics:** bound large recurrence and series results ([e167792](https://github.com/morluto/jacobian/commit/e167792bbf4820e433e240edfb2ad37235ee53a4))
* **deploy:** build release runtime at final path ([#193](https://github.com/morluto/jacobian/issues/193)) ([d63c4ff](https://github.com/morluto/jacobian/commit/d63c4ff052f3a7861a3a0efb71872c43156b81b5))
* **devex:** align CLI onboarding and validation workflows ([95f2141](https://github.com/morluto/jacobian/commit/95f21417980fb9b940c9841d8b4f56a2cafa64f5))
* **dev:** preserve transaction and platform safety ([199ab26](https://github.com/morluto/jacobian/commit/199ab26391d8206c786eb981ee5db7312283685e))
* encode the idempotence example ([36e0fd6](https://github.com/morluto/jacobian/commit/36e0fd654587e156e2445619b226fb7fb19c1802))
* migrate merged benchmark runtime callers ([70d42ab](https://github.com/morluto/jacobian/commit/70d42ab78b9a8b8c558e065461634f2c2cbd3d4a))
* **nauty:** harden optional provider boundary ([92e44d4](https://github.com/morluto/jacobian/commit/92e44d40adee3c93cfda5634801e071b57310eee))
* **posets:** validate ranks and bound recurrence artifacts ([144d100](https://github.com/morluto/jacobian/commit/144d1000671712af66e1718d732d7387b3d1d7e9))
* **probability:** bound convolution and decouple verification ([85bee90](https://github.com/morluto/jacobian/commit/85bee90b607aba9ac2a3a0cac8a5467510183654))
* **release:** keep npm trusted publishing OIDC-only ([c763724](https://github.com/morluto/jacobian/commit/c76372452a7108984d73ab7bd408180e27210c6f))
* **release:** make npm retries idempotent ([43b6a56](https://github.com/morluto/jacobian/commit/43b6a56c3aab50d0e470ddf7b20b305ec1e87918))
* **release:** remove disabled immutable release gate ([723077d](https://github.com/morluto/jacobian/commit/723077ddb97513094df8fa0474d9412a8c305005))
* **release:** run publishers from main workflow ([04d11a5](https://github.com/morluto/jacobian/commit/04d11a5c82ad033e4d80d5757446a8759d392261))
* **release:** tag prerelease npm publications ([2bdaf5a](https://github.com/morluto/jacobian/commit/2bdaf5ad135bb0156ea225f4aa11c67f0bcedbb0))
* **testing:** resolve package initializer imports ([377cc63](https://github.com/morluto/jacobian/commit/377cc63bddc44fb69ff3f1b7cbaf750cbadbf3bc))
* type matrix invocation examples ([a1ab021](https://github.com/morluto/jacobian/commit/a1ab021bda6e84a17e2c7b6c25d87c177640de67))
* use a unit segment invocation example ([3ffe406](https://github.com/morluto/jacobian/commit/3ffe406bfb1df5dcaf92f1c27206c34900481ce9))


### Performance Improvements

* **ci:** avoid unnecessary Lean validation work ([d9cd246](https://github.com/morluto/jacobian/commit/d9cd246ad2c07e974c090287846075bb079ace01))
* **cli:** initialize the kernel only when commands need it ([73b4c5d](https://github.com/morluto/jacobian/commit/73b4c5d69cc53acaab3fa1a047dddaea09a4aa28))
* **kernel:** reduce repeated schema registration work ([c220830](https://github.com/morluto/jacobian/commit/c2208301de14872c890cc15170ac7a19b32e26b3))
* **runtime:** make populated portfolio attachment write-free ([faa0ceb](https://github.com/morluto/jacobian/commit/faa0ceb9b9d5efd7b790d99d7c86b37e5097044e))
* **store:** batch durable blob publication syncs ([44aac8c](https://github.com/morluto/jacobian/commit/44aac8cf35d3a726509a32bfeadc15d1b775893a))
* **tests:** group reference fixtures in the core lane ([95e138b](https://github.com/morluto/jacobian/commit/95e138b2d46daa59eb88133280feaf51ac5f36fa))
* **topology:** maintain homology bases incrementally ([a28609f](https://github.com/morluto/jacobian/commit/a28609f261c6dfbf028073b0e50d1e83ccdeacfc))


### Documentation

* **audit:** refresh developer experience findings ([124238a](https://github.com/morluto/jacobian/commit/124238a06a26ab41d2374fd0b1b383428a6b666b))
* **capabilities:** correct discovery handoff evidence ([486d663](https://github.com/morluto/jacobian/commit/486d6637c71e41a048d610c7a3f6fa883a654ce8))
* **capabilities:** freeze four-domain discovery gates ([#198](https://github.com/morluto/jacobian/issues/198)) ([d2080f2](https://github.com/morluto/jacobian/commit/d2080f2c013df41816af220f4c2d5f68cf160846))
* **dev:** clarify the local validation workflow ([4144720](https://github.com/morluto/jacobian/commit/41447206d3068b5f80bf87156662d004aafe7774))
* **devex:** align onboarding with validation lanes ([6d4e12f](https://github.com/morluto/jacobian/commit/6d4e12f239eda1361b4b68e8519616633b4cc8ca))
* replace legacy kernel architecture references ([3a185c4](https://github.com/morluto/jacobian/commit/3a185c43ba71879ac3385ebb85a5ccfce9b19ded))
* update probability reproduction paths ([b4bcb1d](https://github.com/morluto/jacobian/commit/b4bcb1d46e6981b405ec153518493d57d511ca95))

## [0.4.1-alpha.0](https://github.com/morluto/jacobian/compare/jacobian-v0.4.0-alpha.0...jacobian-v0.4.1-alpha.0) (2026-07-28)


### Bug Fixes

* **ci:** stabilize release and integration shard gates ([6893e3b](https://github.com/morluto/jacobian/commit/6893e3b2ff599cd1cd31c1b1fb285e871ae2a018))
* **ci:** stabilize release and integration shard gates ([9bfe43a](https://github.com/morluto/jacobian/commit/9bfe43a6a36bb2cd94af9b99452e178405157255))

## [0.4.0-alpha.0](https://github.com/morluto/jacobian/compare/jacobian-v0.3.0-alpha.0...jacobian-v0.4.0-alpha.0) (2026-07-28)


### Features

* add bounded LRAT certificate verification ([fe321e9](https://github.com/morluto/jacobian/commit/fe321e9b008655ddafcdc100d4bc686b77e60f3f))
* **analysis:** add validated computation bundle ([6467d75](https://github.com/morluto/jacobian/commit/6467d75318457967c10646d565eee8433d83db35))
* **capabilities:** add deterministic installed discovery ([9dd9855](https://github.com/morluto/jacobian/commit/9dd9855301f9643888f36ae94cf74df6ff474bea))
* **capabilities:** add exact primitive and geometry operations ([273ee39](https://github.com/morluto/jacobian/commit/273ee396cf3f29d33e52086bc572d4c5d9ca8356))
* **capabilities:** add resource-mined atomic domains ([d36dc1e](https://github.com/morluto/jacobian/commit/d36dc1ed07c801d00552634b6e8c1dec80524cd5))
* **capabilities:** define discovery and invocation contracts ([771c453](https://github.com/morluto/jacobian/commit/771c45336ed88a96c129dd1a3c2386c68a5c7893))
* **capabilities:** publish validated invocation examples ([01c804c](https://github.com/morluto/jacobian/commit/01c804c14f66dc2f8bf23f3f22d4cc3202ca611e))
* **checkers:** verify exact domain results independently ([baf3f94](https://github.com/morluto/jacobian/commit/baf3f947ad5a70cc8a90983bcd973d8f6d91a52e))
* **claims:** add deterministic logical decomposition ([8f55572](https://github.com/morluto/jacobian/commit/8f55572cbd54a6e013f61e7d7d29b9a55334f695))
* **claims:** add deterministic logical decomposition ([53453e8](https://github.com/morluto/jacobian/commit/53453e860c994c00d9fee98f012188ef434c4067))
* **claims:** add deterministic logical decomposition ([7bf6621](https://github.com/morluto/jacobian/commit/7bf66219f8f2806fc1e0b26e1aa16d9b887c49ed))
* **claims:** add deterministic logical decomposition ([cda8bf5](https://github.com/morluto/jacobian/commit/cda8bf5c68ba95584e264296f12d4eacfd67cbfc))
* **claims:** add deterministic logical decomposition ([90dd120](https://github.com/morluto/jacobian/commit/90dd120c785a7614243cfac153a93c67c0b6235d))
* **claims:** add deterministic logical decomposition ([82c8f91](https://github.com/morluto/jacobian/commit/82c8f91e95ebfed009218c1b46e165fc8f15ed78))
* **graph:** add exact counterexample invariants ([5932286](https://github.com/morluto/jacobian/commit/5932286a64adaa43d22ad3e640d572689d985ead))
* **lean:** add typed formal intermediates ([8f7d0cd](https://github.com/morluto/jacobian/commit/8f7d0cd62a7acbee2b2e05b29bc03dace5b30f65))
* **matrix-graph:** migrate exact domain operations ([c17c6fe](https://github.com/morluto/jacobian/commit/c17c6fef5d0cd4d14d49d70023cc9144b93d871c))
* **mcp:** expose intent-led discovery and guidance ([5a1322d](https://github.com/morluto/jacobian/commit/5a1322d488f5d50338a80dface57d2e5a7f867dd))
* **mcp:** unblock autonomous SAT composition ([833e960](https://github.com/morluto/jacobian/commit/833e960f019b00ca4101adecbaba69d35aac4660))
* **mcp:** unblock autonomous SAT composition ([a48d835](https://github.com/morluto/jacobian/commit/a48d835a369cf5e8175e64f870d3799f6dfab36a))
* **number-theory:** bound factorization operations ([dd86643](https://github.com/morluto/jacobian/commit/dd866438ed00f23511a166d6c66791a949f7128f))
* **polynomial:** add exact operation bundle ([ac0edd4](https://github.com/morluto/jacobian/commit/ac0edd4bfa829adb91d02886dc38fe68900a86e8))
* **polynomial:** synthesize bounded inverse candidates ([af2303d](https://github.com/morluto/jacobian/commit/af2303d1b19a52ae2a18143fcfa20772c4d5e106))
* **polynomial:** verify two-sided map inverses ([784ae77](https://github.com/morluto/jacobian/commit/784ae77ef03a1cba21562a419e1daa5d8d4799a7))
* **search:** expose bounded optimization obligations ([a26885a](https://github.com/morluto/jacobian/commit/a26885a21005dd31a4cde10f2b54bb7e8a69b944))
* **skills:** add operation implementation workflows ([078c3a4](https://github.com/morluto/jacobian/commit/078c3a4f6a782d836f0570ad7e76902d3f6fc645))


### Bug Fixes

* **analysis:** bind optimization and provider outcomes ([721a260](https://github.com/morluto/jacobian/commit/721a260528229ad8e295f1e1717ed6a4766388d9))
* **capabilities:** use canonical library-backed primitives ([88f951a](https://github.com/morluto/jacobian/commit/88f951a61a55810373887d75f757b9a0e5641cce))
* **checkers:** accept canonical polynomial artifacts ([3a9ed35](https://github.com/morluto/jacobian/commit/3a9ed353441a6429d0458b47e93e9600b40e89e0))
* **checkers:** accept distribution runtime identities ([fabb9d1](https://github.com/morluto/jacobian/commit/fabb9d14d8573f9e752c94ba327963300caa0b7b))
* **checkers:** bind exact FLINT runtime identity ([e1c8dc3](https://github.com/morluto/jacobian/commit/e1c8dc3db96484d482af7a4aec6ec55260b3fd88))
* **checkers:** bind exact replay to semantics artifacts ([7a206f8](https://github.com/morluto/jacobian/commit/7a206f8589e88287035ee8979a1797d46e1a55fb))
* **checkers:** validate geometry semantics digests ([889333d](https://github.com/morluto/jacobian/commit/889333d6e1257e45d7e48c2b213663b817b6c891))
* **ci:** release superseded workflow runs promptly ([eeff467](https://github.com/morluto/jacobian/commit/eeff4677c260e8ac53930fe6f0a38da544f9bc9d))
* **ci:** release superseded workflow runs promptly ([2768027](https://github.com/morluto/jacobian/commit/276802728a40e9d8f9a7eac25340d219597472ce))
* **ci:** use --locked instead of --frozen for fail-fast lockfile validation ([391ab3a](https://github.com/morluto/jacobian/commit/391ab3a73168317ac340d5c7dcbcf0117934f4cf))
* **container:** install optional math providers ([940f2a9](https://github.com/morluto/jacobian/commit/940f2a97035d4af443f9099745094d489dc1e25b))
* **discovery:** match phrases on token boundaries ([27ffad8](https://github.com/morluto/jacobian/commit/27ffad80e04f6698c7ff4170cadd557086b80eae))
* **domains:** handle exact operation edge cases ([c79fb0a](https://github.com/morluto/jacobian/commit/c79fb0aa1905657411fba4ea44f7b5fa76433c7a))
* enforce coverage threshold, fail closed on out-of-tree tests, and harden devex tooling ([c899559](https://github.com/morluto/jacobian/commit/c899559685ed83e3f0a3b20409e84250e9ce98cf))
* enforce coverage threshold, fail closed on out-of-tree tests, and harden devex tooling ([d726368](https://github.com/morluto/jacobian/commit/d7263688cc5f30cc167a5c0b1a12e64c61dd0df1))
* exact rational grid bounds and test hygiene ([751d306](https://github.com/morluto/jacobian/commit/751d306f720ad741e63fe55ce280547639cd8dfd))
* **finite-sets:** cover maximum binary outputs ([af68257](https://github.com/morluto/jacobian/commit/af68257f5af48a39ae94db2c42877d814c3511b9))
* **graph:** bind bounded results to source graphs ([6096dc9](https://github.com/morluto/jacobian/commit/6096dc91db43453bb2502933f94916454c05ca19))
* **graph:** bound exponential oracle paths ([c3d8f0d](https://github.com/morluto/jacobian/commit/c3d8f0db7942208f21a857cd8443bfc313d22d59))
* **graph:** correct proposal artifact return type ([180eb16](https://github.com/morluto/jacobian/commit/180eb16bc4073efe338ddfaf7588975600c14569))
* **graph:** require exact reducer deletions ([153eb05](https://github.com/morluto/jacobian/commit/153eb056bc506cefc38ad8940383dd9dee5f9439))
* harden operation trust and evaluation boundaries ([d8610b1](https://github.com/morluto/jacobian/commit/d8610b1d5b66012d7a6f97c5d03ab4899b4bd409))
* harden operation trust and evaluation boundaries ([7f11734](https://github.com/morluto/jacobian/commit/7f11734cf18d9ae473e9fd3b283294e5f6f8e3a4))
* harden remote MCP solver workflows ([2c7ebd5](https://github.com/morluto/jacobian/commit/2c7ebd5222cf029e5e7a612fe401c5232133585f))
* harden remote MCP solver workflows ([193fe30](https://github.com/morluto/jacobian/commit/193fe30384d4115783423b7344e92e5e99913adf))
* **lean:** align v2 operation contracts ([b0e7383](https://github.com/morluto/jacobian/commit/b0e738380481479907de07a24668d85e6e455ee9))
* **lean:** bind proof edit replay results ([5bf33eb](https://github.com/morluto/jacobian/commit/5bf33eb969a43ac21112ce0140bd87ae3ecbec68))
* **lean:** fail closed on proof validation gaps ([a54c8a5](https://github.com/morluto/jacobian/commit/a54c8a5efc5458cb46e1f60cdfee11ce773dbfcd))
* **lean:** parse Lean 4.31 proposition elaboration output ([b0f1efb](https://github.com/morluto/jacobian/commit/b0f1efb188f8720a7abc1d39deacc1c002231654))
* **make:** drop pcre2 requirement from todo-check ([ace396b](https://github.com/morluto/jacobian/commit/ace396b1101dbbf9d6aea700df70a7b4a40b99f2))
* **mcp:** bound tenant kernel admission ([05f7161](https://github.com/morluto/jacobian/commit/05f716186f610abc2389b9be6109a852483f6514))
* **mcp:** harden token and scope boundary tests ([bc73d6a](https://github.com/morluto/jacobian/commit/bc73d6afb52e1bcdbdf32d61b39c71832ffc34be))
* **memory:** validate structured search requests ([f5b5944](https://github.com/morluto/jacobian/commit/f5b59442ba0ddc64b23fbc50d7433db289cee633))
* **number-theory:** bound factorization predicates ([6bc99ed](https://github.com/morluto/jacobian/commit/6bc99edd88c5de4cd64b3e58f6cb36b091d4f290))
* **optimization:** validate bounded worker requests ([40f7f61](https://github.com/morluto/jacobian/commit/40f7f61dfb919060493c6b6a66dcca5508df06da))
* **polynomial:** enumerate rational grids lazily ([b31a317](https://github.com/morluto/jacobian/commit/b31a317c9509ce9c363bac775a9ebfc0b27b853a))
* **polynomial:** preserve rational partial fractions ([b69a7ab](https://github.com/morluto/jacobian/commit/b69a7ab83c54b051da0f5a3b8740330dcdb7f0a0))
* **polynomials:** validate rational grids by exact deduplicated size ([86293f3](https://github.com/morluto/jacobian/commit/86293f3ce251ab3c2929a7a596ab7bc2df48bd3f))
* **registry:** serialize checker policy updates ([ac1cb46](https://github.com/morluto/jacobian/commit/ac1cb46bd54258a77a7f1bcb62e4d7601bb75d3e))
* **runtime:** bind worker and provider identities ([3d601b0](https://github.com/morluto/jacobian/commit/3d601b0d7d2d6f9b81073e1880869615e799b5b4))
* **runtime:** harden optional bundle startup ([1915e0a](https://github.com/morluto/jacobian/commit/1915e0a29f59878cd9195328c766798989948bbf))
* **runtime:** lazy-load optional solver backends ([2a62415](https://github.com/morluto/jacobian/commit/2a6241553c8e01c8dd60105fc056325abb7da33b))
* **runtime:** preserve clean worker completion ([273a96a](https://github.com/morluto/jacobian/commit/273a96a145184cd29d197b8145f4dcfeb330a5ee))
* **runtime:** preserve installed provider profiles ([1c84179](https://github.com/morluto/jacobian/commit/1c84179466f987cb7f1c4cb8dc0121aee5afbe2a))
* **runtime:** replace removable boundary assertions ([24579e7](https://github.com/morluto/jacobian/commit/24579e769d222eccf4dee45eec619803df4a6fb2))
* **sat:** bound LRAT encoding before decoding ([159da89](https://github.com/morluto/jacobian/commit/159da893c1d63e5ce1b0c3f8e7df2929a6c62269))
* **store:** make blob quota accounting constant time ([20ff6cf](https://github.com/morluto/jacobian/commit/20ff6cf65298ef24d1d9925cebf4b220bd9f515e))
* **store:** recover quota metadata after interrupted writes ([41b1426](https://github.com/morluto/jacobian/commit/41b142675046789869ffd53614dee9f40b851107))
* **store:** serialize database initialization ([c151c10](https://github.com/morluto/jacobian/commit/c151c1005657970fac9177881c308ba0c677e1af))
* **verification:** preserve fail-closed checker outcomes ([836dee1](https://github.com/morluto/jacobian/commit/836dee15d9d419bbba713e7783102de9b53e6407))
* **workers:** preserve bounded operation failures ([7fef60b](https://github.com/morluto/jacobian/commit/7fef60be4d189df175708d4f166651b8ec7b1a4f))


### Performance Improvements

* **benchmarks:** measure populated startup ([eb8ea3c](https://github.com/morluto/jacobian/commit/eb8ea3c192e3ec8c97b88766ef0bc6c06f63d336))
* **graph:** cache Graph Atlas representatives ([77babd0](https://github.com/morluto/jacobian/commit/77babd07f2ee6a105fed4e13561ece3631bf2ede))
* **kernel:** batch durable bootstrap registration ([b71aa9b](https://github.com/morluto/jacobian/commit/b71aa9bc1d082e6fe97179be0c78f4299f580897))
* **polynomial:** enumerate inverse supports directly ([ff81fae](https://github.com/morluto/jacobian/commit/ff81faeac681d7f1e42db8dfb1d458d52f1e1be3))


### Dependencies

* **deps-dev:** bump hypothesis from 6.161.5 to 6.161.6 ([dc6a9b2](https://github.com/morluto/jacobian/commit/dc6a9b25fa038d1f8434057bba1ec932f556d75a))
* **deps:** bump actions/download-artifact from 5.0.0 to 8.0.1 ([3917598](https://github.com/morluto/jacobian/commit/3917598e5cf45d6e6ce456397791cfbbdba63658))
* **deps:** bump actions/setup-python from 6.3.0 to 7.0.0 ([2fed2d5](https://github.com/morluto/jacobian/commit/2fed2d562ec795087c81d5062e5d0d277ff34f49))
* **deps:** bump actions/upload-artifact from 4.6.2 to 7.0.1 ([9ed9229](https://github.com/morluto/jacobian/commit/9ed922948bc928c4f97891988659fbc35e941517))
* **deps:** bump z3-solver from 4.16.0.0 to 5.0.0.0 ([7ef3b2e](https://github.com/morluto/jacobian/commit/7ef3b2e0a5a74fa5b0d66b7b58457a2630c4f5f9))


### Documentation

* add Cursor Cloud environment setup notes to AGENTS.md ([d8d9697](https://github.com/morluto/jacobian/commit/d8d96975565a60407c8bf605031fb39a8929c439))
* add Cursor Cloud environment setup notes to AGENTS.md ([6f6cd97](https://github.com/morluto/jacobian/commit/6f6cd971f2eb08844d6e3743803376f85a5a6157))
* **agents:** preserve agent-owned mathematical strategy ([b9cfd06](https://github.com/morluto/jacobian/commit/b9cfd062c4f288480b00f175e7064505981ae35c))
* clarify domain operation contributor rules ([5857daa](https://github.com/morluto/jacobian/commit/5857daa21eea0ff41591cba9e275940073a2ed20))
* clarify workflow ownership ([d228212](https://github.com/morluto/jacobian/commit/d228212a498841ee4d07d05e740e97a97a7ce108))
* define test suite ownership model ([8ab266e](https://github.com/morluto/jacobian/commit/8ab266e9a092e386a25fbb183ae31996b5cd6153))
* explain domain operation workflows ([42faa2a](https://github.com/morluto/jacobian/commit/42faa2a410e210763518cf77e605150e6a6955ee))
* **mcp:** document discovery prompts and resources ([4fb064e](https://github.com/morluto/jacobian/commit/4fb064e6fde5174b473eb20d40301d74684e2af6))
* record atomic operation mining evidence ([c8458af](https://github.com/morluto/jacobian/commit/c8458af57376b2dbfba4bf2c84a2909f5137aac6))
* record fixture cost fixes after the ownership merge ([cb2706f](https://github.com/morluto/jacobian/commit/cb2706f96013077f6b46c61ecbcf579dfa8c02ce))
* remove operation mining ledger ([c996a2d](https://github.com/morluto/jacobian/commit/c996a2dbe9686a86abd823c0f17cdd796e485931))
* remove obsolete timing history ([1e2ba99](https://github.com/morluto/jacobian/commit/1e2ba996d6e23f2f1c8cb875d4e9409fcfb6d636))
* **skills:** standardize operation handoffs ([6df410e](https://github.com/morluto/jacobian/commit/6df410e0553cbdcf98309abbcac79b1d96b08123))
* **testing:** clarify local validation scope ([af1bb9f](https://github.com/morluto/jacobian/commit/af1bb9fd0e95d55b955832312ba42d4a2e16a36e))

## [0.3.0-alpha.0](https://github.com/morluto/jacobian/compare/jacobian-v0.2.0-alpha.0...jacobian-v0.3.0-alpha.0) (2026-07-26)


### Features

* add agent-ready MCP verification and Lean ([ed1c429](https://github.com/morluto/jacobian/commit/ed1c429bc10704a90f7592e90523b127b2c2d3b2))
* add bounded cvc5 Alethe producer ([7b99e62](https://github.com/morluto/jacobian/commit/7b99e6224833899a4674991b401478b4234c8671))
* add bounded cvc5 Alethe producer ([90a1f91](https://github.com/morluto/jacobian/commit/90a1f912e3584164e42c12919943791c7d3aa768))
* add bounded Erdos-Straus verification ([f08c5aa](https://github.com/morluto/jacobian/commit/f08c5aa73f653cf85e37f9867f988c23790b6c96))
* add operation-first agent workbench ([163b665](https://github.com/morluto/jacobian/commit/163b665f4f764cd94ce5a866c4e07f30af119b4f))
* add operation-first agent workbench ([a24429c](https://github.com/morluto/jacobian/commit/a24429c78f8206a478eaeddc799c241fc79c7441))
* add durable epistemic workspaces ([dc1fa84](https://github.com/morluto/jacobian/commit/dc1fa846dd1c53aa52290568f86bd4b867f0ff26))
* add durable epistemic workspaces ([7128bbe](https://github.com/morluto/jacobian/commit/7128bbe4009c12791497278744f7aad0a6c35e6c))
* add exact mathematical contracts and replay checkers ([686da0a](https://github.com/morluto/jacobian/commit/686da0a3ca0b22e8e028bdf051625e0f78f34e21))
* add replayable math exploration and gated evals ([c143899](https://github.com/morluto/jacobian/commit/c14389953afe8903cc4b0f06cd05e84bf7d25e45))
* add security audit, coverage PR comments, and build caching metrics ([ff6a42f](https://github.com/morluto/jacobian/commit/ff6a42f47db996bcbad6b20438b9353b594725bf))
* add verified finite case partitioning ([e3ae74b](https://github.com/morluto/jacobian/commit/e3ae74bdc15e303bcc344ca1c4577dae271b4e82))
* add verified finite partitioning and paired operation evals ([bb8cd3a](https://github.com/morluto/jacobian/commit/bb8cd3a452c1eb10363f597431354a09a3806e6a))
* **capabilities:** add composable result metadata ([8cb3a4e](https://github.com/morluto/jacobian/commit/8cb3a4ef2a795947444b7f6c1b24c24655f3c59f))
* define canonical SAT artifacts ([f303ff3](https://github.com/morluto/jacobian/commit/f303ff31f7b6145309f7695ae76f72bbfa556bf1))
* expose exact graph, polynomial, and magma operations ([9280602](https://github.com/morluto/jacobian/commit/9280602f429f22661de3f9d662d794732c58c718))
* **graph:** add bounded atlas operation slice ([976fa2f](https://github.com/morluto/jacobian/commit/976fa2fb22d52260819cf570a30660401a570eb5))
* **graph:** add replayable degree-sequence realization ([879b1e9](https://github.com/morluto/jacobian/commit/879b1e913dc6f609d127f56f11195f7d8bbb5bd0))
* implement bounded discovery kernel ([fe66be1](https://github.com/morluto/jacobian/commit/fe66be1b9c290574a6f13a02eee6d0559aadd853))
* implement the v0.1 verification kernel ([35578c7](https://github.com/morluto/jacobian/commit/35578c7da46c0006c90cc11e1fbd14c10a40d553))
* improve agent readiness across 7 criteria ([5f4890f](https://github.com/morluto/jacobian/commit/5f4890ffadf5ae5c4efb378d12984ed6e1bd5942))
* improve autonomous MCP operation use ([27e1f1c](https://github.com/morluto/jacobian/commit/27e1f1c22ed869e63e7b46490090f200715332b2))
* **lean:** expose replayable proof exploration ([2cb6257](https://github.com/morluto/jacobian/commit/2cb6257fb33189dfbca13ef8ba65e93cbffadc69))
* **npm:** add jacobian npm package with MCP launcher and setup wizard ([87d87bf](https://github.com/morluto/jacobian/commit/87d87bf97ecc95e60ffd1cb76703e9071fee0ec3))
* **plugins:** seal installed operation snapshots ([5221c5f](https://github.com/morluto/jacobian/commit/5221c5f03704ea640a6417baecf2edf4c2698746))
* **runtime:** harden provisional M3 and M4 workflows ([8d65b29](https://github.com/morluto/jacobian/commit/8d65b29e4cc4afc3649748f1978b6d02c1769739))
* **runtime:** harden provisional M3 and M4 workflows ([2d2a990](https://github.com/morluto/jacobian/commit/2d2a990e9e04b4440dbb4db5dd143bc91580c8b3))
* verify compatible SMT proofs with Carcara ([#72](https://github.com/morluto/jacobian/issues/72)) ([c5f9c3f](https://github.com/morluto/jacobian/commit/c5f9c3f29c9721055bcdf61b8cfb95362c3137ef))
* verify exact SAT assignments ([520c200](https://github.com/morluto/jacobian/commit/520c200dc9924ecfa11d85d523415efedf28faaf))
* verify exact SAT assignments ([d42de60](https://github.com/morluto/jacobian/commit/d42de60bde7b5db0f9dcc1bb1feb196855450b6c))
* **workflows:** add durable search and conjecture orchestration ([955a03e](https://github.com/morluto/jacobian/commit/955a03ebce8a06925cb29335555962537478d834))


### Bug Fixes

* bind evaluation evidence to reported results ([6eee19d](https://github.com/morluto/jacobian/commit/6eee19d9a49f1ad9d83ec45a14627e05e14648e9))
* bind graph isomorphism verification inputs ([9a7b518](https://github.com/morluto/jacobian/commit/9a7b5185177b2cdc64670a0b789e189af537c660))
* bind polynomial identity verification exactly ([c6e7ce7](https://github.com/morluto/jacobian/commit/c6e7ce70ee2b582a8c60ce696266ee073a68e6e6))
* bind verified relationship endpoints ([edfc5a3](https://github.com/morluto/jacobian/commit/edfc5a3710818f0628a3064ed73a09924dcd1304))
* bind verified relationship obligations ([ac33182](https://github.com/morluto/jacobian/commit/ac33182a05d13aea1300948bb3b4aa60c5d3e0d3))
* **ci:** complete required checks for isolated plans ([60d1c12](https://github.com/morluto/jacobian/commit/60d1c12e9f5e2abbaca434e0e122c84f00eac27e))
* **ci:** complete required skipped checks ([135cc59](https://github.com/morluto/jacobian/commit/135cc590fd0ae421ad96f19c7052356fa81cd480))
* **ci:** preserve complete validation gates ([797a1a5](https://github.com/morluto/jacobian/commit/797a1a5524cc83d786544cca2615f0c35f6c3322))
* **ci:** repair workflow validation permissions ([9891436](https://github.com/morluto/jacobian/commit/98914361a37d981a4b3f8631721e05de1cd09f65))
* **ci:** restore stable Lean status gate ([0f31cda](https://github.com/morluto/jacobian/commit/0f31cda9e9186fe11d4318fcd2166521ab04528a))
* **cli:** stabilize local client error output ([f953af2](https://github.com/morluto/jacobian/commit/f953af27c6ad506d3344426212b3470f478562da))
* **conjectures:** replay source verification records ([6f164be](https://github.com/morluto/jacobian/commit/6f164bec9ae480af8f86e5736350b78dec4f6828))
* declare direct pydantic-core dependency ([94b7cc7](https://github.com/morluto/jacobian/commit/94b7cc731bec278511ec2e6c057bd6a2be7b63e6))
* enforce deadlines across inherited plugin pipes ([4123326](https://github.com/morluto/jacobian/commit/41233266c66b66c30d1756df5a18ebd3dd42044b))
* enforce held-out task contracts ([0116e16](https://github.com/morluto/jacobian/commit/0116e164a6d186fca7b53bf858817cfb2794b9ce))
* **evaluation:** reject incomplete evaluation batches ([353f586](https://github.com/morluto/jacobian/commit/353f5860f949ffe989cf5dc361d7065ee8118490))
* **experiments:** quarantine invalid recovery snapshots and enforce wall-clock budget ([f209bcc](https://github.com/morluto/jacobian/commit/f209bcc6cb1a39ca44a4f8125c731905bef126ca))
* harden container and Lean cold starts ([d559531](https://github.com/morluto/jacobian/commit/d55953156c7bba26602bc1b14237a377305497c8))
* **lean:** enforce runtime resource boundaries ([84c6826](https://github.com/morluto/jacobian/commit/84c682698544623ead2fa6ab5711360ea1972716))
* **lean:** pin runtime resolution and diagnostics ([c1f7d0d](https://github.com/morluto/jacobian/commit/c1f7d0d731d8e567269930730158cc4d6972017a))
* **mcp:** return actionable operation errors ([828eca7](https://github.com/morluto/jacobian/commit/828eca72b8e4e46ed3e73ad69bceb2ed6b804d97))
* **plugins:** bind package data in implementation digests ([79bbe29](https://github.com/morluto/jacobian/commit/79bbe299f4f33009bc03795a572d0ee19a220295))
* **plugins:** reject non-source module implementations ([7c91431](https://github.com/morluto/jacobian/commit/7c91431b543158a4cdd469f95f0dcb708360d828))
* **plugins:** serialize snapshot installation ([dcabfa7](https://github.com/morluto/jacobian/commit/dcabfa7f808668310ce2afd796304ce4a0774074))
* **polynomial:** bind certified satisfaction endpoints ([fc47e48](https://github.com/morluto/jacobian/commit/fc47e48230140843b2b7f3c766ecec5d3557bbc4))
* **polynomial:** bind verified collision relationship ([9ac0a7e](https://github.com/morluto/jacobian/commit/9ac0a7e85db737b26d2914769927d20e1ed17b91))
* **polynomial:** expose collision rejection details ([7d45dd2](https://github.com/morluto/jacobian/commit/7d45dd26fd194f86f0f4e6f6441105cfb7d8139c))
* **polynomial:** narrow collision relationship inputs ([0f5a5af](https://github.com/morluto/jacobian/commit/0f5a5af08b0571e793b97e63a129917bda58aa5a))
* **polynomial:** preserve solution verification truth states ([9ea16f6](https://github.com/morluto/jacobian/commit/9ea16f6397e6b0c9863e6b7d15a9d1392a9cd0a2))
* **polynomial:** report bounded search coverage honestly ([f0c7fe7](https://github.com/morluto/jacobian/commit/f0c7fe7b5fa1a26c771201143cefb117be92af30))
* preserve in-progress release runs ([c795418](https://github.com/morluto/jacobian/commit/c79541828dd4ef26f84efb00010d390f5ebfdb13))
* publish Release Please artifacts in the creating run ([72539f0](https://github.com/morluto/jacobian/commit/72539f03a5dfc00414e7019737eae12e473f3db5))
* **registry:** enforce checker compatibility allowlists and identifier consistency ([5914b1b](https://github.com/morluto/jacobian/commit/5914b1b9b2e9f595e528ca5fe8247746c6ca2c2a))
* report checked graph relationship endpoints ([3317287](https://github.com/morluto/jacobian/commit/3317287e8174178ad2aa7116185a9c1dc84286e3))
* **runtime:** close worker process trees and cover boundaries ([761baef](https://github.com/morluto/jacobian/commit/761baef5449abe0530b62092698ece92574defea))
* **runtime:** make checker and plugin failures actionable ([c09cb30](https://github.com/morluto/jacobian/commit/c09cb300b4aa401e60b01a2f970b567035407bdb))
* **search:** harden archive and checkpoint integrity ([8d0e43f](https://github.com/morluto/jacobian/commit/8d0e43f369caaa8f103997c048e10135a4bbd3df))
* **search:** preserve partial accounting and recovery archives ([b04359e](https://github.com/morluto/jacobian/commit/b04359e308c2b589e66dbe41c0cbed38ea220500))
* **search:** reject incomplete evaluation batches ([3d656fe](https://github.com/morluto/jacobian/commit/3d656fe6fdd45bdec8b8c1b34328d01916a9a2f5))
* **shrinking:** require boundary rejection for local minimality ([d82d6af](https://github.com/morluto/jacobian/commit/d82d6af651358200e2af5d308f8007f6f906c7ce))
* **store:** validate artifact parents and schema references ([976a1c4](https://github.com/morluto/jacobian/commit/976a1c4aa8cfcb6fa0d9719bfa4d7840dd368cc9))
* strengthen MCP verification feedback ([4692421](https://github.com/morluto/jacobian/commit/4692421a4d6e1dc0b4f8d3625b12238cea15385e))
* validate enumerated candidate semantics ([c84cfc1](https://github.com/morluto/jacobian/commit/c84cfc1e4c176f79411ddff2d4095bf63b86f289))
* **workflows:** clarify recovery across math services ([75e0174](https://github.com/morluto/jacobian/commit/75e0174659ec42a549b6eba586e02a9883fa6ad8))


### Performance Improvements

* **capabilities:** cache compiled schema validators ([146d6fc](https://github.com/morluto/jacobian/commit/146d6fc65a11f71a163fe254eeb4d3c3ffbd331a))
* reuse Lean declaration catalog ([9837da4](https://github.com/morluto/jacobian/commit/9837da43402a5a3aa83d329c2391ac0b50bf4ecc))
* reuse Lean declaration catalog ([5b89434](https://github.com/morluto/jacobian/commit/5b89434c2e42062043fc8dff27a2babad591d2d1))
* **schema:** cache compiled validators ([2000b5b](https://github.com/morluto/jacobian/commit/2000b5b662b698036d641c89b936e821c2896183))
* **schemas:** cache generated model schemas ([77b0817](https://github.com/morluto/jacobian/commit/77b081732753d1d10fcf03f65dac4b7cbfaffc3d))
* **schemas:** cache Lean discovery schemas ([b430b07](https://github.com/morluto/jacobian/commit/b430b07d6682b43d1f369f22b302887ec02ae2b9))


### Dependencies

* **deps-dev:** bump hypothesis from 6.161.0 to 6.161.5 ([75e8749](https://github.com/morluto/jacobian/commit/75e8749ce8c36076387decc0b16cf6e5265cdd8d))
* **deps-dev:** bump mypy from 1.20.2 to 2.3.0 ([c2cbc84](https://github.com/morluto/jacobian/commit/c2cbc8459a5a608f98d677bc3e3743f093db932f))
* **deps-dev:** bump pytest-rerunfailures from 15.1 to 16.4 ([fb622ed](https://github.com/morluto/jacobian/commit/fb622ed93ae7b477a55e09402a5253e5c1f2d22f))
* **deps-dev:** bump ruff from 0.15.22 to 0.16.0 ([aba509d](https://github.com/morluto/jacobian/commit/aba509d8cf3b17730f8cebc03edf0a8d1d8f4401))
* **deps-dev:** bump types-networkx ([48e6d9d](https://github.com/morluto/jacobian/commit/48e6d9d71c712215afc2ead3248b66b1eaf161e6))
* **deps-dev:** update mypy requirement from &lt;2,&gt;=1.15 to &gt;=1.15,&lt;3 ([b90be74](https://github.com/morluto/jacobian/commit/b90be743d1f11ea722d5be23a1582be57cd8ffbc))
* **deps-dev:** update pytest-rerunfailures requirement ([b0fb6df](https://github.com/morluto/jacobian/commit/b0fb6dfea63f8e3bac58aac9133324127f585285))
* **deps:** bump actions/checkout from 5.1.0 to 7.0.1 ([183e8f4](https://github.com/morluto/jacobian/commit/183e8f40ebd6becd1e7b3a08c7ecd27507de23f9))
* **deps:** bump actions/setup-node from 4.4.0 to 7.0.0 ([b559abe](https://github.com/morluto/jacobian/commit/b559abeff6f5b8d1c59446625b4ad8ab5d66222f))
* **deps:** bump actions/setup-python from 6.3.0 to 7.0.0 ([af537b6](https://github.com/morluto/jacobian/commit/af537b6f5e340e8e0b2963e370295a107c5dc0e2))
* **deps:** bump https://github.com/astral-sh/ruff-pre-commit ([7490e6f](https://github.com/morluto/jacobian/commit/7490e6f755c92e189336895d046c3d73da723df8))
* **deps:** bump https://github.com/jendrikseipp/vulture ([37ada8a](https://github.com/morluto/jacobian/commit/37ada8a137d3e45a71638b36e56d8dbe642d298b))


### Documentation

* add error message audit ([8eb23cc](https://github.com/morluto/jacobian/commit/8eb23cce3fa1a614bc6a7b8f06ff25d4dbd80148))
* add math operation discovery skill ([7d77034](https://github.com/morluto/jacobian/commit/7d77034c90a66ca29d8554aa07b336d2f0e414bc))
* align contributor guidance with operation model ([15280a3](https://github.com/morluto/jacobian/commit/15280a34abc0d2a5f9cf612ef03e36907271c433))
* align product model around mathematical primitives ([07bddde](https://github.com/morluto/jacobian/commit/07bddde50e9888bd70941636518bbeecc67e9193))
* align the roadmap with the v0.1 implementation ([24d1610](https://github.com/morluto/jacobian/commit/24d16106f5fef5329bb167fa669f0c5685447afc))
* clarify operation portfolio overlap ([a8b9db5](https://github.com/morluto/jacobian/commit/a8b9db5161487d7ca9d73e951ce9262523327e57))
* clarify mathematical operation design ([e682d75](https://github.com/morluto/jacobian/commit/e682d75bb9170e2a4555990514e16b5e5ab66edb))
* clarify mathematical tool contract ([e794768](https://github.com/morluto/jacobian/commit/e794768fdd60f4889f79fa1b2f88ad1bde03ad23))
* clarify named SAT assignment output ([b6d8a20](https://github.com/morluto/jacobian/commit/b6d8a207d7eecde62005d7b0317944a38304202d))
* clarify npm distribution ([ca74d80](https://github.com/morluto/jacobian/commit/ca74d80ff4988dbee3459f6c8ce3bc7aca96f770))
* clarify trust-sensitive Python APIs ([e0ed28e](https://github.com/morluto/jacobian/commit/e0ed28e5fbd8faf2182c1f7405b2843548b6ccc9))
* consolidate release contract on v0.2 ([2c098ee](https://github.com/morluto/jacobian/commit/2c098ee4b09b11de95b7b1f674b147883ccd0ee4))
* define operation overlap evaluation ([41ac50f](https://github.com/morluto/jacobian/commit/41ac50f0f033826606f5c253277ca88e3d885332))
* define operation portfolio direction ([1bac85a](https://github.com/morluto/jacobian/commit/1bac85a865d6a83e1940442aa2d744d405a834ef))
* define operation workflow rollout ([63a77ab](https://github.com/morluto/jacobian/commit/63a77ab66a19929b13f76817570f20601310088a))
* define composable mathematical primitives as the product model ([9a29a1d](https://github.com/morluto/jacobian/commit/9a29a1d6b7665db601d6587c581b7085e9920a06))
* define composable mathematics product model ([dbbbbef](https://github.com/morluto/jacobian/commit/dbbbbef87f531542e25451bda2885c4506c7bba0))
* define Jacobian as mathematical agent tools ([43a5349](https://github.com/morluto/jacobian/commit/43a534935ea780614eb8ee7853da1749ac48cf29))
* define verification kernel roadmap ([42049a2](https://github.com/morluto/jacobian/commit/42049a2181c29cdb0c857cf251bdf951fd1f63c7))
* document Z3 installation on macOS ([8559e53](https://github.com/morluto/jacobian/commit/8559e53ac7ed27810ffbdc8847e66f6a8ab3982d))
* format tutorial example ([0c4eb1b](https://github.com/morluto/jacobian/commit/0c4eb1bdf527b4dcd1998af6fc8e7a147fc0c722))
* gate foundational issues on v0.1 decision ([fdb5bda](https://github.com/morluto/jacobian/commit/fdb5bdad730a0a1ea8777c07e7bb889a525561c2))
* guide agent-facing operation design ([af972cf](https://github.com/morluto/jacobian/commit/af972cf549cfaebe74452011ffcb83eda687d59c))
* link v0.1 decision references ([9ca9103](https://github.com/morluto/jacobian/commit/9ca9103e30d4f8839374a444cccac016bc8a1d43))
* make runtime catalog the operation inventory ([ebff6d1](https://github.com/morluto/jacobian/commit/ebff6d1590ef3c4fdadebc562f509f48ff94113d))
* make the roadmap tool-first ([24e195c](https://github.com/morluto/jacobian/commit/24e195c9e74d14908d148c13f0aba0adffb9bc0f))
* **npm:** add README for the jacobian npm package ([5060425](https://github.com/morluto/jacobian/commit/5060425351e6f151f425f9e5fdc8931dae95586b))
* organize project design and workflow guidance ([fc51e5d](https://github.com/morluto/jacobian/commit/fc51e5d75742dd8d36233190b9cf57cabfb9446e))
* rank atomic operation backends ([1265151](https://github.com/morluto/jacobian/commit/12651519cd3188a4f75f7742661e08f7fe2fa667))
* rank atomic operation backends ([879a255](https://github.com/morluto/jacobian/commit/879a255ebf04357d0cb7c5752215777d8d31d4c5))
* **readme:** add operation list, inline example, collapse macOS block ([8990871](https://github.com/morluto/jacobian/commit/89908711d867d10bb2e5c321b94a9b41232de3b7))
* **readme:** add tagline, badges, and "Why Jacobian?" framing ([febab85](https://github.com/morluto/jacobian/commit/febab85898d9b35290adbc4892069b38e7a83424))
* **readme:** add tagline, badges, and Why Jacobian framing ([b5c2b63](https://github.com/morluto/jacobian/commit/b5c2b63fd9a20d90583125acd0edbc8fced476dd))
* **readme:** adopt archival blackboard imagery ([edf779c](https://github.com/morluto/jacobian/commit/edf779ca0bb2069f64d0b260bbcdf1271d058103))
* **readme:** clarify verification process ([34c5009](https://github.com/morluto/jacobian/commit/34c5009e01379025e40d88640837336575e5d94d))
* **readme:** consolidate opening, trim redundant status prose, fix em dashes ([0b70513](https://github.com/morluto/jacobian/commit/0b7051320af466cfb5aeb8bdca16db15a4bdf57e))
* **readme:** extend archival blackboard style ([81ae909](https://github.com/morluto/jacobian/commit/81ae9091aea5d07d5b0a66e087b5c69138d11618))
* **readme:** improve verification graphic legibility ([0192529](https://github.com/morluto/jacobian/commit/01925293e2590985bba7ce91b3260828096ddcc4))
* **readme:** redesign project overview ([2a02eed](https://github.com/morluto/jacobian/commit/2a02eedfd7f11e097c82bc89d837197e19b3f2da))
* record operation pilot decisions ([8c90d42](https://github.com/morluto/jacobian/commit/8c90d42602fe77c94963a292801822a5307655c6))
* record follow-up test cost audit ([8af73e3](https://github.com/morluto/jacobian/commit/8af73e372f905c6357a90a2a25c179653cb418d4))
* record follow-up test cost audit ([f225130](https://github.com/morluto/jacobian/commit/f225130f04ea3c6d15169c2d9583575916bda39e))
* record published issue mapping ([4677eaa](https://github.com/morluto/jacobian/commit/4677eaa81c52f1504cab862f18ecc6dff6374d7c))
* record SAT portfolio evaluation ([f91d08f](https://github.com/morluto/jacobian/commit/f91d08f4b18c92694ad7feb570cfd830dab148ca))
* remove threat-model doc and all references ([36ab219](https://github.com/morluto/jacobian/commit/36ab2196ec6af4c43b295f288782fb527015533c))
* **security:** document sealed workflow controls ([c43a792](https://github.com/morluto/jacobian/commit/c43a792101c4dd38aa5a04dc6c68f33726036dc4))
* streamline agent repository guidance ([9f0df2d](https://github.com/morluto/jacobian/commit/9f0df2de44ab259d58116d9646662a4b234b1969))
* update v0.2 conformance and specification for runtime hardening ([3e576e3](https://github.com/morluto/jacobian/commit/3e576e359ebb11bb3b3bfb31646e382df1c6aa43))
