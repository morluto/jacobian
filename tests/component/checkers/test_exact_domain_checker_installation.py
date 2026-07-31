from __future__ import annotations

from pathlib import Path

from tests.unit.contracts.artifacts import artifact_uri as _uri

from jacobian.contracts.graph_invariant_operations import GraphInvariantRequest
from jacobian.contracts.graph_optimization import (
    GraphHamiltonianPathRequest,
    GraphOptimizationRequest,
)
from jacobian.contracts.jacobian_syzygy import GradedJacobianSyzygyRequest
from jacobian.contracts.matrix_operations import (
    IntegerMatrixRequest,
    RationalMatrixProductRequest,
    RationalMatrixRequest,
    SquareRationalMatrixRequest,
)
from jacobian.contracts.number_theory import (
    FactorizationRequest,
    ModularPolynomialResidueImageRequest,
    PowerfulNumberRequest,
)
from jacobian.contracts.polynomial_operations import (
    PolynomialDiscriminantRequest,
    PolynomialGcdRequest,
    PolynomialResultantRequest,
    PolynomialSquareFreeRequest,
)
from jacobian.contracts.projective_geometry import ProjectiveLineArrangementRequest
from jacobian.contracts.results import ContractModel
from jacobian.exact_domain_checkers import (
    install_exact_domain_checkers as _install_exact_domain_checkers,
)
from jacobian.operation_installation import InstalledDomainBundle
from jacobian.portfolio import build_builtin_portfolio
from jacobian.registry import CheckerRegistry
from jacobian.store import ArtifactStore


def _installed(
    request_models: tuple[type[ContractModel], ...],
    capability_ids: tuple[str, ...],
    *,
    character: str,
) -> InstalledDomainBundle:
    return InstalledDomainBundle(
        adapters=(),
        semantics_uri=_uri(character),
        input_schema_uris={
            model: _uri(str(index + 1)) for index, model in enumerate(request_models)
        },
        result_schema_uris={
            capability_id: _uri(chr(ord("a") + index))
            for index, capability_id in enumerate(capability_ids)
        },
        obligation_schema_uris={},
    )


def install_exact_domain_checkers(
    registry: CheckerRegistry,
    *,
    authorize: bool,
    **installed: InstalledDomainBundle,
):
    domain_ids = {
        "graph": "graph_optimization",
        "graph_invariants": "graph_invariants",
        "graph_symmetry": "graph_symmetry",
    }
    plan = build_builtin_portfolio()
    bundles = {}
    for name, installation in installed.items():
        domain_id = domain_ids.get(name, name)
        bundle = plan.bundle_for(domain_id)
        assert bundle is not None
        bundles[domain_id] = (bundle, installation)
    return _install_exact_domain_checkers(
        registry,
        bundles=bundles,
        authorize=authorize,
    )


def test_installer_authorizes_all_exact_domain_replays(tmp_path: Path) -> None:
    polynomial_ids = (
        "polynomial.jacobian_syzygy.minimum_degree.compute",
        "polynomial.compute.gcd",
        "polynomial.compute.resultant",
        "polynomial.compute.discriminant",
        "polynomial.compute.square_free_decomposition",
    )
    matrix_ids = (
        "matrix.multiply.compute",
        "matrix.normal_form.rref.compute",
        "matrix.nullspace.compute",
        "matrix.characteristic_polynomial.compute",
        "matrix.normal_form.smith.compute",
    )
    graph_ids = (
        "graph.hamiltonian_path.decide",
        "graph.induced_tree.maximum.compute",
        "graph.invariant.diameter.compute",
        "graph.invariant.radius.compute",
        "graph.invariant.maximum_matching.compute",
    )
    number_theory_ids = (
        "integer.compute.prime_factorization",
        "integer.decide.powerful",
        "modular.polynomial_residue_image.compute",
    )
    projective_ids = ("geometry.projective_line_arrangement.flats.materialize",)
    registry = CheckerRegistry(ArtifactStore(tmp_path / "store"))

    installation = install_exact_domain_checkers(
        registry,
        polynomial=_installed(
            (
                GradedJacobianSyzygyRequest,
                PolynomialGcdRequest,
                PolynomialResultantRequest,
                PolynomialDiscriminantRequest,
                PolynomialSquareFreeRequest,
            ),
            polynomial_ids,
            character="e",
        ),
        matrix=_installed(
            (
                RationalMatrixProductRequest,
                RationalMatrixRequest,
                SquareRationalMatrixRequest,
                IntegerMatrixRequest,
            ),
            matrix_ids,
            character="f",
        ),
        graph=_installed(
            (GraphHamiltonianPathRequest, GraphOptimizationRequest),
            graph_ids[:2],
            character="7",
        ),
        graph_invariants=_installed(
            (GraphInvariantRequest,),
            graph_ids[2:],
            character="8",
        ),
        number_theory=_installed(
            (
                FactorizationRequest,
                PowerfulNumberRequest,
                ModularPolynomialResidueImageRequest,
            ),
            number_theory_ids,
            character="6",
        ),
        projective_geometry=_installed(
            (ProjectiveLineArrangementRequest,),
            projective_ids,
            character="9",
        ),
        authorize=True,
    )

    assert set(installation.checker_ids) == set(
        polynomial_ids + matrix_ids + graph_ids + number_theory_ids + projective_ids
    )
    assert all(installation.checker_ids.values())
    for capability_id, checker_id in installation.checker_ids.items():
        assert checker_id is not None
        registration = registry.require_active(checker_id)
        expected_module = (
            "jacobian_checkers.graph_exact_operations:"
            if capability_id.startswith("graph.")
            else (
                "jacobian_checkers.projective_arrangements:"
                if capability_id.startswith("geometry.projective_")
                else (
                    "jacobian_checkers.jacobian_syzygy:"
                    if "jacobian_syzygy" in capability_id
                    else "jacobian_checkers.exact_domain_operations:"
                )
            )
        )
        assert registration.entrypoint.startswith(expected_module)
    graph_runtime = installation.provider_runtimes["finite-graph"]
    assert graph_runtime.provider == "jacobian.graph-exact-checkers"
    assert {
        component["provider"] for component in graph_runtime.configuration["components"]
    } == {"jacobian.graph-exact-checker-source"}
    syzygy_runtime = installation.provider_runtimes["graded-syzygy"]
    assert syzygy_runtime.provider == "jacobian.graded-syzygy-checkers"
    assert {
        component["provider"]
        for component in syzygy_runtime.configuration["components"]
    } == {"jacobian.graded-syzygy-checker-source"}
    projective_runtime = installation.provider_runtimes["projective-arrangement"]
    assert projective_runtime.provider == "jacobian.projective-arrangement-checkers"


def test_installer_preserves_operator_control(tmp_path: Path) -> None:
    registry = CheckerRegistry(ArtifactStore(tmp_path / "store"))

    installation = install_exact_domain_checkers(
        registry,
        polynomial=_installed(
            (
                GradedJacobianSyzygyRequest,
                PolynomialGcdRequest,
                PolynomialResultantRequest,
                PolynomialDiscriminantRequest,
                PolynomialSquareFreeRequest,
            ),
            (
                "polynomial.jacobian_syzygy.minimum_degree.compute",
                "polynomial.compute.gcd",
                "polynomial.compute.resultant",
                "polynomial.compute.discriminant",
                "polynomial.compute.square_free_decomposition",
            ),
            character="e",
        ),
        matrix=_installed(
            (
                RationalMatrixProductRequest,
                RationalMatrixRequest,
                SquareRationalMatrixRequest,
                IntegerMatrixRequest,
            ),
            (
                "matrix.multiply.compute",
                "matrix.normal_form.rref.compute",
                "matrix.nullspace.compute",
                "matrix.characteristic_polynomial.compute",
                "matrix.normal_form.smith.compute",
            ),
            character="f",
        ),
        authorize=False,
    )

    assert set(installation.checker_ids.values()) == {None}


def test_installer_skips_checkers_for_an_unavailable_graph_bundle(
    tmp_path: Path,
) -> None:
    polynomial = _installed(
        (
            GradedJacobianSyzygyRequest,
            PolynomialGcdRequest,
            PolynomialResultantRequest,
            PolynomialDiscriminantRequest,
            PolynomialSquareFreeRequest,
        ),
        (
            "polynomial.jacobian_syzygy.minimum_degree.compute",
            "polynomial.compute.gcd",
            "polynomial.compute.resultant",
            "polynomial.compute.discriminant",
            "polynomial.compute.square_free_decomposition",
        ),
        character="e",
    )
    matrix = _installed(
        (
            RationalMatrixProductRequest,
            RationalMatrixRequest,
            SquareRationalMatrixRequest,
            IntegerMatrixRequest,
        ),
        (
            "matrix.multiply.compute",
            "matrix.normal_form.rref.compute",
            "matrix.nullspace.compute",
            "matrix.characteristic_polynomial.compute",
            "matrix.normal_form.smith.compute",
        ),
        character="f",
    )
    graph = _installed(
        (GraphHamiltonianPathRequest, GraphOptimizationRequest),
        (
            "graph.hamiltonian_path.decide",
            "graph.induced_tree.maximum.compute",
        ),
        character="7",
    )
    graph_invariants = _installed(
        (GraphInvariantRequest,),
        (
            "graph.invariant.diameter.compute",
            "graph.invariant.radius.compute",
            "graph.invariant.maximum_matching.compute",
        ),
        character="8",
    )

    for name, optional_bundles, expected_graph_ids in (
        (
            "optimization-only",
            {"graph": graph},
            {
                "graph.hamiltonian_path.decide",
                "graph.induced_tree.maximum.compute",
            },
        ),
        (
            "invariants-only",
            {"graph_invariants": graph_invariants},
            {
                "graph.invariant.diameter.compute",
                "graph.invariant.radius.compute",
                "graph.invariant.maximum_matching.compute",
            },
        ),
    ):
        registry_root = tmp_path / name
        installation = install_exact_domain_checkers(
            CheckerRegistry(ArtifactStore(registry_root)),
            polynomial=polynomial,
            matrix=matrix,
            authorize=True,
            **optional_bundles,
        )

        graph_ids = {
            capability_id
            for capability_id in installation.checker_ids
            if capability_id.startswith("graph.")
        }
        assert graph_ids == expected_graph_ids
