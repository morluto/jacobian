from pathlib import Path

from tests.support.catalog_build_options import CheckerAuthorityMode
from tests.support.services import open_domain_services

from jacobian.contracts.operations import (
    OperationRequest,
    OperationResult,
)
from jacobian.polynomial_system_operations import (
    install_polynomial_system_operations,
)
from jacobian.polynomial_system_search import PolynomialSystemRationalSearchAdapter


def _request(constant: int) -> OperationRequest:
    return OperationRequest(
        operation_id="polynomial.system.rational_solution.search",
        input={
            "system": {
                "variables": ["x"],
                "equations": [
                    {
                        "terms": [
                            {"coefficient": {"num": "1", "den": "1"}, "exponents": [1]},
                            {
                                "coefficient": {"num": str(constant), "den": "1"},
                                "exponents": [0],
                            },
                        ]
                    }
                ],
            },
            "max_abs_numerator": 1,
            "max_denominator": 1,
        },
    )


def _invoke_search(
    root: Path,
    *,
    checker_authority: CheckerAuthorityMode,
    constant: int,
) -> OperationResult:
    with open_domain_services(
        root,
        checker_authority=checker_authority,
    ) as services:
        checker, installation = install_polynomial_system_operations(
            services.core.store,
            services.core.schemas,
            services.core.artifacts,
            services.installation.verification,
            services.core.checkers,
            authorize_checker=services.installation.authorize_bundled_checkers,
        )
        if checker is not None:
            services.installation.register_operation(checker)
        services.installation.register_operation(
            PolynomialSystemRationalSearchAdapter(
                services.core.artifacts,
                installation,
            )
        )
        return services.core.operations.invoke(_request(constant))


def test_rational_solution_search_returns_first_exact_candidate(tmp_path: Path) -> None:
    result = _invoke_search(
        tmp_path,
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
        constant=1,
    )
    assert result.output["found"] is True
    assert result.output["assignment"] == [{"num": "-1", "den": "1"}]
    assert result.output["examined_assignment_count"] == 1


def test_rational_solution_search_reports_completed_bounded_absence(
    tmp_path: Path,
) -> None:
    result = _invoke_search(
        tmp_path,
        checker_authority=CheckerAuthorityMode.NONE,
        constant=2,
    )
    assert result.output["found"] is False
    assert result.output["examined_assignment_count"] == 3
    assert result.output["grid_assignment_count"] == 3
