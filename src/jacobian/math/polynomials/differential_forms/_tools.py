"""Publication of polynomial differential-form exterior-calculus operations."""

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.differential_forms.operations import (
    affine_homotopy_primitive,
    exterior_derivative,
    interior_product,
    lie_derivative,
    pullback,
    wedge,
)
from jacobian.math.polynomials.differential_forms.values import (
    MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS,
    MAX_DIFFERENTIAL_FORM_EXPONENT,
    MAX_DIFFERENTIAL_FORM_TERMS,
    PolynomialDifferentialForm,
    PolynomialMap,
    PolynomialVectorField,
    PrimitiveResult,
)

_FORM_ENVELOPE = (
    "Each form coefficient admits at most "
    f"{MAX_DIFFERENTIAL_FORM_TERMS} terms, exponents at most "
    f"{MAX_DIFFERENTIAL_FORM_EXPONENT}, and rational components at most "
    f"{MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS} digits."
)


class WedgeRequest(StrictModel):
    left: PolynomialDifferentialForm = Field(
        description="Left polynomial differential form. " + _FORM_ENVELOPE,
    )
    right: PolynomialDifferentialForm = Field(
        description="Right polynomial differential form. " + _FORM_ENVELOPE,
    )


class ExteriorDerivativeRequest(StrictModel):
    form: PolynomialDifferentialForm = Field(
        description="Polynomial differential form. " + _FORM_ENVELOPE,
    )


class ContractRequest(StrictModel):
    field: PolynomialVectorField = Field(
        description="Polynomial vector field sharing the form variable axis.",
    )
    form: PolynomialDifferentialForm = Field(
        description="Polynomial differential form. " + _FORM_ENVELOPE,
    )


class PullbackRequest(StrictModel):
    mapping: PolynomialMap = Field(
        description="Polynomial map whose target axis matches the form axis.",
    )
    form: PolynomialDifferentialForm = Field(
        description="Polynomial differential form on the map target axis. "
        + _FORM_ENVELOPE,
    )


class LieDerivativeRequest(StrictModel):
    field: PolynomialVectorField = Field(
        description="Polynomial vector field sharing the form variable axis.",
    )
    form: PolynomialDifferentialForm = Field(
        description="Polynomial differential form. " + _FORM_ENVELOPE,
    )


class PrimitiveRequest(StrictModel):
    form: PolynomialDifferentialForm = Field(
        description="Closed polynomial differential form of positive degree. "
        + _FORM_ENVELOPE,
    )


FormTermPayload = tuple[tuple[int, int], tuple[int, ...]]
FormComponentPayload = tuple[tuple[int, ...], list[FormTermPayload]]


def _form_payload(
    variables: tuple[str, ...], degree: int, *components: FormComponentPayload
) -> dict[str, object]:
    return {
        "variables": list(variables),
        "degree": str(degree),
        "components": [
            {
                "indices": list(indices),
                "coefficient": {
                    "variables": list(variables),
                    "polynomial": {
                        "terms": [
                            {
                                "coefficient": {"num": str(num), "den": str(den)},
                                "exponents": list(exponents),
                            }
                            for (num, den), exponents in terms
                        ]
                    },
                },
            }
            for indices, terms in components
        ],
    }


def _field_payload(
    variables: tuple[str, ...], *components: list[FormTermPayload]
) -> dict[str, object]:
    return {
        "variables": list(variables),
        "components": [
            {
                "variables": list(variables),
                "polynomial": {
                    "terms": [
                        {
                            "coefficient": {"num": str(num), "den": str(den)},
                            "exponents": list(exponents),
                        }
                        for (num, den), exponents in terms
                    ]
                },
            }
            for terms in components
        ],
    }


_X_Y = ("x", "y")
_DX = _form_payload(_X_Y, 1, ((0,), [((1, 1), (1, 0))]))
_DY = _form_payload(_X_Y, 1, ((1,), [((1, 1), (0, 1))]))
_X_DY = _form_payload(_X_Y, 1, ((1,), [((1, 1), (1, 0))]))
_DX_WEDGE_DY = _form_payload(_X_Y, 2, ((0, 1), [((1, 1), (0, 0))]))
_PARTIAL_X = _field_payload(_X_Y, [((1, 1), (1, 0))], [])


TOOLS = (
    MathTool(
        operation_id="differential_form.wedge.compute",
        title="Compute the exact wedge product of polynomial differential forms",
        description=(
            "Compute the sparse polynomial form alpha wedge beta using the "
            "increasing differential-index basis and exact permutation signs. "
            "Both forms must use the same ordered variable axis; repeated "
            "differentials vanish and a degree above the ambient dimension "
            "returns a degree-labelled canonical zero form so the graded "
            "target and subsequent compositions remain deterministic. " + _FORM_ENVELOPE
        ),
        request_type=WedgeRequest,
        result_type=PolynomialDifferentialForm,
        run=lambda request: wedge(request.left, request.right),
        tags=("polynomial", "differential-form", "wedge", "exterior-algebra", "exact"),
        examples=(
            OperationExample(
                name="dx_wedge_dy",
                description=(
                    "Compute (x dx) wedge (y dy) = xy dx wedge dy; both forms "
                    "must use the same ordered QQ variable axis."
                ),
                input={
                    "left": {
                        "variables": ["x", "y"],
                        "degree": "1",
                        "components": [
                            {
                                "indices": [0],
                                "coefficient": {
                                    "variables": ["x", "y"],
                                    "polynomial": {
                                        "terms": [
                                            {
                                                "coefficient": {"num": "1", "den": "1"},
                                                "exponents": [1, 0],
                                            }
                                        ]
                                    },
                                },
                            }
                        ],
                    },
                    "right": {
                        "variables": ["x", "y"],
                        "degree": "1",
                        "components": [
                            {
                                "indices": [1],
                                "coefficient": {
                                    "variables": ["x", "y"],
                                    "polynomial": {
                                        "terms": [
                                            {
                                                "coefficient": {"num": "1", "den": "1"},
                                                "exponents": [0, 1],
                                            }
                                        ]
                                    },
                                },
                            }
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="differential_form.exterior_derivative.compute",
        title="Compute the exact exterior derivative of a polynomial form",
        description=(
            "Differentiate every coefficient and wedge with its coordinate "
            "differential, so d(f dx_I) sums signed partials; a result above "
            "the ambient dimension is the degree-labelled canonical zero. "
            + _FORM_ENVELOPE
        ),
        request_type=ExteriorDerivativeRequest,
        result_type=PolynomialDifferentialForm,
        run=lambda request: exterior_derivative(request.form),
        tags=(
            "polynomial",
            "differential-form",
            "exterior-derivative",
            "exterior-algebra",
            "exact",
        ),
        examples=(
            OperationExample(
                name="d_of_x_dy",
                description=(
                    "Compute d(x dy) = dx wedge dy; the form must use an ordered QQ variable axis."
                ),
                input={"form": _X_DY},
            ),
        ),
    ),
    MathTool(
        operation_id="differential_form.contract.compute",
        title="Contract a polynomial form with a polynomial vector field",
        description=(
            "Compute the interior product inserting the field into the first "
            "differential slot with alternating signs; the field must share "
            "the form variable axis and contraction lowers the degree by one."
        ),
        request_type=ContractRequest,
        result_type=PolynomialDifferentialForm,
        run=lambda request: interior_product(request.field, request.form),
        tags=(
            "polynomial",
            "differential-form",
            "contraction",
            "exterior-algebra",
            "exact",
        ),
        examples=(
            OperationExample(
                name="contract_dx_wedge_dy",
                description=(
                    "Contract dx wedge dy with partial_x to get dy; the field must share the form axis."
                ),
                input={"field": _PARTIAL_X, "form": _DX_WEDGE_DY},
            ),
        ),
    ),
    MathTool(
        operation_id="differential_form.pullback.compute",
        title="Pull a polynomial form back along a polynomial map",
        description=(
            "Substitute the map images into every coefficient and expand the "
            "image differentials; the form must use the map target axis and "
            "the result lives on the source axis."
        ),
        request_type=PullbackRequest,
        result_type=PolynomialDifferentialForm,
        run=lambda request: pullback(request.mapping, request.form),
        tags=(
            "polynomial",
            "differential-form",
            "pullback",
            "exterior-algebra",
            "exact",
        ),
        examples=(
            OperationExample(
                name="pullback_dx_wedge_dy",
                description=(
                    "Pull dx wedge dy back along t -> (t, t) to get zero; the form must use the map target axis."
                ),
                input={
                    "mapping": {
                        "source_variables": ["t"],
                        "target_variables": ["x", "y"],
                        "images": [
                            {
                                "variables": ["t"],
                                "polynomial": {
                                    "terms": [
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [1],
                                        }
                                    ]
                                },
                            },
                            {
                                "variables": ["t"],
                                "polynomial": {
                                    "terms": [
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [1],
                                        }
                                    ]
                                },
                            },
                        ],
                    },
                    "form": _DX_WEDGE_DY,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="differential_form.lie_derivative.compute",
        title="Compute the Lie derivative of a polynomial form",
        description=(
            "Evaluate Cartan's formula L = interior d + d interior exactly; "
            "the field must share the form variable axis and the result keeps "
            "the source degree."
        ),
        request_type=LieDerivativeRequest,
        result_type=PolynomialDifferentialForm,
        run=lambda request: lie_derivative(request.field, request.form),
        tags=(
            "polynomial",
            "differential-form",
            "lie-derivative",
            "exterior-algebra",
            "exact",
        ),
        examples=(
            OperationExample(
                name="lie_of_x_dx",
                description=(
                    "Differentiate x dx along partial_x to get dx; the field must share the form axis."
                ),
                input={"field": _PARTIAL_X, "form": _DX},
            ),
        ),
    ),
    MathTool(
        operation_id="differential_form.primitive.compute",
        title="Compute an affine-homotopy primitive of a closed polynomial form",
        description=(
            "Return the cone-homotopy primitive of a closed positive-degree "
            "form, or NOT_APPLICABLE for a non-closed or degree-zero source; "
            "a constructed primitive differentiates back to its source."
        ),
        request_type=PrimitiveRequest,
        result_type=PrimitiveResult,
        run=lambda request: affine_homotopy_primitive(request.form),
        tags=(
            "polynomial",
            "differential-form",
            "primitive",
            "exterior-algebra",
            "exact",
        ),
        examples=(
            OperationExample(
                name="primitive_of_dx_wedge_dy",
                description=(
                    "Find a primitive of the closed form dx wedge dy; the source must be closed of positive degree."
                ),
                input={"form": _DX_WEDGE_DY},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
