import REPL.Snapshots

open Lean REPL

structure ProofStateBounds where
  pickle_path : String
  request_id : String
  max_goals : Nat
  max_local_declarations : Nat
  max_rendered_bytes : Nat
  /-- "typed_goals" emits the existing typed-goal payload.
      "metavariable_fields" emits structured metavariable, local-instance,
      and elaboration-context fields. -/
  mode : String
  deriving FromJson

def readBounds : IO ProofStateBounds := do
  let some path ← IO.getEnv "JACOBIAN_LEAN_PROOF_STATE_QUERY"
    | throw <| IO.userError "JACOBIAN_LEAN_PROOF_STATE_QUERY is required"
  let contents ← IO.FS.readFile path
  let json ← match Json.parse contents with
    | .ok value => pure value
    | .error detail => throw <| IO.userError detail
  match fromJson? json with
  | .ok bounds => pure bounds
  | .error detail => throw <| IO.userError detail

def binderInfoName : BinderInfo → String
  | .default => "DEFAULT"
  | .implicit => "IMPLICIT"
  | .strictImplicit => "STRICT_IMPLICIT"
  | .instImplicit => "INSTANCE_IMPLICIT"

def metavarKindName : MetavarKind → String
  | .natural => "NATURAL"
  | .synthetic => "SYNTHETIC"
  | .syntheticOpaque => "SYNTHETIC_OPAQUE"

def renderedExpr (expr : Expr) : MetaM String := do
  return (← Meta.ppExpr (← instantiateMVars expr)).pretty

def localDeclarationJson (decl : LocalDecl) : MetaM Json := do
  let type ← renderedExpr decl.type
  let value ← decl.value?.mapM renderedExpr
  return Json.mkObj [
    ("user_name", toJson decl.userName.toString),
    ("binder_info", toJson (binderInfoName decl.binderInfo)),
    ("type", toJson type),
    ("value", toJson value)
  ]

def goalJson (goal : MVarId) (index maxLocals : Nat) : MetaM Json :=
  goal.withContext do
    let target ← renderedExpr (← goal.getType)
    let mut locals : Array Json := #[]
    for decl in ← getLCtx do
      if decl.isImplementationDetail then continue
      if locals.size == maxLocals then
        throwError "LEAN_PROOF_STATE_LOCAL_LIMIT"
      locals := locals.push (← localDeclarationJson decl)
    return Json.mkObj [
      ("goal_index", toJson index),
      ("target_type", toJson target),
      ("local_declarations", toJson locals)
    ]

/-- Render one `LocalInstance` using the goal's local context.

    `LocalInstance` stores only the class name and the instance free
    variable; the variable's user name and type are recovered from the
    local context via `LocalContext.find?`, the maintained accessor. -/
def localInstanceJson (inst : LocalInstance) : MetaM Json := do
  let lctx ← getLCtx
  match lctx.find? inst.fvar.fvarId! with
  | some decl =>
    let fvarType ← renderedExpr decl.type
    return Json.mkObj [
      ("class_name", toJson inst.className.toString),
      ("fvar_user_name", toJson decl.userName.toString),
      ("fvar_type", toJson fvarType)
    ]
  | none =>
    -- The instance free variable is not in the current local context.
    -- Report the class name honestly and mark the variable fields absent.
    return Json.mkObj [
      ("class_name", toJson inst.className.toString),
      ("fvar_user_name", toJson ""),
      ("fvar_type", toJson "")
    ]

/-- Structured metavariable fields from `MetavarDecl` and assignment status.

    Uses `MVarId.getDecl`, `MVarId.isAssigned`, and
    `MVarId.isDelayedAssigned` from the maintained `Lean.Meta` API. -/
def structuredMetavariableJson (goal : MVarId) (index maxLocals : Nat) : MetaM Json := do
  goal.withContext do
    let decl ← goal.getDecl
    let isAssigned ← goal.isAssigned
    let isDelayedAssigned ← goal.isDelayedAssigned
    let target ← renderedExpr (← goal.getType)
    let mut insts : Array Json := #[]
    for inst in decl.localInstances do
      if insts.size == maxLocals then
        throwError "LEAN_PROOF_STATE_LOCAL_LIMIT"
      insts := insts.push (← localInstanceJson inst)
    return Json.mkObj [
      ("goal_index", toJson index),
      ("user_name", toJson decl.userName.toString),
      ("is_user_name_anonymous", toJson decl.userName.isAnonymous),
      ("kind", toJson (metavarKindName decl.kind)),
      ("is_assigned", toJson isAssigned),
      ("is_delayed_assigned", toJson isDelayedAssigned),
      ("depth", toJson decl.depth),
      ("num_scope_args", toJson decl.numScopeArgs),
      ("target_type", toJson target),
      ("local_instances", toJson insts)
    ]

/-- Elaboration context fields from the pickled `Term.Context`.

    Only fields that survive pickling as plain values are reported; the
    closure and reference fields (`auxDeclToFullName`, `autoBoundImplicits`,
    `tacticCache?`) are intentionally omitted because they are not
    inspectable across the pickle boundary. -/
def elaborationContextJson (snapshot : ProofSnapshot) : Json :=
  let declName := match snapshot.termContext.declName? with
    | some n => n.toString
    | none => ""
  Json.mkObj [
    ("decl_name", toJson declName),
    ("may_postpone", toJson snapshot.termContext.mayPostpone),
    ("err_to_sorry", toJson snapshot.termContext.errToSorry),
    ("auto_bound_implicit", toJson snapshot.termContext.autoBoundImplicitContext.isSome),
    ("implicit_lambda", toJson snapshot.termContext.implicitLambda),
    ("is_noncomputable_section", toJson snapshot.termContext.isNoncomputableSection),
    ("ignore_tc_failures", toJson snapshot.termContext.ignoreTCFailures),
    ("in_pattern", toJson snapshot.termContext.inPattern),
    ("save_rec_app_syntax", toJson snapshot.termContext.saveRecAppSyntax),
    ("holes_as_synthetic_opaque", toJson snapshot.termContext.holesAsSyntheticOpaque)
  ]

def resultEnvelope (requestId : String) (payload : Json) : Json :=
  Json.mkObj [
    ("request_id", toJson requestId),
    ("payload", payload)
  ]

def errorEnvelope (requestId code message : String) : Json :=
  Json.mkObj [
    ("request_id", toJson requestId),
    ("code", toJson code),
    ("message", toJson message)
  ]

def emit (marker : String) (payload : Json) : IO Unit := do
  let stdout ← IO.getStdout
  stdout.putStrLn s!"{marker} {payload.compress}"
  stdout.flush

def validateBounds (bounds : ProofStateBounds) : IO Unit := do
  if bounds.max_goals == 0 || bounds.max_goals > 64 ||
      bounds.max_local_declarations == 0 ||
      bounds.max_local_declarations > 256 ||
      bounds.max_rendered_bytes < 1024 ||
      bounds.max_rendered_bytes > 262144 then
    throw <| IO.userError "LEAN_PROOF_STATE_INVALID_BOUNDS"

def loadSnapshot (bounds : ProofStateBounds) : IO ProofSnapshot := do
  -- This helper is one-shot. Let process exit release the compacted region:
  -- freeing it while multiple goals share local declarations can invalidate
  -- expressions before their JSON rendering is emitted.
  let (snapshot, _) ← ProofSnapshot.unpickle bounds.pickle_path none
  if snapshot.tacticState.goals.length > bounds.max_goals then
    throw <| IO.userError "LEAN_PROOF_STATE_GOAL_LIMIT"
  return snapshot

def runTypedGoalsQuery (bounds : ProofStateBounds) (snapshot : ProofSnapshot) : IO Json := do
  let (goals, _) ← snapshot.runMetaM do
    snapshot.tacticState.goals.toArray.mapIdxM fun index goal =>
      goalJson goal index bounds.max_local_declarations
  let payload := Json.mkObj [
    ("expression_serialization", "LEAN_PRETTY_PRINTED_EXPR"),
    ("typed_goals", toJson goals)
  ]
  if payload.compress.toUTF8.size > bounds.max_rendered_bytes then
    throw <| IO.userError "LEAN_PROOF_STATE_OUTPUT_LIMIT"
  return payload

/-- Structured metavariable, local-instance, and elaboration-context query.

    Coercion provenance is intentionally reported as `UNAVAILABLE`: the
    maintained `Lean.Meta.Coe` APIs (`expandCoe`, `getCoeFnInfo?`) operate
    on expressions during elaboration and do not retain a per-metavariable
    coercion log on a pickled proof state. Inferring coercions by parsing
    pretty-printed output is forbidden, so the honest contract reports the
    limitation rather than fabricating provenance. -/
def runMetavariableFieldsQuery (bounds : ProofStateBounds) (snapshot : ProofSnapshot) : IO Json := do
  let (mvars, _) ← snapshot.runMetaM do
    snapshot.tacticState.goals.toArray.mapIdxM fun index goal =>
      structuredMetavariableJson goal index bounds.max_local_declarations
  let elabContext := elaborationContextJson snapshot
  let payload := Json.mkObj [
    ("expression_serialization", "LEAN_PRETTY_PRINTED_EXPR"),
    ("structured_metavariables", toJson mvars),
    ("elaboration_context", toJson elabContext),
    ("coercion_provenance", "UNAVAILABLE"),
    ("coercion_provenance_basis",
      "maintained Lean.Meta.Coe APIs operate on expressions during elaboration; \
       a pickled proof state retains no per-metavariable coercion log")
  ]
  if payload.compress.toUTF8.size > bounds.max_rendered_bytes then
    throw <| IO.userError "LEAN_PROOF_STATE_OUTPUT_LIMIT"
  return payload

unsafe def runQuery (bounds : ProofStateBounds) : IO Json := do
  validateBounds bounds
  let snapshot ← loadSnapshot bounds
  match bounds.mode with
  | "typed_goals" => runTypedGoalsQuery bounds snapshot
  | "metavariable_fields" => runMetavariableFieldsQuery bounds snapshot
  | _ => throw <| IO.userError "LEAN_PROOF_STATE_UNKNOWN_MODE"

unsafe def main : IO Unit := do
  initSearchPath (← Lean.findSysroot)
  let bounds ← readBounds
  try
    emit "JACOBIAN_PROOF_STATE_RESULT" <|
      resultEnvelope bounds.request_id (← runQuery bounds)
  catch error =>
    let message := toString error
    let code :=
      if message.contains "LEAN_PROOF_STATE_GOAL_LIMIT" then
        "LEAN_PROOF_STATE_GOAL_LIMIT"
      else if message.contains "LEAN_PROOF_STATE_LOCAL_LIMIT" then
        "LEAN_PROOF_STATE_LOCAL_LIMIT"
      else if message.contains "LEAN_PROOF_STATE_OUTPUT_LIMIT" then
        "LEAN_PROOF_STATE_OUTPUT_LIMIT"
      else if message.contains "LEAN_PROOF_STATE_UNKNOWN_MODE" then
        "LEAN_PROOF_STATE_UNKNOWN_MODE"
      else
        "LEAN_PROOF_STATE_QUERY_FAILED"
    emit "JACOBIAN_PROOF_STATE_ERROR" <|
      errorEnvelope bounds.request_id code "typed proof-state extraction failed"
