import Mathlib
import Lean.DeclarationRange
import Lean.Elab.Command

open Lean

structure DeclarationQuery where
  name_contains : Option String
  type_constants : Array String
  namespace_prefixes : Array String
  kinds : Array String
  limit : Nat
  deriving FromJson

def declarationKind : ConstantInfo → String
  | .axiomInfo _ => "AXIOM"
  | .defnInfo _ => "DEFINITION"
  | .thmInfo _ => "THEOREM"
  | .opaqueInfo _ => "OPAQUE"
  | .quotInfo _ => "QUOTIENT"
  | .inductInfo _ => "INDUCTIVE"
  | .ctorInfo _ => "CONSTRUCTOR"
  | .recInfo _ => "RECURSOR"

def moduleContaining? (env : Environment) (declarationName : Name) : Option Name := do
  let some moduleIndex := env.getModuleIdxFor? declarationName
    | none
  env.allImportedModuleNames[moduleIndex]?

def namespaceMatches (query : DeclarationQuery) (name : Name) : Bool :=
  query.namespace_prefixes.isEmpty ||
    query.namespace_prefixes.any fun namespacePrefix =>
      name.toString == namespacePrefix ||
        name.toString.startsWith (namespacePrefix ++ ".")

def kindMatches (query : DeclarationQuery) (info : ConstantInfo) : Bool :=
  query.kinds.isEmpty || query.kinds.contains (declarationKind info)

def typeMatches (query : DeclarationQuery) (info : ConstantInfo) : Bool :=
  let used := info.type.getUsedConstantsAsSet
  query.type_constants.all fun constant => used.contains constant.toName

def renderedType (info : ConstantInfo) : Elab.Command.CommandElabM String :=
  Elab.Command.liftTermElabM do
    return (← Meta.ppExpr info.type).pretty

def declarationJson (env : Environment) (name : Name) (info : ConstantInfo) :
    Elab.Command.CommandElabM Json := do
  let type ← renderedType info
  let typeTruncated := decide (type.length > 8000)
  let typePreview := if typeTruncated then type.take 8000 else type
  return Json.mkObj [
    ("name", toJson name.toString),
    ("type", toJson typePreview),
    ("type_truncated", toJson typeTruncated),
    ("kind", toJson (declarationKind info)),
    ("module", toJson ((moduleContaining? env name).map toString))
  ]

def readQuery : IO DeclarationQuery := do
  let some path ← IO.getEnv "JACOBIAN_LEAN_QUERY_FILE"
    | throw <| IO.userError "JACOBIAN_LEAN_QUERY_FILE is required"
  let contents ← IO.FS.readFile path
  let json ← match Json.parse contents with
    | .ok json => pure json
    | .error detail => throw <| IO.userError s!"invalid query JSON: {detail}"
  match fromJson? json with
  | .ok query => pure query
  | .error detail => throw <| IO.userError s!"invalid query contract: {detail}"

def executeQuery (env : Environment) (query : DeclarationQuery) :
    Elab.Command.CommandElabM Json := do
  if query.limit == 0 || query.limit > 20 then
    throwError "limit must be between 1 and 20"
  if query.name_contains.isNone && query.type_constants.isEmpty then
    throwError "name_contains or type_constants is required"
  let names := env.constants.toList.toArray.map (·.1) |>.qsort Name.lt
  let mut declarations : Array Json := #[]
  let mut scanned := 0
  let mut stopReason := "EXHAUSTED"
  for name in names do
    if declarations.size == query.limit then
      stopReason := "RESULT_LIMIT"
      break
    if isPrivateName name then continue
    scanned := scanned + 1
    let some info := env.find? name | continue
    if !namespaceMatches query name || !kindMatches query info then continue
    let nameMatches :=
      query.name_contains.map (name.toString.contains ·) |>.getD true
    if !nameMatches || !typeMatches query info then continue
    declarations := declarations.push (← declarationJson env name info)
  return Json.mkObj [
    ("declarations", toJson declarations),
    ("scanned_declarations", toJson scanned),
    ("stop_reason", toJson stopReason)
  ]

run_cmd do
  try
    let query ← readQuery
    let output ← executeQuery (← getEnv) query
    let stdout ← IO.getStdout
    stdout.putStrLn s!"JACOBIAN_LEAN_SEARCH_RESULT {output.compress}"
    stdout.flush
  catch _ =>
    let stdout ← IO.getStdout
    stdout.putStrLn "JACOBIAN_LEAN_SEARCH_ERROR"
    stdout.flush
