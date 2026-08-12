import {{JACOBIAN_IMPORT}}
import Lean.DeclarationRange
import Lean.DocString
import Lean.Elab.Command

open Lean

structure DeclarationQuery where
  request_id : String
  operation : String
  declaration_name : Option String
  name_contains : Option String
  type_constants : Array String
  namespace_prefixes : Array String
  target_module_prefixes : Array String
  kinds : Array String
  limit : Nat
  candidate_names : Array String
  candidate_scan_positions : Array Nat
  scanned_declarations_total : Option Nat
  max_depth : Nat
  max_nodes : Nat
  deriving FromJson

structure DeclarationIndexEntry where
  name : Name
  module_name : String
  kind : String

def declarationKind : ConstantInfo → String
  | .axiomInfo _ => "AXIOM"
  | .defnInfo _ => "DEFINITION"
  | .thmInfo _ => "THEOREM"
  | .opaqueInfo _ => "OPAQUE"
  | .quotInfo _ => "QUOTIENT"
  | .inductInfo _ => "INDUCTIVE"
  | .ctorInfo _ => "CONSTRUCTOR"
  | .recInfo _ => "RECURSOR"

def moduleContaining? (env : Environment) (declName : Name) : Option Name := do
  let some moduleIdx := env.getModuleIdxFor? declName
    | none
  env.allImportedModuleNames[moduleIdx]?

def declarationNamespace (name : Name) : Option String :=
  let ns := name.getPrefix
  if ns.isAnonymous then none else some ns.toString

def sourceJson (env : Environment) (name : Name) :
    Elab.Command.CommandElabM Json := do
  let ranges ← findDeclarationRanges? name
  return match ranges with
    | none => Json.null
    | some ranges =>
      Json.mkObj [
        ("module", toJson ((moduleContaining? env name).map toString)),
        ("line", toJson ranges.selectionRange.pos.line),
        ("column", toJson ranges.selectionRange.pos.column),
        ("end_line", toJson ranges.selectionRange.endPos.line),
        ("end_column", toJson ranges.selectionRange.endPos.column)
      ]

def renderedType (info : ConstantInfo) :
    Elab.Command.CommandElabM String :=
  Elab.Command.liftTermElabM do
    return (← Meta.ppExpr info.type).pretty

def namespaceMatches (query : DeclarationQuery) (name : Name) : Bool :=
  query.namespace_prefixes.isEmpty ||
    query.namespace_prefixes.any fun nsPrefix =>
      name.toString == nsPrefix || name.toString.startsWith (nsPrefix ++ ".")

def stringPrefixMatches (prefixes : Array String) (value : String) : Bool :=
  prefixes.isEmpty || prefixes.any fun itemPrefix =>
    value == itemPrefix || value.startsWith (itemPrefix ++ ".")

def kindMatches (query : DeclarationQuery) (info : ConstantInfo) : Bool :=
  query.kinds.isEmpty || query.kinds.contains (declarationKind info)

def targetModuleMatches (query : DeclarationQuery) (env : Environment)
    (name : Name) : Bool :=
  match moduleContaining? env name with
  | none => false
  | some modName =>
    query.target_module_prefixes.isEmpty ||
      query.target_module_prefixes.any fun modulePrefix =>
        modName.toString == modulePrefix ||
          modName.toString.startsWith (modulePrefix ++ ".")

def typeMatches (query : DeclarationQuery) (info : ConstantInfo) : Bool :=
  let used := info.type.getUsedConstantsAsSet
  query.type_constants.all fun constant => used.contains constant.toName

def directDependencies (env : Environment) (info : ConstantInfo) :
    Array (Name × Bool × Bool) :=
  Id.run do
    let typeRefs := info.type.getUsedConstantsAsSet
    let valueRefs :=
      match info.value? (allowOpaque := true) with
      | some value => value.getUsedConstantsAsSet
      | none => {}
    let names := (typeRefs.toArray ++ valueRefs.toArray).qsort Name.lt
    let mut dependencies := #[]
    let mut previous : Option Name := none
    for name in names do
      if previous == some name then continue
      previous := some name
      if env.contains name then
        dependencies := dependencies.push (
          name,
          typeRefs.contains name,
          valueRefs.contains name
        )
    return dependencies

def dependencyNodeJson (name : Name) (info : ConstantInfo) (depth : Nat) : Json :=
  Json.mkObj [
    ("name", toJson name.toString),
    ("kind", toJson (declarationKind info)),
    ("depth", toJson depth)
  ]

def dependencyEdgeJson (source target : Name) (inType inValue : Bool) : Json :=
  Id.run do
    let mut kinds : Array String := #[]
    if inType then kinds := kinds.push "TYPE"
    if inValue then kinds := kinds.push "VALUE"
    return Json.mkObj [
      ("source", toJson source.toString),
      ("target", toJson target.toString),
      ("kinds", toJson kinds)
    ]

def declarationJson (env : Environment) (name : Name) (info : ConstantInfo)
    (type : String) (matchReasons : Array String) (includeDetails : Bool) :
    Elab.Command.CommandElabM Json := do
  let source ← sourceJson env name
  let docString ←
    if includeDetails then findDocString? env name else pure none
  return Json.mkObj [
    ("name", toJson name.toString),
    ("type", toJson type),
    ("kind", toJson (declarationKind info)),
    ("namespace", toJson (declarationNamespace name)),
    ("docstring", toJson docString),
    ("source", source),
    ("match_reasons", toJson matchReasons)
  ]

def parseQuery (contents : String) : Except String DeclarationQuery := do
  let json ← match Json.parse contents with
    | .ok json => pure json
    | .error detail => throw s!"invalid query JSON: {detail}"
  match fromJson? json with
  | .ok query => pure query
  | .error detail => throw s!"invalid query contract: {detail}"

def resultEnvelope (query : DeclarationQuery) (payload : Json) : Json :=
  Json.mkObj [
    ("request_id", toJson query.request_id),
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

def executeQuery (env : Environment) (names : Array Name)
    (query : DeclarationQuery) :
    Elab.Command.CommandElabM (Except (String × String) Json) := do
  if query.limit == 0 || query.limit > 50 then
    return .error ("LEAN_QUERY_FAILED", "limit must be between 1 and 50")
  if query.max_depth > 8 || query.max_nodes == 0 || query.max_nodes > 500 then
    return .error ("LEAN_QUERY_FAILED", "dependency budgets are invalid")
  if query.operation == "dependencies" then
      let some rawName := query.declaration_name
        | return .error ("LEAN_QUERY_FAILED", "declaration_name is required")
      let root := rawName.toName
      let some rootInfo := env.find? root
        | return .error (
            "LEAN_DECLARATION_NOT_FOUND",
            s!"declaration not found: {rawName}"
          )
      if root.toString != rawName || !targetModuleMatches query env root then
        return .error (
          "LEAN_DECLARATION_NOT_FOUND",
          s!"declaration not found: {rawName}"
        )
      let mut queue : Array (Name × Nat) := #[(root, 0)]
      let mut cursor := 0
      let mut visited : Array Name := #[root]
      let mut nodes : Array Json := #[dependencyNodeJson root rootInfo 0]
      let mut edges : Array Json := #[]
      let mut frontier : Array String := #[]
      let mut nodeBudgetExhausted := false
      while cursor < queue.size do
        let (source, depth) := queue[cursor]!
        cursor := cursor + 1
        let some info := env.find? source | continue
        let dependencies := directDependencies env info
        if depth == query.max_depth then
          if !dependencies.isEmpty then
            frontier := frontier.push source.toString
          continue
        for (target, inType, inValue) in dependencies do
          if target == source then continue
          if !visited.contains target then
            if visited.size == query.max_nodes then
              nodeBudgetExhausted := true
              if !frontier.contains source.toString then
                frontier := frontier.push source.toString
              continue
            let some targetInfo := env.find? target | continue
            visited := visited.push target
            nodes := nodes.push (dependencyNodeJson target targetInfo (depth + 1))
            queue := queue.push (target, depth + 1)
          if visited.contains target then
            edges := edges.push (
              dependencyEdgeJson source target inType inValue
            )
      return .ok <| Json.mkObj [
        ("operation", "dependencies"),
        ("nodes", toJson nodes),
        ("edges", toJson edges),
        ("frontier", toJson frontier),
        ("node_budget_exhausted", toJson nodeBudgetExhausted),
        ("closure_complete", toJson frontier.isEmpty)
      ]
  else if query.operation == "inspect" then
      let some rawName := query.declaration_name
        | return .error ("LEAN_QUERY_FAILED", "declaration_name is required")
      let name := rawName.toName
      let some info := env.find? name
        | return .error (
            "LEAN_DECLARATION_NOT_FOUND",
            s!"declaration not found: {rawName}"
          )
      if name.toString != rawName || !targetModuleMatches query env name then
        return .error (
          "LEAN_DECLARATION_NOT_FOUND",
          s!"declaration not found: {rawName}"
        )
      let type ← renderedType info
      let declaration ← declarationJson env name info type #[] true
      return .ok <| Json.mkObj [
        ("operation", "inspect"),
        ("declaration", declaration)
      ]
  else if query.operation == "search" then
      if query.name_contains.isNone && query.type_constants.isEmpty then
        return .error (
          "LEAN_QUERY_FAILED",
          "name_contains or type_constants is required"
        )
      match query.scanned_declarations_total with
      | some scannedTotal =>
        if query.candidate_names.size != query.candidate_scan_positions.size then
          return .error ("LEAN_QUERY_FAILED", "candidate catalog is inconsistent")
        let mut results : Array Json := #[]
        let mut scanned := 0
        let mut stopReason := "EXHAUSTED"
        let mut previousPosition := 0
        for index in [:query.candidate_names.size] do
          let position := query.candidate_scan_positions[index]!
          if position == 0 || position <= previousPosition || position > scannedTotal then
            return .error ("LEAN_QUERY_FAILED", "candidate positions are invalid")
          previousPosition := position
          scanned := position
          let rawName := query.candidate_names[index]!
          let name := rawName.toName
          if name.toString != rawName then continue
          let some info := env.find? name | continue
          if isPrivateName name || !targetModuleMatches query env name then continue
          if !namespaceMatches query name || !kindMatches query info then continue
          let nameMatched :=
            query.name_contains.map (name.toString.contains ·) |>.getD true
          if !nameMatched || !typeMatches query info then continue
          let type ← renderedType info
          let mut reasons : Array String := #[]
          if query.name_contains.isSome then
            reasons := reasons.push "NAME_SUBSTRING"
          if !query.type_constants.isEmpty then
            reasons := reasons.push "TYPE_CONSTANTS"
          results := results.push (
            ← declarationJson env name info type reasons false
          )
          if results.size == query.limit then
            if position < scannedTotal then
              stopReason := "RESULT_LIMIT"
            break
        if results.size < query.limit then
          scanned := scannedTotal
        return .ok <| Json.mkObj [
          ("operation", "search"),
          ("declarations", toJson results),
          ("scanned_declarations", scanned),
          ("stop_reason", stopReason)
        ]
      | none =>
        let mut results : Array Json := #[]
        let mut scanned := 0
        let mut stopReason := "EXHAUSTED"
        for name in names do
          if results.size == query.limit then
            stopReason := "RESULT_LIMIT"
            break
          if isPrivateName name || !targetModuleMatches query env name then continue
          scanned := scanned + 1
          let some info := env.find? name | continue
          if !namespaceMatches query name || !kindMatches query info then continue
          let nameMatched :=
            query.name_contains.map (name.toString.contains ·) |>.getD true
          if !nameMatched || !typeMatches query info then continue
          let type ← renderedType info
          let mut reasons : Array String := #[]
          if query.name_contains.isSome then
            reasons := reasons.push "NAME_SUBSTRING"
          if !query.type_constants.isEmpty then
            reasons := reasons.push "TYPE_CONSTANTS"
          results := results.push (
            ← declarationJson env name info type reasons false
          )
        return .ok <| Json.mkObj [
          ("operation", "search"),
          ("declarations", toJson results),
          ("scanned_declarations", scanned),
          ("stop_reason", stopReason)
        ]
  else
    return .error (
      "LEAN_QUERY_FAILED",
      "operation must be search, inspect, or dependencies"
    )

def readQuery : IO DeclarationQuery := do
  let some path ← IO.getEnv "JACOBIAN_LEAN_QUERY_FILE"
    | throw <| IO.userError "JACOBIAN_LEAN_QUERY_FILE is required"
  let contents ← IO.FS.readFile path
  match parseQuery contents with
  | .ok query => pure query
  | .error detail => throw <| IO.userError detail

def importedIndexEntries (env : Environment) : Array DeclarationIndexEntry :=
  Id.run do
    let mut metadata : NameMap (String × String) := {}
    let moduleNames := env.header.moduleNames
    for h : index in [:env.header.moduleData.size] do
      let data := env.header.moduleData[index]
      let some moduleName := moduleNames[index]?
        | continue
      for name in data.constNames ++ data.extraConstNames do
        if isPrivateName name || metadata.contains name then continue
        let some info := env.find? name | continue
        metadata := metadata.insert name (
          moduleName.toString,
          declarationKind info
        )
    let names := env.constants.toList.toArray.map (·.1) |>.qsort Name.lt
    let mut entries : Array DeclarationIndexEntry := #[]
    for name in names do
      if isPrivateName name then continue
      let some (moduleName, kind) := metadata.find? name | continue
      entries := entries.push { name, module_name := moduleName, kind }
    return entries

def catalogQuery? (query : DeclarationQuery)
    (entries : Array DeclarationIndexEntry) : Option DeclarationQuery :=
  Id.run do
    let some nameContains := query.name_contains | return none
    let mut candidates : Array String := #[]
    let mut positions : Array Nat := #[]
    let mut scanned := 0
    for entry in entries do
      if !stringPrefixMatches query.target_module_prefixes entry.module_name then
        continue
      scanned := scanned + 1
      if !namespaceMatches query entry.name ||
          (!query.kinds.isEmpty && !query.kinds.contains entry.kind) ||
          !entry.name.toString.contains nameContains then
        continue
      if !query.type_constants.isEmpty || candidates.size < query.limit then
        candidates := candidates.push entry.name.toString
        positions := positions.push scanned
        if candidates.size > 10000 then return none
    return some {
      query with
      candidate_names := candidates
      candidate_scan_positions := positions
      scanned_declarations_total := some scanned
    }

def readOrBuildEntries (env : Environment) :
    IO (Array DeclarationIndexEntry) := do
  let some rawIndexPath ← IO.getEnv "JACOBIAN_LEAN_INDEX_FILE"
    | throw <| IO.userError "JACOBIAN_LEAN_INDEX_FILE is required"
  let some environmentDigest ← IO.getEnv "JACOBIAN_LEAN_ENVIRONMENT_DIGEST"
    | throw <| IO.userError "JACOBIAN_LEAN_ENVIRONMENT_DIGEST is required"
  let indexPath := System.FilePath.mk rawIndexPath
  let entries := importedIndexEntries env
  if ← indexPath.pathExists then
    return entries
  let rows := entries.toList.map fun entry =>
    s!"{entry.name}\t{entry.module_name}\t{entry.kind}"
  let serialized := String.intercalate "\n" (environmentDigest :: rows) ++ "\n"
  let temporaryPath := System.FilePath.mk (rawIndexPath ++ ".tmp")
  if ← temporaryPath.pathExists then
    IO.FS.removeFile temporaryPath
  IO.FS.writeFile temporaryPath serialized
  IO.FS.rename temporaryPath indexPath
  return entries

elab "#jacobian_declaration_query" : command => do
  let mut query ← readQuery
  let env ← getEnv
  let mut names : Array Name := #[]
  if query.operation == "search" &&
      query.scanned_declarations_total.isNone then
    let entries ← readOrBuildEntries env
    match catalogQuery? query entries with
    | some catalogQuery => query := catalogQuery
    | none => names := entries.map (·.name)
  try
    match ← executeQuery env names query with
    | .ok output =>
      logInfo m!"JACOBIAN_DECLARATION_RESULT {(resultEnvelope query output).compress}"
    | .error (code, message) =>
      logInfo m!"JACOBIAN_DECLARATION_ERROR {(errorEnvelope query.request_id code message).compress}"
  catch _ =>
    logInfo m!"JACOBIAN_DECLARATION_ERROR {(errorEnvelope query.request_id "LEAN_QUERY_FAILED" "query execution failed").compress}"

-- JACOBIAN_DECLARATION_ENTRYPOINT
run_cmd do
  let mut query ← readQuery
  let env ← getEnv
  let mut names : Array Name := #[]
  if query.operation == "search" &&
      query.scanned_declarations_total.isNone then
    let entries ← readOrBuildEntries env
    match catalogQuery? query entries with
    | some catalogQuery => query := catalogQuery
    | none => names := entries.map (·.name)
  try
    match ← executeQuery env names query with
    | .ok output =>
      emit "JACOBIAN_DECLARATION_RESULT" <| resultEnvelope query output
    | .error (code, message) =>
      emit "JACOBIAN_DECLARATION_ERROR" <|
        errorEnvelope query.request_id code message
  catch _ =>
    emit "JACOBIAN_DECLARATION_ERROR" <|
      errorEnvelope query.request_id "LEAN_QUERY_FAILED" "query execution failed"
