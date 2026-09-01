module TypesAndInvolved

import Infra
import Language.Reflection.Compat

%language ElabReflection

%default total

public export
typesAndInvolved : List (Name, Count, List Name)
typesAndInvolved =
  [ ("Nat", M0, ["Nat"])
  , ("List", M0, ["List"])
  , ("Vect", M0, ["Vect", "Nat"])
  , ("Vect", MW, ["Vect"])
  ]

%runElab for_ typesAndInvolved $ \(n, r, ns) => printInvolvedTypesVerdict n r ns
