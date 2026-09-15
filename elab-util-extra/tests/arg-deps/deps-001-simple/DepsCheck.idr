module DepsCheck

import Infra

%language ElabReflection

%runElab ppTys
  [ Unit
  , (Nat -> Nat)
  , (Nat -> Nat -> Nat)
  , ({a : Type} -> List a -> Nat)
  ]
