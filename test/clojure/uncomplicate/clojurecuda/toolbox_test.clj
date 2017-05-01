;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.clojurecuda.toolbox-test
  (:require [midje.sweet :refer [facts =>]]
            [uncomplicate.commons.core :refer [release with-release]]
            [uncomplicate.clojurecuda
             [core :refer :all]
             [nvrtc :refer [compile! program]]
             [toolbox :refer :all]])
  (:import clojure.lang.ExceptionInfo
           [java.nio ByteBuffer ByteOrder]))

(init)

(let [program-source (str (slurp "src/cuda/kernels/reduction.cu") "\n"
                          (slurp "test/cuda/kernels/sum.cu"))
      cnt 2000]

  (with-context (context (device))
    (with-release [prog (compile! (program program-source) ["-DREAL=float"])
                   m (module prog)
                   sum-reduction (function m "sum_reduction")
                   sum (function m "sum")
                   host-x (float-array (range cnt))
                   gpu-x (mem-alloc (* Float/BYTES cnt))
                   gpu-acc (mem-alloc (* Float/BYTES (count-blocks 1024 cnt)))]
      (facts
       "Simple sum reduction."
       (memcpy-host! host-x gpu-x)
       (launch-reduce! nil sum sum-reduction
                       (parameters cnt gpu-x gpu-acc) (parameters cnt gpu-acc) cnt 1024 )
       (seq (memcpy-host! gpu-acc (float-array 1))) => [(double (apply + (range 2000)))]))))
