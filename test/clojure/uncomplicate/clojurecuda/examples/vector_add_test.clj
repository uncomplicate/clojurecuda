;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.clojurecuda.examples.vector-add-test
  (:require [midje.sweet :refer [facts =>]]
            [uncomplicate.commons.core :refer [with-release size]]
            [uncomplicate.clojure-cpp :refer [float-pointer pointer-seq]]
            [uncomplicate.clojurecuda.core
             :refer [compile! context device function grid-1d init launch! mem-alloc-driver
                     mem-alloc-pinned mem-alloc-runtime memcpy-host! module parameters program
                     synchronize! with-context]]))

(init)

(let [program-source (slurp "test/cuda/examples/jnvrtc-vector-add.cu")]
  (with-context (context (device))
    (with-release [prog (compile! (program program-source))
                   m (module prog)
                   add (function m "add")
                   host-a (float-pointer [1 2 3])
                   host-b (float-pointer [2 3 4])
                   host-sum (float-pointer 3)
                   gpu-a (mem-alloc-runtime (* Float/BYTES 3))
                   gpu-b (mem-alloc-driver (* Float/BYTES 3))
                   gpu-sum (mem-alloc-pinned (* Float/BYTES 3))]
      (facts
       "Vector add JCuda example."
       (memcpy-host! host-a gpu-a)
       (memcpy-host! host-b gpu-b)
       (launch! add (grid-1d (size host-sum)) (parameters (size host-sum) gpu-a gpu-b gpu-sum))
       (synchronize!)
       (pointer-seq (memcpy-host! gpu-sum host-sum)) => (seq [3.0 5.0 7.0])))))
