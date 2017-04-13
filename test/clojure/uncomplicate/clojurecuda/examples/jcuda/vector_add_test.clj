;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.clojurecuda.examples.jcuda.vector-add-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [release with-release wrap-int]]
            [uncomplicate.clojurecuda.core :refer :all])
  (:import clojure.lang.ExceptionInfo
           [java.nio ByteBuffer ByteOrder]))

(extend-type (Class/forName "[Ljcuda.driver.CUdeviceptr;")
  Mem
  (ptr [this]
    (jcuda.Pointer/to ^"[Ljcuda.driver.CUdeviceptr;"this)))

(init)
(let [program-source (slurp "test/cuda/examples/jcuda/jnvrtc-vector-add.cuda")
      block-size-x 256]
  (with-context (context (device))
    (with-release [program (compile-program! (create-program program-source))
                   m (module program)
                   func (function m "add")
                   host-a (float-array [1 2 3])
                   host-b (float-array [2 3 4])
                   host-sum (float-array 3)
                   gpu-a (mem-alloc (* Float/BYTES 3))
                   gpu-b (mem-alloc (* Float/BYTES 3))
                   gpu-sum (mem-alloc (* Float/BYTES 3))]
      (facts
       "Vector add JCuda example."
       (memcpy-host! host-a gpu-a)
       (memcpy-host! host-b gpu-b)
       (launch! func (count host-sum) block-size-x (parameters (count host-sum) gpu-a gpu-b gpu-sum)) => func
       (synchronize)
       (seq (memcpy-host! gpu-sum host-sum)) => (seq [3.0 5.0 7.0])))))
