;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.clojurecuda.examples.jcuda.dynamic-parallelism-test
  (:require [midje.sweet :refer :all]
            [clojure.java.io :as io]
            [uncomplicate.commons.core :refer [release with-release]]
            [uncomplicate.clojurecuda
             [core :refer :all]
             [nvrtc :refer :all]])
  (:import clojure.lang.ExceptionInfo
           [java.nio ByteBuffer ByteOrder]))

(init)

(let [program-source (slurp "test/cuda/examples/jcuda/dynamic-parallelism.cu")
      num-parent-threads 8
      num-child-threads 8
      num-elements (* num-parent-threads num-child-threads)]
  (with-context (context (device))
    (with-release [prog (compile! (program program-source)
                                  ["-arch=compute_35" "--relocatable-device-code=true"
                                   "-default-device"])
                   linked-prog (link [[:library (io/file "/usr/local/cuda/lib64/libcudadevrt.a")]
                                      [:ptx prog]])
                   m (module linked-prog)
                   parent (function m "parentKernel")
                   data (mem-alloc (* Float/BYTES num-elements))]
      (facts
       "Dynamic parallelism JCuda example."
       (memcpy-host! (float-array num-elements) data)
       (launch! parent (grid-1d (+ num-elements num-elements (- 1)) num-parent-threads)
                (parameters num-elements data))
       (seq (memcpy-host! data (float-array num-elements)))
       => (map float (seq [0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7
                           1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7
                           2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7
                           3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7
                           4.0 4.1 4.2 4.3 4.4 4.5 4.6 4.7
                           5.0 5.1 5.2 5.3 5.4 5.5 5.6 5.7
                           6.0 6.1 6.2 6.3 6.4 6.5 6.6 6.7
                           7.0 7.1 7.2 7.3 7.4 7.5 7.6 7.7]))))))
