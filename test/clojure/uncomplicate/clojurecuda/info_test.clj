;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.clojurecuda.info-test
  (:require [midje.sweet :refer [facts =>]]
            [uncomplicate.commons.core :refer [with-release info]]
            [uncomplicate.clojurecuda
             [core :refer [compile! context device function init module program stream with-context
                           current-context push-context!]]
             [info :refer [driver-version limit limit! stream-flag stream-ctx
                           max-active-blocks-per-multiprocessor available-dynamic-mem-per-block
                           max-potential-block-size]]]
            [uncomplicate.clojurecuda.internal.constants :refer [stream-flags]]))

(init)

(facts
 "Driver info tests."
 (pos? (driver-version)) => true)

(facts
 "Device info tests."
 (count (info (device 0))) => 83)

(with-release [ctx (context (device))]
  (push-context! ctx)
  (facts
   "Context info tests."
   (count (info ctx)) => 16
   (limit! :stack-size 512) => 512
   (limit :stack-size) => 512))

(with-context (context (device))
  (with-release [hstream (stream :non-blocking)]
    (facts
      "Stream info tests."
      (stream-flag hstream) => (stream-flags :non-blocking)
      (:flag (info hstream )) => :non-blocking
      (count (info hstream)) => 2
      (stream-ctx hstream) => (current-context))))

(let [program-source (slurp "test/cuda/uncomplicate/clojurecuda/kernels/test.cu")]
  (with-context (context (device))
    (with-release [prog (compile! (program program-source))
                   modl (module prog)
                   fun (function modl "inc")]
      (facts
       "function info tests."
       (count (info fun)) => 7
       (max-potential-block-size fun) => [68 1024]
       (available-dynamic-mem-per-block fun 1 1024) => 49152
       (max-active-blocks-per-multiprocessor fun 1024 0) => 1))))
