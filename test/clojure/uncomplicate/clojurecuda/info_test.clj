;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.clojurecuda.info-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [release with-release info]]
            [uncomplicate.clojurecuda
             [core :refer :all]
             [info :refer :all]]
            [uncomplicate.clojurecuda.internal.constants :refer [stream-flags]]))

(init)

(facts
 "Driver info tests."
 (pos? (driver-version)) => true)

(facts
 "Device info tests."
 (count (info (device 0))) => 83)

(with-release [ctx (context (device))]
  (facts
   "Context info tests."
   (count (info ctx)) => 1
   (limit! :stack-size 512) => 512
   (limit :stack-size) => 512
   (count (context-info)) => 10))

(with-context (context (device))
  (with-release [hstream (stream :non-blocking)]
    (facts
     "Stream info tests."
     (count (info hstream)) => 2
     (stream-flag hstream) => (stream-flags :non-blocking)
     (:flag (info hstream))))) => :non-blocking

(let [program-source (slurp "test/cuda/uncomplicate/clojurecuda/kernels/test.cu")]
  (with-context (context (device))
    (with-release [prog (compile! (program program-source))
                   modl (module prog)
                   fun (function modl "inc")]
      (facts
       "function info tests."
       (count (info fun)) => 7))))
