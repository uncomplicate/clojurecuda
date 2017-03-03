;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.clojurecuda.core-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [release with-release]]
            [uncomplicate.clojurecuda
             [core :refer :all]])
  (:import clojure.lang.ExceptionInfo
           [java.nio ByteBuffer ByteOrder]))

;; ================== Driver tests ========================
(facts
 "Driver tests."
 (init) => true)

(facts
 "Device tests."
 (<= 0 (device-count)) => true
 (device 0) => truthy
 (device -1) => (throws ExceptionInfo)
 (device 33) => (throws ExceptionInfo))

(facts
 "Context tests"
 (let [dev (device 0)
       ctx (context* dev 0)]
    ctx => truthy
    (release ctx) => true
    (context dev :unknown) => (throws NullPointerException)
    (with-context (context dev :block-sync)
      *context* => truthy)))

(with-context (context (device 0))

  (facts
   "Linear memory tests"
   (with-release [cuda1 (mem-alloc Float/BYTES)
                  cuda2 (mem-alloc Float/BYTES)
                  host1 (float-array [173.0])
                  host2 (.order (ByteBuffer/allocateDirect Float/BYTES) (ByteOrder/nativeOrder))]
     (memcpy-host! host1 cuda1) => cuda1
     (memcpy! cuda1 cuda2) => cuda2
     (memcpy-host! cuda2 host2) => host2
     (.getFloat ^ByteBuffer host2 0) => 173.0)))
