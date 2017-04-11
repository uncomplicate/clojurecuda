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
   (context dev :unknown) => (throws ExceptionInfo)
   (with-context (context dev :sched-blocking-sync)
     *context* => truthy)))

(with-context (context (device 0) :map-host)

  (facts
   "mem-alloc tests."
   (mem-alloc 0) => (throws ExceptionInfo)
   (with-release [buf (mem-alloc Float/BYTES)]
     (size buf) => Float/BYTES))

  (facts
   "Linear memory tests."
   (with-release [cuda1 (mem-alloc Float/BYTES)
                  cuda2 (mem-alloc Float/BYTES)
                  host1 (float-array [173.0])
                  host2 (.order (ByteBuffer/allocateDirect Float/BYTES) (ByteOrder/nativeOrder))]
     (memcpy-host! host1 cuda1) => cuda1
     (memcpy! cuda1 cuda2) => cuda2
     (memcpy-host! cuda2 host2) => host2
     (.getFloat ^ByteBuffer host2 0) => 173.0))

  (facts
   "mem-host-alloc tests."
   (with-release [mapped-host (mem-host-alloc Float/BYTES :devicemap)
                  host (float-array 1)]
     (mem-host-alloc Float/BYTES :unknown) => (throws ExceptionInfo)
     (size mapped-host) => Float/BYTES
     (.putFloat ^ByteBuffer (host-buffer mapped-host) 0 13) => (host-buffer mapped-host)
     (memcpy-host! mapped-host host) => host
     (aget ^floats host 0) => 13.0))

  (facts
   "mem-alloc-host tests."
   (with-release [mapped-host (mem-alloc-host Float/BYTES)
                  host (float-array 1)]
     (size mapped-host) => Float/BYTES
     (.putFloat ^ByteBuffer (host-buffer mapped-host) 0 14) => (host-buffer mapped-host)
     (memcpy-host! mapped-host host) => host
     (aget ^floats host 0) => 14.0))

  (facts
   "mem-alloc-managed tests."
   (with-release [host0 (float-array [15])
                  host1 (float-array 1)
                  cuda0 (mem-alloc-managed Float/BYTES :host)
                  cuda1 (mem-alloc-managed Float/BYTES :global)]

     (size cuda0) => Float/BYTES
     (mem-alloc-managed Float/BYTES :unknown) => (throws ExceptionInfo)
     (memcpy-host! host0 cuda0) => cuda0
     (memcpy! cuda0 cuda1) => cuda1
     (memcpy-host! cuda1 host1) => host1
     (aget ^floats host1 0) => 15.0))

  (facts
   "mem-alloc-managed tests."
   (with-release [host0 (.order (ByteBuffer/allocateDirect Float/BYTES) (ByteOrder/nativeOrder))
                  host1 (.order (ByteBuffer/allocateDirect Float/BYTES) (ByteOrder/nativeOrder))
                  cuda0 (mem-host-register host0)
                  cuda1 (mem-host-register host1)]

     (size cuda0) => Float/BYTES
     (.putFloat host0 0 44.0)
     (memcpy! cuda0 cuda1) => cuda1
     (.getFloat host1 0) => 44.0))

  )
