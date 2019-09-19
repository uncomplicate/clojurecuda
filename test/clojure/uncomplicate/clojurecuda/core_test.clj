;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.clojurecuda.core-test
  (:require [midje.sweet :refer :all]
            [clojure.core.async :refer [chan <!!]]
            [uncomplicate.commons.core :refer [release with-release]]
            [uncomplicate.clojurecuda
             [core :refer :all]
             [info :as info :refer [pci-bus-id-string]]]
            [uncomplicate.clojurecuda.internal.protocols :refer [size host-buffer]])
  (:import clojure.lang.ExceptionInfo
           [java.nio ByteBuffer ByteOrder]))

;; ================== Driver tests ======================================================

(facts
 "Driver tests."
 (init) => true)

(facts
 "Device tests."
 (<= 0 (device-count)) => true
 (device 0) => truthy
 (device -1) => (throws ExceptionInfo)
 (device 33) => (throws ExceptionInfo)
 (device (pci-bus-id-string (device))) => (device))

;; ===================== Context Management Tests =======================================

(facts
 "Context tests"
 (with-release [dev (device 0)]
   (let [ctx (context dev :sched-auto)]
     ctx => truthy
     (release ctx) => true
     (context dev :unknown) => (throws ExceptionInfo))
   (let [ctx1 (context dev :sched-blocking-sync)
         ctx2 (context dev :sched-blocking-sync)]
     (with-context ctx1
       (with-context ctx2
         (current-context) => ctx2
         (do (pop-context!) (current-context)) => ctx1
         (current-context! ctx2) => ctx2
         (current-context) => ctx2
         (release ctx2) => true
         (release ctx2) => true)))))

;; =============== Module Management & Execution Control Tests =====================================

(let [program-source (slurp "test/cuda/uncomplicate/clojurecuda/kernels/test.cu")
      cnt 300
      extra 5]
  (with-context (context (device))
      (with-release [prog (compile! (program program-source {"dummy" "placeholder"}))
                     grid (grid-1d cnt (min 256 cnt))]
        (with-release [modl (module prog)
                       fun (function modl "inc")
                       host-a (float-array (+ cnt extra))
                       gpu-a (mem-alloc (* Float/BYTES (+ cnt extra)))]

          (facts
           "Test launch"
           (aset-float host-a 0 1)
           (aset-float host-a 10 100)
           (memcpy-host! host-a gpu-a) => gpu-a
           (launch! fun grid (parameters cnt gpu-a)) => nil
           (synchronize!) => true
           (memcpy-host! gpu-a host-a) => host-a
           (aget host-a 0) => 2.0
           (aget host-a 10) => 101.0
           (aget host-a (dec cnt)) => 1.0
           (aget host-a cnt) => 0.0
           (aget host-a (dec (+ cnt extra))) => 0.0))

        (with-release [modl (module)]
          (facts
           "Test device globals"
           (load! modl prog) => modl
           (with-release [fun (function modl "constant_inc")
                          gpu-a (global modl "gpu_a")
                          constant-gpu-a (global modl "constant_gpu_a")]
             (seq (memcpy-host! gpu-a  (float-array 3))) => (seq [1.0 2.0 3.0])
             (memcpy! gpu-a constant-gpu-a) => constant-gpu-a
             (launch! fun (grid-1d 3) (parameters 3 gpu-a))
             (seq (memcpy-host! constant-gpu-a (float-array 3))) => (seq [1.0 2.0 3.0])
             (seq (memcpy-host! gpu-a (float-array 3))) => (seq [2.0 4.0 6.0])))))))

;; =============== Stream Management Tests ==============================================

(with-context (context (device 0) :map-host)

  (facts
   "Stream creation and memory copy tests."
   (with-release [strm (stream :non-blocking)
                  cuda1 (mem-alloc Float/BYTES)
                  cuda2 (mem-alloc Float/BYTES)
                  host1 (float-array [173.0])
                  host2 (.order (ByteBuffer/allocateDirect Float/BYTES) (ByteOrder/nativeOrder))]
     (memcpy-host! host1 cuda1 strm) => cuda1
     (synchronize! strm)
     (memcpy! cuda1 cuda2) => cuda2
     (memcpy-host! cuda2 host2 strm) => host2
     (synchronize! strm)
     (.getFloat ^ByteBuffer host2 0) => 173.0))

  (facts
   "Stream and memory release."
   (with-release [strm (stream :non-blocking)
                  cuda (mem-alloc Float/BYTES)]
     (release strm) => true
     (release strm) => true
     (release cuda) => true
     (memcpy! cuda cuda) => (throws NullPointerException)
     (release cuda) => true)))

(with-context (context (device 0) :map-host)

  (facts
   "Stream callbacks."
   (let [ch (chan)]
     (with-release [strm (stream :non-blocking)
                    cuda1 (mem-alloc Float/BYTES)
                    cuda2 (mem-alloc Float/BYTES)
                    host1 (float-array [173.0])
                    host2 (.order (ByteBuffer/allocateDirect Float/BYTES) (ByteOrder/nativeOrder))
                    cbk (callback ch)]
       (add-callback! strm cbk)
       (memcpy-host! host1 cuda1 strm) => cuda1
       (memcpy! cuda1 cuda2) => cuda2
       (:data (<!! ch)) => strm))))

;; =============== Memory Management Tests ==============================================

(with-release [dev (device 0)]
  (with-context (context dev :map-host)

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
     "Linear memory sub-region tests."
     (with-release [cuda (mem-alloc 20)]
       (memcpy-host! (float-array [1 2 3 4 5]) cuda) => cuda
       (let [cuda1 (mem-sub-region cuda 0 8)
             cuda2 (mem-sub-region cuda 8 1000)]
         (seq (memcpy-host! cuda1 (float-array 2))) => (seq (float-array [1 2]))
         (seq (memcpy-host! cuda2 (float-array 3))) => (seq (float-array [3 4 5]))
         (do (release cuda1)
             (release cuda2)
             (seq (memcpy-host! cuda (float-array 5))) => (seq (float-array [1 2 3 4 5]))))))

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
     "memset tests."
     (with-release [mapped-host (mem-alloc-host (* 2 Integer/BYTES))
                    host (int-array 2)]
       (.putInt ^ByteBuffer (host-buffer mapped-host) 0 24) => (host-buffer mapped-host)
       (.putInt ^ByteBuffer (host-buffer mapped-host) Integer/BYTES 34) => (host-buffer mapped-host)
       (memcpy-host! mapped-host host) => host
       (seq host) => (list 24 34)
       (memcpy-host! (memset! mapped-host 0 1) host)
       (seq host) => (list 0 34)
       (memcpy-host! (memset! mapped-host 0) host)
       (seq host) => (list 0 0)) )

    (when (and (info/managed-memory dev) (info/concurrent-managed-access dev))
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
         (aget ^floats host1 0) => 15.0)))

    (when (info/managed-memory dev)
      (facts
        "mem-alloc-managed with globally shared attached memory tests."
        (with-release [host0 (float-array [16])
                       host1 (float-array 1)
                       cuda0 (mem-alloc-managed Float/BYTES :host)
                       cuda1 (mem-alloc-managed Float/BYTES :global)]
          (attach-mem! nil cuda0 Float/BYTES :global) => nil
          (size cuda0) => Float/BYTES
          (memcpy-host! host0 cuda0) => cuda0
          (memcpy! cuda0 cuda1) => cuda1
          (memcpy-host! cuda1 host1) => host1
          (aget ^floats host1 0) => 16.0))
      (facts
        "mem-alloc-managed with attached memory tests."
        (with-release [host0 (float-array [17])
                       host1 (float-array 1)
                       cuda0 (mem-alloc-managed Float/BYTES :host)
                       cuda1 (mem-alloc-managed Float/BYTES :global)]
          (let [hstream (attach-mem! cuda0 Float/BYTES :single)]
            (size cuda0) => Float/BYTES
            (if (info/concurrent-managed-access dev)
              (memcpy-host! host0 cuda0) => cuda0
              (memcpy-host! host0 cuda0) => (throws ExceptionInfo))
            (memcpy-host! host0 cuda0 hstream) => cuda0
            (memcpy! cuda0 cuda1 hstream) => cuda1
            (memcpy-host! cuda1 host1 hstream) => host1
            (synchronize! hstream)
            (aget ^floats host1 0) => 17.0))))

    (facts
     "mem-alloc-registered tests."
     (with-release [host0 (.order (ByteBuffer/allocateDirect Float/BYTES) (ByteOrder/nativeOrder))
                    host1 (.order (ByteBuffer/allocateDirect Float/BYTES) (ByteOrder/nativeOrder))
                    cuda0 (mem-host-register host0)
                    cuda1 (mem-host-register host1)]

       (size cuda0) => Float/BYTES
       (.putFloat host0 0 44.0)
       (memcpy! cuda0 cuda1) => cuda1
       (.getFloat host1 0) => 44.0))))

;; ================= Peer Access Management Tests =====================================

(facts
  "Peer access tests"
  (let [num-dev (device-count)
        devices (mapv device (range num-dev))
        combinations (set (for [x (range num-dev) y (range num-dev) :when (not= x y)] #{x y}))
        p2p? (fn [num-pair] (let [[a b] (vec num-pair)
                                  dev-a (nth devices a)
                                  dev-b (nth devices b)]
                              (when (and (p2p-attribute dev-a dev-b :access-supported)
                                         (can-access-peer dev-a dev-b)
                                         (can-access-peer dev-b dev-a))
                                [dev-a dev-b])))]
    (if-let [[dev-a dev-b] (some p2p? combinations)]
      (let [program-source (slurp "test/cuda/examples/jcuda/jnvrtc-vector-add.cu")
            ^:const vctr-len 3]
        (with-release [host-a (float-array [1 2 3])
                       host-b (float-array [2 3 4])
                       host-sum (float-array vctr-len)
                       ctx (context dev-a)
                       peer-ctx (context dev-b)]
          (in-context ctx
            (with-release [prog (compile! (program program-source))
                           m (module prog)
                           vector-add (function m "add")
                           gpu-a (mem-alloc (* Float/BYTES vctr-len))
                           gpu-a-sum (mem-alloc (* Float/BYTES vctr-len))
                           gpu-b (in-context peer-ctx (mem-alloc (* Float/BYTES vctr-len)))]
              (disable-peer-access! peer-ctx) => (throws ExceptionInfo)
              (in-context peer-ctx (disable-peer-access! ctx) => (throws ExceptionInfo))
              (memcpy-host! host-a gpu-a) => gpu-a
              (in-context peer-ctx (memcpy-host! host-b gpu-b) => gpu-b)
              (enable-peer-access! peer-ctx) => peer-ctx
              (in-context peer-ctx (enable-peer-access! ctx) => ctx)
              (launch! vector-add (grid-1d vctr-len) (parameters vctr-len gpu-a gpu-b gpu-a-sum))
              (synchronize!)
              (seq (memcpy-host! gpu-a-sum host-sum)) => (seq [3.0 5.0 7.0])
              (disable-peer-access! peer-ctx) => peer-ctx
              (in-context peer-ctx (disable-peer-access! ctx) => ctx)))))
      (when-let [dev (first devices)]
        (p2p-attribute dev dev :access-supported) => (throws ExceptionInfo)
        (can-access-peer dev dev) => false))))
