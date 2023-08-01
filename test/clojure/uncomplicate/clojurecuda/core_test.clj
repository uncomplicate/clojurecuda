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
            [uncomplicate.commons.core :refer [release with-release size bytesize extract let-release]]
            [uncomplicate.clojure-cpp :as cpp
             :refer [pointer float-pointer byte-pointer get-entry get-float put-float! int-pointer
                     long-pointer put-int! pointer-seq put-entry! fill!]]
            [uncomplicate.clojurecuda
             [core :refer :all]
             [info :as info :refer [pci-bus-id-string]]])
  (:import clojure.lang.ExceptionInfo))

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

(facts
 "Test Parameters"
 (with-context (context (device))
   (with-release [cnt 3
                  extra 4
                  gpu-a (mem-alloc (* Float/BYTES (+ cnt extra)))
                  params (parameters cnt gpu-a)]
     (size params) => 2
     (get-entry (int-pointer (get-entry params 0))) => 3
     (get-entry (long-pointer (get-entry params 1))) => (extract gpu-a))))

(let [program-source (slurp "test/cuda/uncomplicate/clojurecuda/kernels/test.cu")
      cnt 300
      extra 5]
  (with-context (context (device))
      (with-release [prog (compile! (program program-source {"dummy" "placeholder"}))
                     grid (grid-1d cnt (min 256 cnt))]
        (with-release [modl (module prog)
                       fun (function modl "inc")
                       strm (stream :non-blocking)
                       host-a (float-pointer (+ cnt extra))
                       gpu-a (mem-alloc (* Float/BYTES (+ cnt extra)))]

          (facts
           "Test launch"
           (fill! host-a 0)
           (put-entry! host-a 0 1)
           (put-entry! host-a 10 100)
           (memcpy-host! host-a gpu-a strm) => gpu-a
           (launch! fun grid strm (parameters (int cnt) gpu-a)) => strm
           (synchronize! strm) => strm
           (memcpy-host! gpu-a host-a strm) => host-a
           (get-entry host-a 0) => 2.0
           (get-entry host-a 10) => 101.0
           (get-entry host-a (dec cnt)) => 1.0
           (get-entry host-a cnt) => 0.0
           (get-entry host-a (dec (+ cnt extra))) => 0.0))

        (with-release [modl (module)]
          (facts
           "Test device globals"
           (load! modl prog) => modl
           (with-release [fun (function modl "constant_inc")
                          gpu-a (global modl "gpu_a")
                          constant-gpu-a (global modl "constant_gpu_a")]
             (pointer-seq (memcpy-host! gpu-a (float-pointer 3))) => (seq [1.0 2.0 3.0])
             (memcpy! gpu-a constant-gpu-a) => constant-gpu-a
             (launch! fun (grid-1d 3) (parameters 3 gpu-a))
             (pointer-seq (memcpy-host! constant-gpu-a (float-pointer 3))) => (seq [1.0 2.0 3.0])
             (pointer-seq (memcpy-host! gpu-a (float-pointer 3))) => (seq [2.0 4.0 6.0])))))))

;; =============== Stream Management Tests ==============================================

(with-context (context (device 0) :map-host)

  (facts
   "Stream creation and memory copy tests."
   (with-release [strm (stream :non-blocking)
                  cuda1 (mem-alloc Float/BYTES)
                  cuda2 (mem-alloc Float/BYTES)
                  host1 (float-array [173.0])
                  host2 (byte-pointer Float/BYTES)]
     (memcpy-host! host1 cuda1 strm) => cuda1
     (synchronize! strm)
     (memcpy! cuda1 cuda2) => cuda2
     (memcpy-host! cuda2 host2 strm) => host2
     (synchronize! strm)
     (get-float host2 0) => 173.0))

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
   "Host functions."
   (let [ch (chan)]
     (with-release [strm (stream :non-blocking)
                    cuda1 (mem-alloc Float/BYTES)
                    cuda2 (mem-alloc Float/BYTES)
                    host1 (float-array [163.0])
                    host2 (float-pointer [12])
                    ch (chan)]
       (listen! strm ch :host)
       (memcpy-host! host1 cuda1 strm) => cuda1
       (memcpy! cuda1 cuda2 strm) => cuda2
       (synchronize! strm)
       (memcpy-host! cuda2 (float-array 1) strm) => (throws Exception)
       (get-entry (memcpy-host! cuda2 host2 strm) 0) => 163.0
       (<!! ch) => :host))))

;; =============== Memory Management Tests ==============================================

(with-release [dev (device 0)]
  (with-context (context dev :map-host)

    (facts
     "mem-alloc tests."
     (mem-alloc 0) => (throws ExceptionInfo)
     (with-release [buf (mem-alloc Float/BYTES)]
       (bytesize buf) => Float/BYTES))

    (facts
     "Linear memory tests."
     (with-release [cuda1 (mem-alloc Float/BYTES)
                    cuda2 (mem-alloc Float/BYTES)
                    host1 (float-array [173.0])
                    host2 (byte-pointer Float/BYTES)]
       (memcpy-host! host1 cuda1) => cuda1
       (memcpy! cuda1 cuda2) => cuda2
       (memcpy-host! cuda2 host2) => host2
       (get-float host2 0) => 173.0))

    (facts
     "Linear memory sub-region tests."
     (with-release [cuda (mem-alloc 20)]
       (memcpy-host! (float-array [1 2 3 4 5]) cuda) => cuda
       (let-release [cuda1 (mem-sub-region cuda 0 8)
                     cuda2 (mem-sub-region cuda 8 12)]
         (mem-sub-region cuda 8 20) => (throws ExceptionInfo)
         (pointer-seq (memcpy-host! cuda1 (float-pointer 2))) => [1.0 2.0]
         (pointer-seq (memcpy-host! cuda2 (float-pointer 3))) => [3.0 4.0 5.0]
         (do (release cuda1)
             (release cuda2)
             (pointer-seq (memcpy-host! cuda (float-pointer 5))) => [1.0 2.0 3.0 4.0 5.0]))))

    (facts
     "Runtime cudaMalloc tests."
     (with-release [cuda1 (mem-alloc-device Float/BYTES :float)
                    host1 (float-pointer [100.0])
                    host2 (mem-alloc-mapped Float/BYTES :float)
                    zero (mem-alloc-device 0)]
       zero => truthy
       (bytesize cuda1) => Float/BYTES
       (memcpy-host! host1 cuda1) => cuda1
       (synchronize!)
       (pointer-seq (memcpy-host! cuda1 (float-pointer 1))) => [100.0]
       (seq (memcpy! cuda1 host2)) => [100.0]))

    (facts
     "Pinned memory tests."
     (with-release [pinned-host (mem-alloc-pinned Float/BYTES :float :devicemap)
                    cuda1 (mem-alloc Float/BYTES)]
       (mem-alloc-pinned Float/BYTES :unknown) => (throws ExceptionInfo)
       (bytesize pinned-host) => Float/BYTES
       (put-entry! pinned-host 0 13)
       (memcpy-host! pinned-host cuda1) => cuda1
       (put-entry! pinned-host 0 11)
       (memcpy! cuda1 pinned-host) => pinned-host
       (synchronize!)
       (get-entry pinned-host 0) => 13.0
       (pointer-seq (memcpy-host! cuda1 (float-pointer 1))) => [13.0]))

    (facts
     "Mapped memory tests."
     (with-release [mapped-host (mem-alloc-mapped Float/BYTES :float)
                    cuda1 (mem-alloc Float/BYTES)
                    mapped-host2 (mem-alloc-mapped Float/BYTES :float)]
       (bytesize mapped-host) => Float/BYTES
       (put-entry! mapped-host 0 14.0)
       (memcpy-host! mapped-host cuda1) => cuda1
       (get-entry (memcpy-host! cuda1 (float-pointer 1)) 0) => 14.0
       (get-entry (memcpy! cuda1 mapped-host2)) => 14.0
       (synchronize!)
       (seq mapped-host2) => [14.0]))

    (facts
     "memset tests."
     (with-release [pinned-host (mem-alloc-pinned (* 2 Integer/BYTES))
                    cuda1 (mem-alloc (* 2 Integer/BYTES))]
       (put-int! (pointer pinned-host) 0 24)
       (put-int! (pointer pinned-host) 1 34)
       (memcpy-host! pinned-host cuda1) => cuda1
       (pointer-seq (memcpy-host! cuda1 (int-pointer 2))) => [24 34]
       (memcpy-host! (memset! cuda1 0 1) pinned-host) => pinned-host
       (pointer-seq (int-pointer pinned-host)) => [0 34]
       (memcpy-host! (memset! cuda1 (int 0)) pinned-host) => pinned-host
       (synchronize!)
       (pointer-seq (int-pointer pinned-host)) => [0 0]) )

    (when (and (info/managed-memory dev) (info/concurrent-managed-access dev))
      (facts
       "mem-alloc-managed tests."
       (with-release [host0 (float-pointer [15])
                      host1 (float-pointer 1)
                      cuda0 (mem-alloc-managed Float/BYTES :host)
                      cuda1 (mem-alloc-managed Float/BYTES :global)]

         (bytesize cuda0) => Float/BYTES
         (mem-alloc-managed Float/BYTES :unknown) => (throws ExceptionInfo)
         (memcpy-host! host0 cuda0) => cuda0
         (memcpy! cuda0 cuda1) => cuda1
         (memcpy-host! cuda1 host1) => host1
         (get-entry host1 0) => 15.0)))

    (when (info/managed-memory dev)
      (facts
        "mem-alloc-managed with globally shared attached memory tests."
        (with-release [host0 (float-pointer [16])
                       host1 (float-pointer 1)
                       cuda0 (mem-alloc-managed Float/BYTES :host)
                       cuda1 (mem-alloc-managed Float/BYTES :global)]
          (attach-mem! nil cuda0 Float/BYTES :global) => nil
          (bytesize cuda0) => Float/BYTES
          (memcpy-host! host0 cuda0) => cuda0
          (memcpy! cuda0 cuda1) => cuda1
          (memcpy-host! cuda1 host1) => host1
          (get-entry host1 0) => 16.0))
      (facts
        "mem-alloc-managed with attached memory tests."
        (with-release [host0 (float-pointer [17])
                       host1 (float-pointer 1)
                       cuda0 (mem-alloc-managed Float/BYTES :host)
                       cuda1 (mem-alloc-managed Float/BYTES :global)]
          (let [hstream (attach-mem! cuda0 Float/BYTES :single)]
            (bytesize cuda0) => Float/BYTES
            (if (info/concurrent-managed-access dev)
              (memcpy-host! host0 cuda0) => cuda0
              (memcpy-host! host0 cuda0) => (throws ExceptionInfo))
            (memcpy-host! host0 cuda0 hstream) => cuda0
            (memcpy! cuda0 cuda1 hstream) => cuda1
            (memcpy-host! cuda1 host1 hstream) => host1
            (synchronize! hstream)
            (get-entry host1 0) => 17.0))))

    (facts
     "mem-alloc-registered tests."
     (with-release [host0 (byte-pointer Float/BYTES)
                    host1 (byte-pointer Float/BYTES)
                    cuda0 (mem-register-pinned host0)
                    cuda1 (mem-register-pinned host1)]

       (bytesize cuda0) => Float/BYTES
       (put-float! host0 0 44.0)
       (memcpy! cuda0 cuda1) => cuda1
       (get-float host1 0) => 44.0))))

;; ================= Peer Access Management Tests =====================================

(facts
  "Peer access tests (requires 2 devices)."
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
                        (memcpy-host! gpu-a-sum host-sum) => (seq [3.0 5.0 700000.0])
                        (disable-peer-access! peer-ctx) => peer-ctx
                        (in-context peer-ctx (disable-peer-access! ctx) => ctx))))
        (when-let [dev (first devices)]
          (p2p-attribute dev dev :access-supported) => (throws ExceptionInfo)
          (can-access-peer dev dev) => false)))))
