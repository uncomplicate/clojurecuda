;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.clojurecuda.toolbox
  "Various helpers that are not needed by ClojureCUDA itself,
  but may be very helpful in applications. See Neanderthal and Bayadera libraries
  for the examples of how to use them."
  (:require [uncomplicate.clojurecuda.core :refer :all]))

(let [pointer-arr-class (class (make-parameters 0))]

  (defn launch-reduce!
    ([hstream main-kernel reduction-kernel main-params reduction-params n local-n]
     (let [main-params (if (instance? pointer-arr-class main-params)
                         (set-parameter! main-params 0 n)
                         (apply parameters n main-params))
           reduction-params (if (instance? pointer-arr-class reduction-params)
                              reduction-params
                              (apply parameters Integer/MAX_VALUE reduction-params))]
       (launch! main-kernel (grid-1d n local-n) hstream main-params)
       (loop [global-size (blocks-count local-n n)]
         (when (< 1 global-size)
           (launch! reduction-kernel (grid-1d global-size local-n) hstream
                    (set-parameter! reduction-params 0 global-size))
           (recur (blocks-count local-n global-size))))
       hstream))
    ([hstream main-kernel reduction-kernel main-params reduction-params m n local-m local-n]
     (let [main-params (if (instance? pointer-arr-class main-params)
                         (set-parameters! main-params 0 m n)
                         (apply parameters m n main-params))
           reduction-params (if (instance? pointer-arr-class reduction-params)
                              reduction-params
                              (apply parameters Integer/MAX_VALUE Integer/MAX_VALUE reduction-params))]
       (if (or (< 1 ^long local-n) (<= ^long local-n ^long n))
         (loop [hstream (launch! main-kernel (grid-2d m n local-m local-n) hstream main-params)
                global-size (blocks-count local-n n)]
           (if (= 1 global-size)
             hstream
             (recur (launch! reduction-kernel (grid-2d m global-size local-m local-n) hstream
                             (set-parameters! reduction-params 0 m global-size))
                    (blocks-count local-n global-size))))
         (throw (IllegalArgumentException.
                 (format "local-n %d would cause infinite recursion for n:%d." local-n n))))))))

(defn read-int
  (^long [cu-buf]
   (let [res (int-array 1)]
     (memcpy-host! cu-buf res)
     (aget res 0)))
  (^long [hstream cu-buf]
   (let [res (int-array 1)]
     (memcpy-host! cu-buf res hstream)
     (aget res 0))))

(defn read-long
  (^long [cu-buf]
   (let [res (long-array 1)]
     (memcpy-host! cu-buf res)
     (aget res 0)))
  (^long [hstream cu-buf]
   (let [res (long-array 1)]
     (memcpy-host! cu-buf res hstream)
     (aget res 0))))

(defn read-double
  (^double [cu-buf]
   (let [res (double-array 1)]
     (memcpy-host! cu-buf res)
     (aget res 0)))
  (^double [hstream cu-buf]
   (let [res (double-array 1)]
     (memcpy-host! cu-buf res hstream)
     (aget res 0))))

(defn read-float
  (^double [cu-buf]
   (let [res (float-array 1)]
     (memcpy-host! cu-buf res)
     (aget res 0)))
  (^double [hstream cu-buf]
   (let [res (float-array 1)]
     (memcpy-host! cu-buf res hstream)
     (aget res 0))))
