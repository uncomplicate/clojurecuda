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
  (:require [uncomplicate.commons
             [core :refer [with-release]]
             [utils :refer [count-groups]]]
            [uncomplicate.clojure-cpp
             :refer [byte-pointer get-long get-int get-double get-float]]
            [uncomplicate.clojurecuda.core :refer :all])
  (:import org.bytedeco.javacpp.PointerPointer))

(defn launch-reduce!
  ([hstream main-kernel reduction-kernel main-params reduction-params n local-n]
   (let [main-params (if (instance? PointerPointer main-params)
                       (set-parameter! main-params 0 n)
                       (apply parameters n main-params))
         reduction-params (if (instance? PointerPointer reduction-params)
                            reduction-params
                            (apply parameters Integer/MAX_VALUE reduction-params))]
     (launch! main-kernel (grid-1d n local-n) hstream main-params)
     (loop [global-size (count-groups local-n n)]
       (when (< 1 global-size)
         (launch! reduction-kernel (grid-1d global-size local-n) hstream
                  (set-parameter! reduction-params 0 global-size))
         (recur (count-groups local-n global-size))))
     hstream))
  ([hstream main-kernel reduction-kernel main-params reduction-params m n local-m local-n]
   (let [main-params (if (instance? PointerPointer main-params)
                       (set-parameters! main-params 0 m n)
                       (apply parameters m n main-params))
         reduction-params (if (instance? PointerPointer reduction-params)
                            reduction-params
                            (apply parameters Integer/MAX_VALUE Integer/MAX_VALUE reduction-params))]
     (if (or (< 1 (long local-n)) (<= (long local-n) (long n)))
       (loop [hstream (launch! main-kernel (grid-2d m n local-m local-n) hstream main-params)
              global-size (count-groups local-n n)]
         (if (= 1 global-size)
           hstream
           (recur (launch! reduction-kernel (grid-2d m global-size local-m local-n) hstream
                           (set-parameters! reduction-params 0 m global-size))
                  (count-groups local-n global-size))))
       (throw (IllegalArgumentException.
               (format "local-n %d would cause infinite recursion for n:%d." local-n n)))))))

(defn read-int
  (^long [cu-buf]
   (with-release [res (byte-pointer Integer/BYTES)]
     (memcpy-host! cu-buf res)
     (get-int res 0)))
  (^long [hstream cu-buf]
   (with-release [res (byte-pointer Integer/BYTES)]
     (memcpy-host! cu-buf res hstream)
     (get-int res 0))))

(defn read-long
  (^long [cu-buf]
   (with-release [res (byte-pointer Long/BYTES)]
     (memcpy-host! cu-buf res)
     (get-long res 0)))
  (^long [hstream cu-buf]
   (with-release [res (byte-pointer Long/BYTES)]
     (memcpy-host! cu-buf res hstream)
     (get-long res 0))))

(defn read-double
  (^double [cu-buf]
   (with-release [res (byte-pointer Double/BYTES)]
     (memcpy-host! cu-buf res)
     (get-double res 0)))
  (^double [hstream cu-buf]
   (with-release [res (byte-pointer Double/BYTES)]
     (memcpy-host! cu-buf res hstream)
     (get-double res 0))))

(defn read-float
  (^double [cu-buf]
   (with-release [res (byte-pointer Float/BYTES)]
     (memcpy-host! cu-buf res)
     (get-float res 0)))
  (^double [hstream cu-buf]
   (with-release [res (byte-pointer Float/BYTES)]
     (memcpy-host! cu-buf res hstream)
     (get-float res 0))))
