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

(defn count-blocks ^long [^long max-block-size ^long n]
  (if (< max-block-size n)
    (quot (+ n (dec max-block-size)) max-block-size)
    1))

(let [pointer-arr-class (class (make-parameters 0))]

  (defn launch-reduce!
    ([hstream main-kernel reduction-kernel main-params reduction-params n block-n]
     (let [main-params (if (instance? pointer-arr-class main-params)
                         (set-parameter! main-params 0 n)
                         (apply parameters n main-params))
           reduction-params (if (instance? pointer-arr-class reduction-params)
                              reduction-params
                              (apply parameters Integer/MAX_VALUE reduction-params))]
       (launch! main-kernel (grid-1d n block-n) hstream main-params)
       (loop [global-size (count-blocks block-n n)]
         (when (< 1 global-size)
           (launch! reduction-kernel (grid-1d global-size block-n) hstream
                    (set-parameter! reduction-params 0 global-size))
           (recur (count-blocks block-n global-size))))))
    ([hstream main-kernel reduction-kernel main-params reduction-params n]
     (launch-reduce! hstream main-kernel reduction-kernel main-params reduction-params n 1024))
    ([hstream main-kernel reduction-kernel main-params reduction-params m n block-m block-n & [wgs-m wgs-n]]
     (let [main-params (if (instance? pointer-arr-class main-params)
                         (set-parameters! main-params 0 m n)
                         (apply parameters m n main-params))
           reduction-params (if (instance? pointer-arr-class reduction-params)
                              reduction-params
                              (apply parameters Integer/MAX_VALUE Integer/MAX_VALUE reduction-params))]
       (launch! main-kernel (grid-2d m n block-m block-n) hstream main-params)
       (let [[m n block-m block-n] (if (and wgs-m wgs-n)
                                     [n (count-blocks block-m m) wgs-m wgs-n]
                                     [m (count-blocks block-n n) block-m block-n])]
         (if (or (< 1 ^long block-n) (= 1 ^long n))
           (loop [n ^long n]
             (when (< 1 n)
               (launch! reduction-kernel (grid-2d m n block-m block-n) hstream
                        (set-parameters! reduction-params 0 m n))
               (recur (count-blocks block-n n))))
           (throw (IllegalArgumentException.
                   (format "block-n %d would cause infinite recursion for n:%d." block-n n)))))))))

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
