;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.clojurecuda.toolbox-test
  (:require [midje.sweet :refer [facts => roughly]]
            [uncomplicate.commons
             [core :refer [release with-release]]
             [utils :refer [direct-buffer put-float count-groups]]]
            [uncomplicate.clojurecuda
             [core :refer :all]
             [info :refer :all]
             [toolbox :refer :all]]))

(init)

(let [dev (device)
      cnt-m 311
      cnt-n 9011
      cnt (* cnt-m cnt-n)
      program-source (str (slurp "src/cuda/uncomplicate/clojurecuda/kernels/reduction.cu") "\n"
                          (slurp "test/cuda/uncomplicate/clojurecuda/kernels/toolbox-test.cu"))]

  (with-context (context dev)
    (with-release [wgs (max-block-dim-x dev)
                   prog (compile! (program program-source)
                                  ["-DREAL=float" "-DACCUMULATOR=double" "-arch=compute_30"
                                   (format "-DWGS=%d" wgs)])
                   modl (module prog)
                   data (let [d (direct-buffer (* cnt Float/BYTES))]
                          (dotimes [n cnt]
                            (put-float d n (float n)))
                          d)
                   cu-data (mem-alloc (* cnt Float/BYTES))
                   sum-reduction-horizontal (function modl "sum_reduction_horizontal")
                   sum-horizontal (function modl "sum_reduce_horizontal")]

      (memcpy-host! data cu-data)

      (let [acc-size (* Double/BYTES (max 1 (count-groups wgs cnt)))]
        (with-release [sum-reduction-kernel (function modl "sum_reduction")
                       sum-reduce-kernel (function modl "sum_reduce")
                       cu-acc (mem-alloc acc-size)]
          (facts
           "Test 1D reduction."
           (launch-reduce! nil sum-reduce-kernel sum-reduction-kernel [cu-acc cu-data] [cu-acc] cnt wgs)
           (read-double cu-acc) => 3926780329410.0)))

      (let [wgs-m 64
            wgs-n 16
            acc-size (* Double/BYTES (max 1 (* cnt-m (count-groups wgs-n cnt-n))))
            res (double-array cnt-m)]
        (with-release [sum-reduction-horizontal (function modl "sum_reduction_horizontal")
                       sum-reduce-horizontal (function modl "sum_reduce_horizontal")
                       cu-acc (mem-alloc acc-size)]
          (facts
           "Test horizontal 2D reduction."
           (launch-reduce! nil sum-reduce-horizontal sum-reduction-horizontal
                           [cu-acc cu-data] [cu-acc] cnt-m cnt-n wgs-m wgs-n)
           (memcpy-host! cu-acc res)
           (apply + (seq res)) => (roughly 3.92678032941E12))))

      (let [wgs-m 64
            wgs-n 16
            acc-size (* Double/BYTES (max 1 (* cnt-n (count-groups wgs-m cnt-m))))
            res (double-array cnt-n)]
        (with-release [sum-reduction-vertical (function modl "sum_reduction_vertical")
                       sum-reduce-vertical (function modl "sum_reduce_vertical")
                       cu-acc (mem-alloc acc-size)]
          (facts
           "Test vertical 2D reduction."
           (launch-reduce! nil sum-reduce-vertical sum-reduction-vertical
                           [cu-acc cu-data] [cu-acc] cnt-n cnt-m wgs-n wgs-m)
           (memcpy-host! cu-acc res)
           (apply + (seq res)) => (roughly 3.92678032941E12)))))))
