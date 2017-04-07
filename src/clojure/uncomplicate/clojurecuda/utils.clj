;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
  uncomplicate.clojurecuda.utils
  "Utility functions used as helpers in other ClojureCUDA namespaces.
  The user of the ClojureCUDA library would probably not need to use
  any of the functions defined here."
  (:require [uncomplicate.commons.utils :as cu]
            [uncomplicate.clojurecuda.constants :refer [dec-error]])
  (:import clojure.lang.ExceptionInfo
           [java.nio ByteBuffer DirectByteBuffer]))

(defn error
  "Converts an CUDA error code to an [ExceptionInfo]
  (http://clojuredocs.org/clojure.core/ex-info)
  with richer, user-friendly information.

  Accepts a long `err-code` that should be one of the codes defined in
  CUDA standard, and an optional `details` argument that could be
  anything that you think is informative.

  See the available codes in the source of [[constants/dec-error]].
  Also see the discussion about

  Examples:

      (error 0) => an ExceptionInfo instance
      (error -5 {:comment \"Why here?\"\"}) => an ExceptionInfo instance
  "
  ([^long err-code details]
   (let [err (dec-error err-code)]
     (ex-info (format "CUDA error: %s." err)
              {:name err :code err-code :type :cuda-error :details details})))
  ([err-code]
   (error err-code nil)))

(defmacro with-check
  "Evaluates `form` if `err-code` is not zero (`CUDA_SUCCESS`), otherwise throws
  an appropriate `ExceptionInfo` with decoded informative details.
  It helps fith JCuda methods that return error codes directly, while
  returning computation results through side-effects in arguments.

  Example:

      (with-check (some-jcuda-call-that-returns-error-code) result)
  "
  ([err-code form]
   `(cu/with-check error ~err-code ~form)))

(defmacro with-check-arr
  "Evaluates `form` if the integer in the `err-code` primitive int array is `0`,
  Otherwise throws an exception corresponding to the error code.
  Similar to [[with-check]], but with the error code being held in an array instead
  of being a primitive number. It helps with JCuda methods that return results
  directly, and signal errors through side-effects in a primitive array argument.

      (let [err (int-array 1)
            res (some-jcuda-call err)]
         (with-checl-arr err res))
  "
  [err-code form]
  `(with-check (aget (ints ~err-code) 0) ~form))

(defmacro maybe
  "Evaluates form in try/catch block; if a CUDA-related exception is caught,
  substitutes the result with the [ExceptionInfo](http://clojuredocs.org/clojure.core/ex-info) object."
  [form]
  `(try ~form
         (catch ExceptionInfo ex-info#
           (if (= :cuda-error (:type (ex-data ex-info#)))
             (:name (ex-data ex-info#))
             (throw ex-info#)))))
