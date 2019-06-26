;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
  uncomplicate.clojurecuda.internal.utils
  "Utility functions used as helpers in other ClojureCUDA namespaces.
  The user of the ClojureCUDA library would probably not need to use
  any of the functions defined here."
  (:require [uncomplicate.commons.utils :as cu])
  (:import clojure.lang.ExceptionInfo
           jcuda.driver.CUresult))

;; ============= Error Codes ===================================================

(defn error
  "Converts an CUDA error code to an [ExceptionInfo]
  (http://clojuredocs.org/clojure.core/ex-info)
  with richer, user-friendly information.

  Accepts a long `err-code` that should be one of the codes defined in
  CUDA standard, and an optional `details` argument that could be
  anything that you think is informative.

  Also see the discussion about

  Examples:

      (error 0) => an ExceptionInfo instance
      (error -5 {:comment \"Why here?\"\"}) => an ExceptionInfo instance
  "
  ([^long err-code details]
   (let [err (CUresult/stringFor err-code)]
     (ex-info (format "CUDA error: %s." err)
              {:name err :code err-code :type :cuda-error :details details})))
  ([^long err-code]
   (error err-code nil)))

(defmacro with-check
  "Evaluates `form` if `status` is zero (`CUDA_SUCCESS`), otherwise throws
  an appropriate `ExceptionInfo` with decoded informative details.
  It helps fith JCuda methods that return error codes directly, while
  returning computation results through side-effects in arguments.

  Example:

      (with-check (some-jcuda-call-that-returns-error-code) result)
  "
  ([status form]
   `(cu/with-check error ~status ~form))
  ([status details form]
   `(let [status# ~status]
      (if (= 0 status#)
        ~form
        (throw (error status# ~details))))))

(defmacro maybe
  "Evaluates form in try/catch block; if a CUDA-related exception is caught,
  substitutes the result with the String identifying the error."
  [form]
  `(try ~form
         (catch ExceptionInfo ex-info#
           (if (= :cuda-error (:type (ex-data ex-info#)))
             (:name (ex-data ex-info#))
             (throw ex-info#)))))
