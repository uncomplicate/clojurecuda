;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.clojurecuda.nvrtc
  "Nvrtc related functions for runtime compilation of CUDA kernels.

  Where applicable, methods throw ExceptionInfo in case of errors thrown by the CUDA driver.
  "
  (:require [uncomplicate.commons
             [core :refer [Releaseable release]]
             [utils :as cu]]
            [uncomplicate.clojurecuda
             [protocols :refer [ModuleLoad module-load]]
             [utils :refer [with-check error]]])
  (:import [jcuda.driver JCudaDriver CUmodule]
           [jcuda.nvrtc JNvrtc nvrtcProgram nvrtcResult]))

;; ======================== Nvrtc utils ============================================

(defn ^:private nvrtc-error
  "Converts an CUDA Nvrtc error code to an ExceptionInfo with richer, user-friendly information. "
  ([^long err-code details]
   (let [err (nvrtcResult/stringFor err-code)]
     (ex-info (format "NVRTC error: %s." err)
              {:name err :code err-code :type :nvrtc-error :details details})))
  ([err-code]
   (error err-code nil)))

(defmacro ^:private with-check-nvrtc
  "Evaluates `form` if `err-code` is not zero (`NVRTC_SUCCESS`), otherwise throws
  an appropriate `ExceptionInfo` with decoded informative details.
  It helps fith JCuda nvrtc methods that return error codes directly, while
  returning computation results through side-effects in arguments.
  "
  ([err-code form]
   `(cu/with-check nvrtc-error ~err-code ~form)))

;; ====================== Nvrtc program JIT ========================================

(defn program*
  "TODO"
  [name source-code source-headers include-names]
  (let [res (nvrtcProgram.)]
    (with-check-nvrtc
      (JNvrtc/nvrtcCreateProgram res source-code name (count source-headers) source-headers include-names)
      res)))

(defn program
  "TODO"
  ([name source-code headers]
   (program* name source-code (into-array String (take-nth 2 (next headers)))
             (into-array String (take-nth 2 headers))))
  ([name source-code]
   (program* name source-code nil nil))
  ([source-code]
   (program* nil source-code nil nil)))

(defn program-log
  "TODO"
  [^nvrtcProgram program]
  (let [res (make-array String 1)]
    (with-check-nvrtc (JNvrtc/nvrtcGetProgramLog program res) (aget ^objects res 0))))

(defn compile*
  "TODO"
  ([^nvrtcProgram program options]
   (let [err (JNvrtc/nvrtcCompileProgram program (count options) options)]
     (if (= nvrtcResult/NVRTC_SUCCESS err)
       program
       (throw (nvrtc-error err (program-log program)))))))

(defn compile!
  "TODO"
  ([^nvrtcProgram program options]
   (compile* program (into-array String options)))
  ([^nvrtcProgram program]
   (compile* program nil)))

(defn ptx
  "TODO"
  ^String [^nvrtcProgram program]
  (let [res (make-array String 1)]
    (with-check-nvrtc (JNvrtc/nvrtcGetPTX program res) (aget ^objects res 0))))

;; ====================== Protocol extensions ======================================

(extend-type nvrtcProgram
  Releaseable
  (release [p]
    (with-check-nvrtc (JNvrtc/nvrtcDestroyProgram p) true))
  ModuleLoad
  (module-load [data m]
    (with-check (JCudaDriver/cuModuleLoadData ^CUmodule m (ptx data)) m)))
